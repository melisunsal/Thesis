# get_attention_hooked.py
import os, argparse, numpy as np
import tensorflow as tf
from processtransformer import constants
from processtransformer.data import loader
import processtransformer.models.transformer as transformer
from types import MethodType
from tensorflow.keras import mixed_precision
from collections import defaultdict, deque

from explainers.shap_current_batch import run_kernel_shap_for_batch
from visualizeAttention.attention_viz_utils import save_batch_metadata
mixed_precision.set_global_policy("float32")
tf.config.run_functions_eagerly(True)


CURRENT_KEY_MASK = None  # shape [B, T], True = real token, False = [PAD]


def build_model(maxlen, vocab_size, num_out):
    return transformer.get_next_activity_model(
        max_case_length=maxlen, vocab_size=vocab_size, output_dim=num_out
    )


def find_block(model):
    try:
        return model.get_layer("transformer_block")
    except Exception:
        for l in model.layers:
            if l.__class__.__name__.lower().startswith("transformerblock"):
                return l
        raise RuntimeError("TransformerBlock couldn't be found.")


def iter_submodules(layer):
    visited, stack = set(), [layer]
    while stack:
        cur = stack.pop()
        if id(cur) in visited: continue
        visited.add(id(cur))
        yield cur

        if hasattr(cur, "layers"):
            stack.extend(list(cur.layers))
        if hasattr(cur, "_self_tracked_trackables"):
            stack.extend(list(cur._self_tracked_trackables))


def hook_mha_instance(mha_layer: tf.keras.layers.MultiHeadAttention):
    orig_call = mha_layer.call

    def wrapped_call(self, query, value, key=None, attention_mask=None, **kwargs):
        # KERAS'a causal'ı bırakmıyoruz; kendimiz oluşturacağız
        kwargs["use_causal_mask"] = False
        kwargs["return_attention_scores"] = True

        if CURRENT_KEY_MASK is not None:
            # Mevcut çağrının mini-batch ve uzunluklarını al
            cur_B = tf.shape(query)[0]
            Tq    = tf.shape(query)[-2]   # query length
            Tk    = tf.shape(value)[-2]   # key length

            base = tf.convert_to_tensor(CURRENT_KEY_MASK, dtype=tf.bool)  # [B, T]
            qv = base[:cur_B, :Tq]   # [B, Tq]  True=gerçek token
            kv = base[:cur_B, :Tk]   # [B, Tk]  True=gerçek token

            # Query ve Key geçerlilik maskelerini 3D'ye genişlet ve kesiştir
            q3 = tf.expand_dims(qv, axis=-1)   # [B, Tq, 1]
            k3 = tf.expand_dims(kv, axis=1)    # [B, 1,  Tk]
            keep_qk = tf.logical_and(q3, k3)   # [B, Tq, Tk]

            # Causal (alt üçgen) maskeyi biz üretelim
            causal = tf.linalg.band_part(tf.ones((Tq, Tk), dtype=tf.bool), -1, 0)  # [Tq, Tk]
            causal = tf.broadcast_to(causal, [cur_B, Tq, Tk])                      # [B, Tq, Tk]

            # Nihai maske: (query&key geçerli) ∧ (causal)
            full_mask = tf.logical_and(keep_qk, causal)  # [B, Tq, Tk]
            attention_mask = full_mask
        else:
            attention_mask = None

        out = orig_call(query, value, key=key, attention_mask=attention_mask, **kwargs)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            y, scores = out
            setattr(self, "last_scores", scores)
            return y
        else:
            setattr(self, "last_scores", None)
            return out

    mha_layer.call = MethodType(wrapped_call, mha_layer)
    return mha_layer


def find_and_hook_first_mha_in_block(block):
    for attr_name, obj in vars(block).items():
        if isinstance(obj, tf.keras.layers.MultiHeadAttention):
            hook_mha_instance(obj)
            return True, obj, attr_name
    for sub in iter_submodules(block):
        for attr_name, obj in vars(sub).items():
            if isinstance(obj, tf.keras.layers.MultiHeadAttention):
                hook_mha_instance(obj)
                return True, obj, f"{sub.name}.{attr_name}"
    return False, None, None


def print_batch(dl, X, use_df, xdict, ydict, maxlen):
    """
    Prints mapping back to DF indices and case ids, and sets
    global decoded_prefixes, case_ids (used for metadata saving).
    """
    global decoded_prefixes, case_ids

    X_unshuf, _ = dl.prepare_data_next_activity(
        use_df, xdict, ydict, maxlen, shuffle=False
    )

    def signature(row):
        return tuple(int(t) for t in row if int(t) != 0)

    sig2idxs = defaultdict(deque)
    for i, row in enumerate(X_unshuf):
        sig2idxs[signature(row)].append(i)

    orig_idx_for_batch = []
    for row in X:
        sig = signature(row)
        if not sig2idxs[sig]:
            raise RuntimeError("Eşleşmeyen satır imzası; aynı use_df/maxlen/xdict ile çağırdığından emin ol.")
        orig_idx_for_batch.append(sig2idxs[sig].popleft())
    orig_idx_for_batch = np.array(orig_idx_for_batch)

    inv_xdict = {v: k for k, v in xdict.items()}
    case_col = "caseid" if "caseid" in use_df.columns else ("case_id" if "case_id" in use_df.columns else None)
    if case_col is None:
        raise KeyError("case id kolonu bulunamadı (caseid / case_id).")

    decoded_prefixes = [[inv_xdict[int(t)] for t in seq if int(t) != 0] for seq in X]
    case_ids = [use_df.iloc[int(i)][case_col] for i in orig_idx_for_batch]

    print("\n--- BATCH (shuffled) - Doğru DF index + Case + decoded prefix ---")
    for i, seq in enumerate(X):
        orig_i = int(orig_idx_for_batch[i])
        df_index = use_df.index[orig_i]
        cid = use_df.iloc[orig_i][case_col]
        decoded = [inv_xdict[int(t)] for t in seq if int(t) != 0]
        print(f"Index {df_index:>4} | Case {cid:<5} | Prefix: {decoded}")


def main(with_shap_current_batch):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Helpdesk")
    ap.add_argument("--ckpt_dir", default="./models")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # Use a dataset-scoped output dir for all artifacts
    outdir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(outdir, exist_ok=True)

    # 1) data
    dl = loader.LogsDataLoader(name=args.dataset)
    train_df, test_df, xdict, ydict, maxlen, vocab, num_out = dl.load_data(
        constants.Task.NEXT_ACTIVITY
    )
    use_df = test_df if len(test_df) > 0 else train_df
    X_all, _ = dl.prepare_data_next_activity(use_df, xdict, ydict, maxlen)  # full set
    X = X_all[: args.batch_size].astype("int32")  # batch to explain

    print_batch(dl, X, use_df, xdict, ydict, maxlen)

    # 2) model + weights
    vocab_size = len(vocab) if isinstance(vocab, (dict, list, tuple)) else int(vocab)
    model = build_model(maxlen, vocab_size, num_out)

    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)  # ./models/Helpdesk
    latest = tf.train.latest_checkpoint(ckpt_dir)  # e.g., ".../next_activity_ckpt"
    print("latest ckpt:", latest)  # must NOT be None

    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(latest)
    try:
        status.assert_consumed()
        print("✅ All variables restored.")
    except Exception:
        status.expect_partial()
        print("ℹ️ Partial restore (some vars unmatched).")

    if latest:
        print(f"[info] loading weights from: {latest}")
        model.load_weights(latest).expect_partial()
    else:
        print(f"[warn] no checkpoint found under {ckpt_dir}; using randomly initialized weights.")

    # Forward once to get logits (for targets)
    logits = model(X, training=False).numpy()  # (B, T, C)

    # 3) prefix metadata for explanation alignment
    global CURRENT_KEY_MASK
    PAD_ID = 0  # ya da: PAD_ID = xdict.get("<pad>", 0)
    CURRENT_KEY_MASK = (X != PAD_ID)  # [B, T] bool

    # 4) hook the first MHA inside the transformer block
    block = find_block(model)
    ok, hooked_mha, where = find_and_hook_first_mha_in_block(block)
    if not ok:
        raise RuntimeError(
            "TransformerBlock içinde Keras MultiHeadAttention instance'ı bulunamadı."
        )
    print(f"Hooked MHA at: {where}")

    # 5) run once more to fill last_scores
    _ = model.predict(X, verbose=0)

    scores = getattr(hooked_mha, "last_scores", None)
    if scores is None:
        raise RuntimeError(
            "last_scores boş geldi. MHA call return_attention_scores parametresini kabul etmedi mi?"
        )

    scores_np = scores.numpy()  # [B, H, Tq, Tk]
    np.save(os.path.join(outdir, "block_mha_scores.npy"), scores_np)

    # save a couple of CSVs only if shapes allow
    if scores_np.shape[0] >= 1 and scores_np.shape[1] >= 1:
        np.savetxt(
            os.path.join(outdir, "block_mha_head0_heatmap.csv"),
            scores_np[0, 0],
            delimiter=",",
        )
    if scores_np.shape[0] >= 1 and scores_np.shape[1] >= 2:
        np.savetxt(
            os.path.join(outdir, "block_mha_head1_heatmap.csv"),
            scores_np[0, 1],
            delimiter=",",
        )

    print(f"✅ Attention scores saved under: {outdir}")

    # 6) save human-readable batch metadata
    save_batch_metadata(
        out_dir=outdir,
        prefix_texts=[" ".join(p) for p in decoded_prefixes],
        case_ids=case_ids,
        pad_token="[PAD]",
        align_right=True,
    )

    run_dir = os.path.join("outputs", args.dataset, "shap")  # or however you name runs
    xdict_json = os.path.join("datasets", args.dataset, "processed", "xdict.json")  # adjust if needed

    # Optional: choose a small background set (faster)
    X_bg = X[:min(16, X.shape[0])]

    _ = run_kernel_shap_for_batch(
        model,
        X_batch=X,
        xdict = xdict,
        run_dir=run_dir,
        background_X=X_bg,
        nsamples="auto",  # can set to 200–500 for more stable values
        link_logit=True,  # explain logits
        class_mode="predicted",  # explain the predicted next activity
    )


if __name__ == "__main__":
    main(False)
