# get_attention_hooked.py
import os, argparse, numpy as np
import tensorflow as tf
from processtransformer import constants
from processtransformer.data import loader
import processtransformer.models.transformer as transformer
from types import MethodType
from tensorflow.keras import mixed_precision
from collections import defaultdict, deque

from visualizeAttention.attention_viz_utils import save_batch_metadata

mixed_precision.set_global_policy("float32")
tf.config.run_functions_eagerly(True)


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
    """Bu MHA instance'ının call'unu runtime'da sarmalar: attention skorlarını kaydeder."""
    orig_call = mha_layer.call

    def wrapped_call(self, query, value, key=None, attention_mask=None, **kwargs):
        # Keras MHA call çoğu sürümde return_attention_scores parametresini kabul eder
        kwargs["return_attention_scores"] = True
        out = orig_call(query, value, key=key, attention_mask=attention_mask, **kwargs)
        # Bazı sürümlerde out=(y, scores), bazılarında y yalnız tensör olabilir (ama return_attention_scores=True ise tuple olur)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            y, scores = out
            setattr(self, "last_scores", scores)
            return y
        else:
            # güvenlik: yine de last_scores yoksa None bırak
            setattr(self, "last_scores", None)
            return out

    # MethodType ile instance üzerine bağla
    mha_layer.call = MethodType(wrapped_call, mha_layer)
    return mha_layer  # referans geri veriyoruz


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
    global decoded_prefixes, case_ids

    X_unshuf, _ = dl.prepare_data_next_activity(
        use_df, xdict, ydict, maxlen, shuffle=False
    )

    # 2) PAD(0) hariç tokenlardan imza oluştur
    def signature(row):
        return tuple(int(t) for t in row if int(t) != 0)

    # 3) Unshuffled tarafta imza -> orijinal index kuyruğu
    sig2idxs = defaultdict(deque)
    for i, row in enumerate(X_unshuf):
        sig2idxs[signature(row)].append(i)

    # 4) Batch'teki (shuffle'lı) satırların orijinal DF index’ini bul
    orig_idx_for_batch = []
    for row in X:
        sig = signature(row)
        if not sig2idxs[sig]:
            raise RuntimeError("Eşleşmeyen satır imzası; aynı use_df/maxlen/xdict ile çağırdığından emin ol.")
        orig_idx_for_batch.append(sig2idxs[sig].popleft())
    orig_idx_for_batch = np.array(orig_idx_for_batch)

    # 5) Doğru DF index + case id + decoded prefix bas
    inv_xdict = {v: k for k, v in xdict.items()}
    case_col = "caseid" if "caseid" in use_df.columns else ("case_id" if "case_id" in use_df.columns else None)
    if case_col is None:
        raise KeyError("case id kolonu bulunamadı (caseid / case_id).")

    decoded_prefixes = [[inv_xdict[int(t)] for t in seq if int(t) != 0] for seq in X]
    case_ids = [use_df.iloc[int(i)][case_col] for i in orig_idx_for_batch]

    print("\n--- BATCH (shuffled) - Doğru DF index + Case + decoded prefix ---")
    for i, seq in enumerate(X):
        orig_i = int(orig_idx_for_batch[i])  # orijinal (unshuffled) DF sıra numarası
        df_index = use_df.index[orig_i]  # DataFrame index (CSV index’i)
        cid = use_df.iloc[orig_i][case_col]  # Case ID
        decoded = [inv_xdict[int(t)] for t in seq if int(t) != 0]
        print(f"Index {df_index:>4} | Case {cid:<5} | Prefix: {decoded}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Helpdesk")
    ap.add_argument("--ckpt_dir", default="./models")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--batch_size", type=int, default=32)

    args = ap.parse_args()

    # 1) veri
    dl = loader.LogsDataLoader(name=args.dataset)
    train_df, test_df, xdict, ydict, maxlen, vocab, num_out = dl.load_data(constants.Task.NEXT_ACTIVITY)
    use_df = test_df if len(test_df) > 0 else train_df
    X, _ = dl.prepare_data_next_activity(use_df, xdict, ydict, maxlen)
    X = X[:args.batch_size]

    print(X)
    print_batch(dl, X, use_df, xdict, ydict, maxlen)

    # 2) model + ağırlık
    model = build_model(maxlen, vocab, num_out)
    ckpt = os.path.join(args.ckpt_dir, args.dataset, "next_activity_ckpt")
    model.load_weights(ckpt).expect_partial()

    # 3) TransformerBlock’u bul ve iç MHA instance’ını HOOK’la
    block = find_block(model)
    ok, hooked_mha, where = find_and_hook_first_mha_in_block(block)
    if not ok:
        raise RuntimeError("TransformerBlock içinde Keras MultiHeadAttention instance'ı bulunamadı.")

    print(f"Hooked MHA at: {where}")

    # 4) ileri geçiş (hook, skorları dolduracak)
    _ = model.predict(X, verbose=0)

    # 5) skoru al ve kaydet
    scores = getattr(hooked_mha, "last_scores", None)
    if scores is None:
        raise RuntimeError("last_scores boş geldi. MHA call return_attention_scores parametresini kabul etmedi mi?")
    outdir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "block_mha_scores.npy"), scores.numpy())  # [B, heads, L, L]
    np.savetxt(os.path.join(outdir, "block_mha_head0_heatmap.csv"), scores.numpy()[0, 0], delimiter=",")
    np.savetxt(os.path.join(outdir, "block_mha_head1_heatmap.csv"), scores.numpy()[0, 1], delimiter=",")
    print(f"✅ Kaydedildi: {outdir}")

    save_batch_metadata(
        out_dir=outdir,
        prefix_texts=[" ".join(p) for p in decoded_prefixes],  # satır başına "register classify ..." gibi
        case_ids=case_ids,
        pad_token="[PAD]",
        align_right=True,  # skorlar sağa hizalı LxL ise etiketleri de sağa hizala
    )


if __name__ == "__main__":
    main()
