# get_attention_hooked.py
import os, argparse, numpy as np
import tensorflow as tf
from processtransformer import constants
from processtransformer.data import loader
import processtransformer.models.transformer as transformer
from types import MethodType
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mockLargerDataset")
    ap.add_argument("--ckpt_dir", default="./models")
    ap.add_argument("--out_dir", default="./attn_dump2")
    ap.add_argument("--batch_size", type=int, default=16)

    args = ap.parse_args()

    # 1) veri
    dl = loader.LogsDataLoader(name=args.dataset)
    train_df, test_df, xdict, ydict, maxlen, vocab, num_out = dl.load_data(constants.Task.NEXT_ACTIVITY)
    use_df = test_df if len(test_df) > 0 else train_df
    X, _ = dl.prepare_data_next_activity(use_df, xdict, ydict, maxlen)
    X = X[:args.batch_size]

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
    np.save(os.path.join(outdir, "block_mha_scores.npy"), scores.numpy())   # [B, heads, L, L]
    np.savetxt(os.path.join(outdir, "block_mha_head0_heatmap.csv"), scores.numpy()[0,0], delimiter=",")
    np.savetxt(os.path.join(outdir, "block_mha_head1_heatmap.csv"), scores.numpy()[0,1], delimiter=",")
    print(f"✅ Kaydedildi: {outdir}")


if __name__ == "__main__":
    main()
