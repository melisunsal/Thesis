# get_attention_hooked.py
"""
Extract attention scores from a trained ProcessTransformer model.

This script:
1. Loads a trained model checkpoint
2. Hooks into the MultiHeadAttention layer to capture attention scores
3. Performs a single forward pass to get both logits and attention
4. Saves predictions, attention scores, and metadata for downstream analysis

Key fixes from original:
- Eliminated global state (CURRENT_KEY_MASK)
- Single forward pass for both logits and attention
- Instance-based mask passing to hooked MHA
"""

import json
import os
import argparse
import numpy as np
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
    """Build the ProcessTransformer model architecture."""
    return transformer.get_next_activity_model(
        max_case_length=maxlen, vocab_size=vocab_size, output_dim=num_out
    )


def find_block(model):
    """Find the TransformerBlock layer in the model."""
    try:
        return model.get_layer("transformer_block")
    except Exception:
        for layer in model.layers:
            if layer.__class__.__name__.lower().startswith("transformerblock"):
                return layer
        raise RuntimeError("TransformerBlock could not be found in model.")


def iter_submodules(layer):
    """Iterate through all submodules of a layer."""
    visited, stack = set(), [layer]
    while stack:
        cur = stack.pop()
        if id(cur) in visited:
            continue
        visited.add(id(cur))
        yield cur

        if hasattr(cur, "layers"):
            stack.extend(list(cur.layers))
        if hasattr(cur, "_self_tracked_trackables"):
            stack.extend(list(cur._self_tracked_trackables))


def hook_mha_instance(mha_layer: tf.keras.layers.MultiHeadAttention):
    """
    Hook into MultiHeadAttention to capture attention scores.

    Uses instance attribute (_current_key_mask) instead of global state
    for thread-safety and clarity.
    """
    orig_call = mha_layer.call
    mha_layer._current_key_mask = None  # Instance attribute for mask
    mha_layer.last_scores = None  # Will store attention scores

    def wrapped_call(self, query, value, key=None, attention_mask=None, **kwargs):
        # Enable causal masking and attention score return
        kwargs["use_causal_mask"] = True
        kwargs["return_attention_scores"] = True

        # Build attention mask from instance attribute
        current_mask = getattr(self, "_current_key_mask", None)

        if current_mask is not None:
            cur_B = tf.shape(query)[0]
            Tq = tf.shape(query)[-2]   # query length
            Tk = tf.shape(value)[-2]   # key length

            # Slice mask to current batch and sequence lengths
            base = current_mask  # [B, T], True = real token
            qv = base[:cur_B, :Tq]  # [B, Tq]
            kv = base[:cur_B, :Tk]  # [B, Tk]

            # Expand to 3D and compute intersection
            q3 = tf.expand_dims(qv, axis=-1)   # [B, Tq, 1]
            k3 = tf.expand_dims(kv, axis=1)    # [B, 1, Tk]
            keep_qk = tf.logical_and(q3, k3)   # [B, Tq, Tk]

            # Create causal mask (lower triangular)
            causal = tf.linalg.band_part(
                tf.ones((Tq, Tk), dtype=tf.bool), -1, 0
            )
            causal = tf.broadcast_to(causal, [cur_B, Tq, Tk])

            # Combined mask: (query & key valid) AND causal
            attention_mask = tf.logical_and(keep_qk, causal)

        out = orig_call(query, value, key=key, attention_mask=attention_mask, **kwargs)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            y, scores = out
            self.last_scores = scores
            return y
        else:
            self.last_scores = None
            return out

    mha_layer.call = MethodType(wrapped_call, mha_layer)
    return mha_layer


def find_and_hook_first_mha_in_block(block):
    """Find and hook the first MultiHeadAttention layer in a transformer block."""
    # Check direct attributes first
    for attr_name, obj in vars(block).items():
        if isinstance(obj, tf.keras.layers.MultiHeadAttention):
            hook_mha_instance(obj)
            return True, obj, attr_name

    # Check submodules
    for sub in iter_submodules(block):
        for attr_name, obj in vars(sub).items():
            if isinstance(obj, tf.keras.layers.MultiHeadAttention):
                hook_mha_instance(obj)
                return True, obj, f"{sub.name}.{attr_name}"

    return False, None, None


def forward_with_attention(model, hooked_mha, X, key_mask):
    """
    Single forward pass that captures both logits and attention scores.

    Parameters
    ----------
    model : tf.keras.Model
        The transformer model
    hooked_mha : MultiHeadAttention
        The hooked MHA layer (already modified by hook_mha_instance)
    X : np.ndarray
        Input token IDs [B, T]
    key_mask : np.ndarray
        Boolean mask [B, T], True = real token, False = PAD

    Returns
    -------
    logits : np.ndarray
        Model output logits [B, T, C] or [B, C]
    attention_scores : np.ndarray
        Attention scores [B, H, Tq, Tk]
    """
    # Set mask in hooked layer for access during forward pass
    hooked_mha._current_key_mask = tf.convert_to_tensor(key_mask, dtype=tf.bool)

    # Single forward pass - captures both logits and attention
    logits = model(X, training=False)

    # Get attention scores captured during forward pass
    attention_scores = getattr(hooked_mha, "last_scores", None)

    # Clean up
    hooked_mha._current_key_mask = None

    if attention_scores is None:
        raise RuntimeError(
            "Attention scores not captured. Ensure MHA layer supports return_attention_scores."
        )

    return logits.numpy(), attention_scores.numpy()


def prepare_batch_data(dl, X, use_df, xdict, ydict, maxlen):
    """
    Prepare batch data including decoded prefixes and case IDs.

    Returns
    -------
    decoded_prefixes : list of list of str
        Activity names for each prefix
    case_ids : list
        Case ID for each sample
    """
    # Get unshuffled data for mapping
    X_unshuf, _ = dl.prepare_data_next_activity(
        use_df, xdict, ydict, maxlen, shuffle=False
    )

    def signature(row):
        return tuple(int(t) for t in row if int(t) != 0)

    # Build signature to index mapping
    sig2idxs = defaultdict(deque)
    for i, row in enumerate(X_unshuf):
        sig2idxs[signature(row)].append(i)

    # Map shuffled batch back to original indices
    orig_idx_for_batch = []
    for row in X:
        sig = signature(row)
        if not sig2idxs[sig]:
            raise RuntimeError(
                "Row signature mismatch. Ensure same use_df/maxlen/xdict."
            )
        orig_idx_for_batch.append(sig2idxs[sig].popleft())
    orig_idx_for_batch = np.array(orig_idx_for_batch)

    # Decode tokens to activity names
    inv_xdict = {v: k for k, v in xdict.items()}

    # Find case ID column
    case_col = None
    for col_name in ["caseid", "case_id", "Case ID", "CaseID"]:
        if col_name in use_df.columns:
            case_col = col_name
            break
    if case_col is None:
        raise KeyError("Case ID column not found (tried: caseid, case_id, Case ID, CaseID)")

    decoded_prefixes = [
        [inv_xdict[int(t)] for t in seq if int(t) != 0]
        for seq in X
    ]
    case_ids = [use_df.iloc[int(i)][case_col] for i in orig_idx_for_batch]

    # Print batch info
    print("\n--- BATCH INFO ---")
    for i, seq in enumerate(X):
        orig_i = int(orig_idx_for_batch[i])
        df_index = use_df.index[orig_i]
        cid = case_ids[i]
        decoded = decoded_prefixes[i]
        print(f"Index {df_index:>4} | Case {cid:<5} | Prefix: {decoded}")

    return decoded_prefixes, case_ids


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1D numpy array."""
    x = x.astype("float64")
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if s == 0.0:
        return np.ones_like(ex) / max(len(ex), 1)
    return ex / s


def save_batch_predictions(
    out_dir: str,
    logits: np.ndarray,
    X: np.ndarray,
    ydict: dict,
    case_ids,
    prefix_activities,
    pad_id: int,
    top_k: int = 3,
):
    """Save batch predictions to JSON file."""
    inv_ydict = {v: k for k, v in ydict.items()}

    if logits.ndim == 3:
        B, T, C = logits.shape
        has_time_dim = True
    elif logits.ndim == 2:
        B, C = logits.shape
        T = None
        has_time_dim = False
    else:
        raise ValueError(
            f"Expected logits with shape [B, C] or [B, T, C], got {logits.shape}"
        )

    lengths = (X != pad_id).sum(axis=1)  # [B]

    preds = []
    for b in range(B):
        L = int(lengths[b])
        if L <= 0:
            continue

        # Select logits for last valid timestep
        if has_time_dim:
            if L > logits.shape[1]:
                raise ValueError(
                    f"Prefix length {L} exceeds logits time dim {logits.shape[1]}"
                )
            last_logits = logits[b, L - 1]
        else:
            last_logits = logits[b]

        probs = _softmax_1d(last_logits)

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_label = inv_ydict[top_idx]

        # Get top-k alternatives (excluding top prediction)
        sorted_idx = np.argsort(probs)[::-1]
        alternatives = []
        for idx in sorted_idx:
            idx = int(idx)
            if idx == top_idx:
                continue
            alternatives.append({
                "class_index": int(idx),
                "label": inv_ydict[idx],
                "prob": float(probs[idx]),
            })
            if len(alternatives) >= top_k:
                break

        # Make JSON-serializable
        cid = case_ids[b]
        if isinstance(cid, (np.generic, np.integer)):
            cid = int(cid)

        pref = prefix_activities[b]
        if isinstance(pref, np.ndarray):
            pref = pref.tolist()

        preds.append({
            "batch_index": int(b),
            "case_id": cid,
            "prefix_len": int(L),
            "prefix_activities": pref,
            "predicted_index": int(top_idx),
            "predicted_label": top_label,
            "pred_prob": float(top_prob),
            "top_alternatives": alternatives,
        })

    payload = {"predictions": preds}

    out_path = os.path.join(out_dir, "batch_predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved batch predictions to: {out_path}")


def save_attention_scores(out_dir: str, scores_np: np.ndarray):
    """Save attention scores to files."""
    np.save(os.path.join(out_dir, "block_mha_scores.npy"), scores_np)

    # Save CSV heatmaps for first sample's heads
    if scores_np.shape[0] >= 1 and scores_np.shape[1] >= 1:
        np.savetxt(
            os.path.join(out_dir, "block_mha_head0_heatmap.csv"),
            scores_np[0, 0],
            delimiter=",",
        )
    if scores_np.shape[0] >= 1 and scores_np.shape[1] >= 2:
        np.savetxt(
            os.path.join(out_dir, "block_mha_head1_heatmap.csv"),
            scores_np[0, 1],
            delimiter=",",
        )

    print(f"Saved attention scores to: {out_dir}")


def main():
    """Main entry point for attention extraction."""
    ap = argparse.ArgumentParser(
        description="Extract attention scores from ProcessTransformer"
    )
    ap.add_argument("--dataset", default="BPIC2012-O", help="Dataset name")
    ap.add_argument("--ckpt_dir", default="./models", help="Model checkpoint directory")
    ap.add_argument("--out_dir", default="./outputs", help="Output directory")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size to process")
    args = ap.parse_args()

    # Setup output directory
    outdir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(outdir, exist_ok=True)

    # ==========================================================================
    # 1) Load data
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Loading data for dataset: {args.dataset}")
    print(f"{'='*60}")

    dl = loader.LogsDataLoader(name=args.dataset)
    train_df, test_df, xdict, ydict, maxlen, vocab, num_out = dl.load_data(
        constants.Task.NEXT_ACTIVITY
    )

    use_df = test_df if len(test_df) > 0 else train_df
    X_all, _ = dl.prepare_data_next_activity(use_df, xdict, ydict, maxlen)
    X = X_all[: args.batch_size].astype("int32")

    # Prepare batch metadata
    decoded_prefixes, case_ids = prepare_batch_data(
        dl, X, use_df, xdict, ydict, maxlen
    )

    # ==========================================================================
    # 2) Build and load model
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Building and loading model")
    print(f"{'='*60}")

    vocab_size = len(vocab) if isinstance(vocab, (dict, list, tuple)) else int(vocab)
    model = build_model(maxlen, vocab_size, num_out)

    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
    latest = tf.train.latest_checkpoint(ckpt_dir)
    print(f"Latest checkpoint: {latest}")

    if latest:
        ckpt = tf.train.Checkpoint(model=model)
        status = ckpt.restore(latest)
        try:
            status.assert_consumed()
            print("All variables restored successfully.")
        except Exception:
            status.expect_partial()
            print("Partial restore (some variables unmatched).")

        model.load_weights(latest).expect_partial()
        print(f"Loaded weights from: {latest}")
    else:
        print(f"WARNING: No checkpoint found under {ckpt_dir}")
        print("Using randomly initialized weights!")

    # ==========================================================================
    # 3) Hook MHA layer BEFORE forward pass
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Hooking MultiHeadAttention layer")
    print(f"{'='*60}")

    block = find_block(model)
    ok, hooked_mha, where = find_and_hook_first_mha_in_block(block)
    if not ok:
        raise RuntimeError(
            "Could not find MultiHeadAttention instance in TransformerBlock"
        )
    print(f"Hooked MHA at: {where}")

    # ==========================================================================
    # 4) Single forward pass to get logits AND attention
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Running forward pass with attention capture")
    print(f"{'='*60}")

    PAD_ID = xdict.get("[PAD]") or xdict.get("<pad>") or xdict.get("PAD") or 0
    key_mask = (X != PAD_ID)  # [B, T], True = real token

    logits, scores_np = forward_with_attention(model, hooked_mha, X, key_mask)

    print(f"Logits shape: {logits.shape}")
    print(f"Attention scores shape: {scores_np.shape}")

    # ==========================================================================
    # 5) Save all outputs
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Saving outputs")
    print(f"{'='*60}")

    # Save predictions
    save_batch_predictions(
        out_dir=outdir,
        logits=logits,
        X=X,
        ydict=ydict,
        case_ids=case_ids,
        prefix_activities=decoded_prefixes,
        pad_id=PAD_ID,
        top_k=3,
    )

    # Save attention scores
    save_attention_scores(outdir, scores_np)

    # Save human-readable metadata
    save_batch_metadata(
        out_dir=outdir,
        prefix_texts=[" ".join(p) for p in decoded_prefixes],
        case_ids=case_ids,
        pad_token="[PAD]",
        align_right=True,
    )

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
