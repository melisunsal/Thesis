import argparse
import os
import re
import json
import random
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("--repo_root", default="/Users/Q671967/PycharmProjects/Thesis", help="Repo root")
ap.add_argument("--dataset", default="BPIC2012-O", help="Dataset name")
ap.add_argument("--prefix_index", help="Index of longest prefix")
args = ap.parse_args()


REPO_ROOT = args.repo_root
MODELS_ROOT = os.path.join(REPO_ROOT, "models")
DATASET = args.dataset
BATCH_INDEX = int(args.prefix_index)

BATCH_PRED_PATH = os.path.join(REPO_ROOT, "outputs", DATASET, "batch_predictions.json")
ATTN_NPY_PATH = os.path.join(REPO_ROOT, "outputs", DATASET, "block_mha_scores.npy")

OUT_DIR = os.path.join(REPO_ROOT, "outputs", DATASET, f"shap_pad_replace_shap_pkg_batch_{BATCH_INDEX}")

EXPLAIN_LOGIT = True

N_PERMUTATIONS = 1000
RANDOM_SEED = 123

DO_LOO_CHECK = True
DO_TOPK_CHECK = True
TOPK = 10


# =============================================================================
# Utilities
# =============================================================================
def resolve_ckpt_prefix(dataset: str) -> str:
    ds_dir = os.path.join(MODELS_ROOT, dataset)
    ckpt_file = os.path.join(ds_dir, "checkpoint")

    if os.path.exists(ckpt_file):
        with open(ckpt_file, "r", encoding="utf-8") as f:
            txt = f.read()
        m = re.search(r'model_checkpoint_path:\s*"([^"]+)"', txt)
        if m:
            path_in_file = m.group(1)
            return path_in_file if os.path.isabs(path_in_file) else os.path.join(ds_dir, path_in_file)

    return os.path.join(ds_dir, "next_activity_ckpt")


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def pad_left_to_max(tokens: np.ndarray, max_len: int, pad_id: int) -> np.ndarray:
    """
    Helper to left-pad a token sequence to max_len.
    (Kept for consistency: the model expects fixed-length, left-padded inputs.)
    """
    out = np.full((max_len,), pad_id, dtype=np.int32)
    if tokens.size == 0:
        return out
    if tokens.size > max_len:
        tokens = tokens[-max_len:]
    out[-tokens.size:] = tokens.astype(np.int32)
    return out


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman_corr_tie_aware(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        import pandas as pd
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        return pearson_corr(ra, rb)
    except Exception:
        def rank(x):
            order = np.argsort(x)
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(x), dtype=float)
            return r
        return pearson_corr(rank(a), rank(b))


def load_prefix_from_batch_predictions(path: str, batch_index: int):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    preds = obj["predictions"]
    for item in preds:
        if int(item.get("batch_index", -1)) == int(batch_index):
            return item["case_id"], item["prefix_activities"], item.get("predicted_label"), item.get("predicted_prob")

    raise ValueError(f"batch_index={batch_index} not found in {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Make randomness deterministic (SHAP permutation uses numpy/random internally)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    from processtransformer.data import loader
    from processtransformer.models import transformer
    from processtransformer import constants

    # 1) Load prefix from batch_predictions.json
    if not os.path.exists(BATCH_PRED_PATH):
        raise FileNotFoundError(f"Cannot find batch predictions at: {BATCH_PRED_PATH}")

    case_id, prefix_activities, stored_pred_label, stored_pred_prob = load_prefix_from_batch_predictions(
        BATCH_PRED_PATH, BATCH_INDEX
    )
    L = len(prefix_activities)

    print(f"Using batch_index={BATCH_INDEX}, case_id={case_id}, L={L}")
    print(f"Stored prediction: {stored_pred_label}, prob={stored_pred_prob}")

    # 2) Load dicts, model params
    dl = loader.LogsDataLoader(name=DATASET)
    train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = dl.load_data(constants.Task.NEXT_ACTIVITY)

    pad_token = getattr(constants, "PAD_TOKEN", "[PAD]")
    pad_id = int(x_word_dict.get(pad_token, 0))
    inv_x = {v: k for k, v in x_word_dict.items()}
    inv_y = {v: k for k, v in y_word_dict.items()}

    # 3) Tokenize prefix
    x_unpadded = np.array([x_word_dict[a] for a in prefix_activities], dtype=np.int32)
    feature_names = [f"E{i+1}:{inv_x.get(int(t), str(int(t)))}" for i, t in enumerate(x_unpadded)]

    # 4) Load model
    model = transformer.get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size,
        output_dim=num_output
    )
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    ckpt = resolve_ckpt_prefix(DATASET)
    print("Loading weights from:", ckpt)
    model.load_weights(ckpt).expect_partial()

    # 5) Determine target class using full prefix
    x_full_padded = pad_left_to_max(x_unpadded, max_case_length, pad_id).reshape(1, -1)
    logits_full = model.predict(x_full_padded, verbose=0)
    probs_full = softmax(logits_full, axis=-1)
    class_idx = int(np.argmax(probs_full[0]))
    pred_label = inv_y.get(class_idx, str(class_idx))
    pred_prob = float(probs_full[0, class_idx])
    print(f"Model prediction on full prefix: {pred_label} (p={pred_prob:.4f}), EXPLAIN_LOGIT={EXPLAIN_LOGIT}")

    # 6) Define value function on mask vectors (pad replacement, position-preserving)
    # masks: (n, L) with 0/1, keep where 1, replace with PAD where 0 (no shifting)
    def f_from_masks(masks: np.ndarray) -> np.ndarray:
        """Value function on binary masks using *pad replacement*.

        masks: (n, L) with 0/1, where 1 keeps the original event at that position
        and 0 replaces it with PAD *in-place* (no shifting). This preserves token
        positions inside the left-padded max_case_length input.
        """
        masks = np.asarray(masks)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)

        keep = masks > 0.5  # (n, L) bool
        n = keep.shape[0]

        # Start from the fully padded input (left pads + the original prefix at the end)
        X_batch = np.repeat(x_full_padded, repeats=n, axis=0)  # (n, max_case_length)

        # Replace masked-out positions inside the prefix window with PAD, without shifting.
        start = max_case_length - L  # where the prefix begins in the padded sequence
        tokens = np.repeat(x_unpadded.reshape(1, -1), repeats=n, axis=0)  # (n, L)
        X_batch[:, start:start + L] = np.where(keep, tokens, pad_id).astype(np.int32)

        logits = model.predict(X_batch, verbose=0)
        if EXPLAIN_LOGIT:
            return logits[:, class_idx].astype(np.float64)

        probs = softmax(logits, axis=-1)
        return probs[:, class_idx].astype(np.float64)

    # Baseline mask is all zeros (all PADs in prefix window)
    mask_empty = np.zeros((1, L), dtype=np.int8)
    base_val_true = float(f_from_masks(mask_empty)[0])

    # Full mask is all ones (full prefix)
    mask_full = np.ones((1, L), dtype=np.int8)
    fx_full_true = float(f_from_masks(mask_full)[0])

    # 7) SHAP permutation Shapley on mask space
    background = np.zeros((1, L), dtype=np.int8)
    masker = shap.maskers.Independent(background)

    max_evals = int((L + 1) * N_PERMUTATIONS)

    # Different SHAP versions expose permutation in different ways, try both
    try:
        explainer = shap.Explainer(f_from_masks, masker, algorithm="permutation")
        exp = explainer(mask_full, max_evals=max_evals)
    except Exception:
        explainer = shap.explainers.Permutation(f_from_masks, masker)
        try:
            exp = explainer(mask_full, max_evals=max_evals)
        except TypeError:
            # fallback if this SHAP version does not accept max_evals
            exp = explainer(mask_full)

    phi = np.array(exp.values[0], dtype=np.float64)  # (L,)
    base_val = float(np.array(exp.base_values).reshape(-1)[0])  # scalar

    fx_recon = float(base_val + phi.sum())
    add_err = float(fx_recon - fx_full_true)

    # Optional sanity print if SHAP base differs from true empty score
    base_diff = float(base_val - base_val_true)

    verif = {
        "phi_method": "shap_permutation_on_mask_space",
        "base_value_true_empty": base_val_true,
        "base_value_shap": base_val,
        "base_value_diff": base_diff,
        "f_full_true": fx_full_true,
        "f_reconstructed": fx_recon,
        "additivity_error_vs_true_full": add_err,
        "n_permutations_target": int(N_PERMUTATIONS),
        "max_evals": int(max_evals),
        "explain_logit": bool(EXPLAIN_LOGIT),
    }

    # 8) Verification: LOO and top-k
    loo = {}
    if DO_LOO_CHECK:
        deltas = np.zeros((L,), dtype=np.float64)
        for i in range(L):
            m = np.ones((1, L), dtype=np.int8)
            m[0, i] = 0
            deltas[i] = fx_full_true - float(f_from_masks(m)[0])

        loo = {
            "pearson_phi_vs_loo": pearson_corr(phi, deltas),
            "spearman_phi_vs_loo": spearman_corr_tie_aware(phi, deltas),
            "pearson_absphi_vs_absloo": pearson_corr(np.abs(phi), np.abs(deltas)),
            "spearman_absphi_vs_absloo": spearman_corr_tie_aware(np.abs(phi), np.abs(deltas)),
            "loo_deltas": deltas.tolist(),
        }

    topk = {}
    if DO_TOPK_CHECK:
        kmax = int(min(TOPK, L))
        order_pos = np.argsort(-phi)
        order_abs = np.argsort(-np.abs(phi))
        drops_pos = []
        drops_abs = []
        increases_neg = []

        for k in range(1, kmax + 1):
            m = np.ones((1, L), dtype=np.int8)
            m[0, order_pos[:k]] = 0
            drops_pos.append(float(fx_full_true - float(f_from_masks(m)[0])))

            m = np.ones((1, L), dtype=np.int8)
            m[0, order_abs[:k]] = 0
            drops_abs.append(float(fx_full_true - float(f_from_masks(m)[0])))

            order_neg = np.argsort(phi)
            m = np.ones((1, L), dtype=np.int8)
            m[0, order_neg[:k]] = 0
            increases_neg.append(float(float(f_from_masks(m)[0]) - fx_full_true))

        topk = {
            "kmax": kmax,
            "top1_positive": feature_names[int(order_pos[0])],
            "top1_abs": feature_names[int(order_abs[0])],
            "drops_topk_positive": drops_pos,
            "drops_topk_abs": drops_abs,
            "increases_remove_topk_negative": increases_neg,
        }

    # 9) Optional attention alignment (same as your file)
    attn_align = {}
    if os.path.exists(ATTN_NPY_PATH):
        try:
            A = np.load(ATTN_NPY_PATH)  # (N, H, T, T)
            if BATCH_INDEX < 0 or BATCH_INDEX >= A.shape[0]:
                raise ValueError("BATCH_INDEX out of range for attention npy")

            sample = A[BATCH_INDEX]  # (H, T, T)
            H, T1, T2 = sample.shape
            if T1 != T2:
                raise ValueError("Attention matrix not square")
            T = T1

            if L > T:
                raise ValueError(f"Prefix length L={L} larger than attention T={T}")

            start_pos = T - L
            last_pos = T - 1

            attn_last_per_head = sample[:, last_pos, start_pos:T]  # (H, L)
            attn_last = attn_last_per_head.mean(axis=0)            # (L,)

            s = float(attn_last.sum())
            if s > 0:
                attn_last = attn_last / s

            absphi = np.abs(phi)
            attn_align = {
                "attention_available": True,
                "attention_shape": [int(A.shape[0]), int(A.shape[1]), int(A.shape[2]), int(A.shape[3])],
                "attn_last_row_mean_heads": attn_last.tolist(),
                "pearson_attn_vs_absphi": pearson_corr(attn_last, absphi),
                "spearman_attn_vs_absphi": spearman_corr_tie_aware(attn_last, absphi),
            }

            k = int(min(5, L))
            top_attn = np.argsort(-attn_last)[:k].tolist()
            top_absphi = np.argsort(-absphi)[:k].tolist()
            overlap = len(set(top_attn).intersection(set(top_absphi)))

            attn_align.update({
                "topk": k,
                "topk_attn_indices_0based": top_attn,
                "topk_absphi_indices_0based": top_absphi,
                "topk_overlap_count": overlap,
                "topk_overlap_ratio": float(overlap) / float(k) if k > 0 else 0.0,
                "topk_attn_features": [feature_names[i] for i in top_attn],
                "topk_absphi_features": [feature_names[i] for i in top_absphi],
            })

        except Exception as e:
            attn_align = {"attention_available": False, "error": str(e)}
    else:
        attn_align = {"attention_available": False, "error": f"Missing file: {ATTN_NPY_PATH}"}

    # 10) Save JSON
    out = {
        "dataset": DATASET,
        "batch_index": BATCH_INDEX,
        "case_id": case_id,
        "prefix_activities": prefix_activities,
        "stored_predicted_label": stored_pred_label,
        "stored_predicted_prob": stored_pred_prob,
        "model_predicted_label": pred_label,
        "model_predicted_prob": pred_prob,
        "predicted_class_index": class_idx,
        "L": L,
        "feature_names": feature_names,
        "phi_pad_replace": phi.tolist(),
        "base_value": float(base_val),
        "f_full": float(fx_full_true),
        "verification": verif,
        "verification_loo": loo,
        "verification_topk": topk,
        "attention_alignment": attn_align,
        "settings": {
            "explain_logit": EXPLAIN_LOGIT,
            "n_permutations": N_PERMUTATIONS,
            "random_seed": RANDOM_SEED,
        }
    }

    with open(os.path.join(OUT_DIR, "shap_explanation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # 11) Plot (reuse your visualization style)
    explanation = shap.Explanation(
        values=np.array(phi, dtype=float),
        base_values=float(base_val),
        data=np.array(prefix_activities, dtype=object),
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Permutation Waterfall (pad-replace, mask-space), pred={pred_label} (batch_index={BATCH_INDEX})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_waterfall.png"), dpi=200)
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Permutation Bar (pad-replace, mask-space), pred={pred_label} (batch_index={BATCH_INDEX})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar.png"), dpi=200)
    plt.close()

    # 12) Print summary
    print("\n" + "=" * 80)
    print(f"Saved outputs to: {OUT_DIR}")
    print(f"True base v(empty): {base_val_true:.6f}")
    print(f"SHAP base value:    {base_val:.6f}  (diff={base_diff:.3e})")
    print(f"True full v(full):  {fx_full_true:.6f}")
    print(f"Reconstructed:      {fx_recon:.6f}")
    print(f"Additivity error:   {add_err:.3e}")
    if loo:
        print(f"LOO corr phi vs delta: pearson={loo['pearson_phi_vs_loo']:.3f}, spearman={loo['spearman_phi_vs_loo']:.3f}")
        print(f"LOO corr abs:         pearson={loo['pearson_absphi_vs_absloo']:.3f}, spearman={loo['spearman_absphi_vs_absloo']:.3f}")
    if topk:
        print(f"Top-1 positive: {topk['top1_positive']}")
        print(f"Top-1 abs:      {topk['top1_abs']}")
    if attn_align.get("attention_available"):
        print(f"Attn vs abs(phi): pearson={attn_align['pearson_attn_vs_absphi']:.3f}, spearman={attn_align['spearman_attn_vs_absphi']:.3f}")
        print(f"Top-{attn_align['topk']} overlap: {attn_align['topk_overlap_count']}/{attn_align['topk']}  "
              f"(ratio={attn_align['topk_overlap_ratio']:.2f})")
        print("Top attn:", attn_align["topk_attn_features"])
        print("Top abs(phi):", attn_align["topk_absphi_features"])
    else:
        print("Attention alignment skipped:", attn_align.get("error"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
