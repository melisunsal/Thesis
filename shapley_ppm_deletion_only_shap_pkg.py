#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis using deletion-only semantics with the SHAP package's permutation explainer.

This script computes Shapley values by operating in mask-space:
- A mask vector of 0s and 1s indicates which tokens to keep
- Kept tokens are extracted in order and left-padded to max_len
- This matches true deletion semantics (removing tokens, not replacing with padding in-place)

The SHAP permutation explainer is used on this mask-space representation.
"""

import os
import re
import json
import random
import argparse
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG (defaults, can be overridden via command line)
# =============================================================================
DEFAULT_CONFIG = {
    "repo_root": ".",
    "dataset": "BPIC2012-O",
    "batch_index": 12,
    "explain_logit": True,
    "n_permutations": 1000,
    "random_seed": 123,
    "do_loo_check": True,
    "do_topk_check": True,
    "topk": 10,
}


# =============================================================================
# Utilities
# =============================================================================
def resolve_ckpt_prefix(models_root: str, dataset: str) -> str:
    ds_dir = os.path.join(models_root, dataset)
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
    Deletion semantics: take kept tokens in order, then left-pad to max_len.
    This preserves right-alignment consistent with the original data preparation.
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


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Kendall's tau correlation coefficient."""
    try:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(a, b)
        return float(tau)
    except ImportError:
        return float("nan")


def load_prefix_from_batch_predictions(path: str, batch_index: int):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    preds = obj["predictions"]
    for item in preds:
        if int(item.get("batch_index", -1)) == int(batch_index):
            return item["case_id"], item["prefix_activities"], item.get("predicted_label"), item.get("pred_prob")

    raise ValueError(f"batch_index={batch_index} not found in {path}")


# =============================================================================
# Main Analysis Function
# =============================================================================
def run_shap_deletion_analysis(
    dataset: str,
    batch_index: int,
    repo_root: str = ".",
    explain_logit: bool = True,
    n_permutations: int = 1000,
    random_seed: int = 123,
    do_loo_check: bool = True,
    do_topk_check: bool = True,
    topk: int = 10,
    out_dir: str = None,
) -> dict:
    """
    Run SHAP deletion-only analysis for a single sample.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., "BPIC2012-O")
    batch_index : int
        Index of sample in batch_predictions.json
    repo_root : str
        Repository root path
    explain_logit : bool
        If True, explain logits; if False, explain probabilities
    n_permutations : int
        Number of permutations for SHAP
    random_seed : int
        Random seed for reproducibility
    do_loo_check : bool
        Whether to perform leave-one-out verification
    do_topk_check : bool
        Whether to perform top-k removal verification
    topk : int
        Maximum k for top-k checks
    out_dir : str, optional
        Output directory (default: outputs/{dataset}/shap_deletion_batch_{batch_index})

    Returns
    -------
    dict
        Analysis results
    """
    # Setup paths
    models_root = os.path.join(repo_root, "models")
    batch_pred_path = os.path.join(repo_root, "outputs", dataset, "batch_predictions.json")
    attn_npy_path = os.path.join(repo_root, "outputs", dataset, "block_mha_scores.npy")

    if out_dir is None:
        out_dir = os.path.join(repo_root, "outputs", dataset, f"shap_deletion_batch_{batch_index}")

    os.makedirs(out_dir, exist_ok=True)

    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    from processtransformer.data import loader
    from processtransformer.models import transformer
    from processtransformer import constants

    # 1) Load prefix from batch_predictions.json
    if not os.path.exists(batch_pred_path):
        raise FileNotFoundError(f"Cannot find batch predictions at: {batch_pred_path}")

    case_id, prefix_activities, stored_pred_label, stored_pred_prob = load_prefix_from_batch_predictions(
        batch_pred_path, batch_index
    )
    L = len(prefix_activities)

    print(f"Using batch_index={batch_index}, case_id={case_id}, L={L}")
    print(f"Stored prediction: {stored_pred_label}, prob={stored_pred_prob}")

    # 2) Load dicts, model params
    dl = loader.LogsDataLoader(name=dataset)
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
    ckpt = resolve_ckpt_prefix(models_root, dataset)
    print("Loading weights from:", ckpt)
    model.load_weights(ckpt).expect_partial()

    # 5) Determine target class using full prefix
    x_full_padded = pad_left_to_max(x_unpadded, max_case_length, pad_id).reshape(1, -1)
    logits_full = model.predict(x_full_padded, verbose=0)
    probs_full = softmax(logits_full, axis=-1)
    class_idx = int(np.argmax(probs_full[0]))
    pred_label = inv_y.get(class_idx, str(class_idx))
    pred_prob = float(probs_full[0, class_idx])
    print(f"Model prediction on full prefix: {pred_label} (p={pred_prob:.4f}), EXPLAIN_LOGIT={explain_logit}")

    # 6) Define value function on mask vectors (deletion-only semantics)
    def f_from_masks(masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)

        keep = masks > 0.5  # bool
        n = keep.shape[0]

        X_batch = np.zeros((n, max_case_length), dtype=np.int32)
        for r in range(n):
            kept_tokens = x_unpadded[keep[r]]
            X_batch[r] = pad_left_to_max(kept_tokens, max_case_length, pad_id)

        logits = model.predict(X_batch, verbose=0)
        if explain_logit:
            return logits[:, class_idx].astype(np.float64)

        probs = softmax(logits, axis=-1)
        return probs[:, class_idx].astype(np.float64)

    # Baseline mask is all zeros (empty)
    mask_empty = np.zeros((1, L), dtype=np.int8)
    base_val_true = float(f_from_masks(mask_empty)[0])

    # Full mask is all ones (full)
    mask_full = np.ones((1, L), dtype=np.int8)
    fx_full_true = float(f_from_masks(mask_full)[0])

    # 7) SHAP permutation Shapley on mask space
    background = np.zeros((1, L), dtype=np.int8)
    masker = shap.maskers.Independent(background)

    max_evals = int((L + 1) * n_permutations)

    # Different SHAP versions expose permutation in different ways
    try:
        explainer = shap.Explainer(f_from_masks, masker, algorithm="permutation")
        exp = explainer(mask_full, max_evals=max_evals)
    except TypeError:
        explainer = shap.Explainer(f_from_masks, masker, algorithm="permutation")
        exp = explainer(mask_full)
    except Exception:
        explainer = shap.explainers.Permutation(f_from_masks, masker)
        try:
            exp = explainer(mask_full, max_evals=max_evals)
        except TypeError:
            exp = explainer(mask_full)

    phi = np.array(exp.values[0], dtype=np.float64)  # (L,)
    base_val = float(np.array(exp.base_values).reshape(-1)[0])  # scalar

    fx_recon = float(base_val + phi.sum())
    add_err = float(fx_recon - fx_full_true)
    base_diff = float(base_val - base_val_true)

    verif = {
        "phi_method": "shap_permutation_on_mask_space",
        "base_value_true_empty": base_val_true,
        "base_value_shap": base_val,
        "base_value_diff": base_diff,
        "f_full_true": fx_full_true,
        "f_reconstructed": fx_recon,
        "additivity_error_vs_true_full": add_err,
        "n_permutations_target": int(n_permutations),
        "max_evals": int(max_evals),
        "explain_logit": bool(explain_logit),
    }

    # 8) Verification: LOO and top-k
    loo = {}
    if do_loo_check:
        deltas = np.zeros((L,), dtype=np.float64)
        for i in range(L):
            m = np.ones((1, L), dtype=np.int8)
            m[0, i] = 0
            deltas[i] = fx_full_true - float(f_from_masks(m)[0])

        loo = {
            "pearson_phi_vs_loo": pearson_corr(phi, deltas),
            "spearman_phi_vs_loo": spearman_corr_tie_aware(phi, deltas),
            "kendall_phi_vs_loo": kendall_tau(phi, deltas),
            "pearson_absphi_vs_absloo": pearson_corr(np.abs(phi), np.abs(deltas)),
            "spearman_absphi_vs_absloo": spearman_corr_tie_aware(np.abs(phi), np.abs(deltas)),
            "loo_deltas": deltas.tolist(),
        }

    topk_results = {}
    if do_topk_check:
        kmax = int(min(topk, L))
        order_pos = np.argsort(-phi)
        order_abs = np.argsort(-np.abs(phi))
        order_neg = np.argsort(phi)

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

            m = np.ones((1, L), dtype=np.int8)
            m[0, order_neg[:k]] = 0
            increases_neg.append(float(float(f_from_masks(m)[0]) - fx_full_true))

        topk_results = {
            "kmax": kmax,
            "top1_positive_feature": feature_names[int(order_pos[0])],
            "top1_abs_feature": feature_names[int(order_abs[0])],
            "drops_topk_positive": drops_pos,
            "drops_topk_abs": drops_abs,
            "increases_remove_topk_negative": increases_neg,
        }

    # 9) Attention alignment
    attn_align = {}
    if os.path.exists(attn_npy_path):
        try:
            A = np.load(attn_npy_path)  # (N, H, T, T)
            if batch_index < 0 or batch_index >= A.shape[0]:
                raise ValueError("batch_index out of range for attention npy")

            sample = A[batch_index]  # (H, T, T)
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
                "pearson_attn_vs_phi": pearson_corr(attn_last, phi),
                "spearman_attn_vs_phi": spearman_corr_tie_aware(attn_last, phi),
                "kendall_attn_vs_phi": kendall_tau(attn_last, phi),
                "pearson_attn_vs_absphi": pearson_corr(attn_last, absphi),
                "spearman_attn_vs_absphi": spearman_corr_tie_aware(attn_last, absphi),
                "kendall_attn_vs_absphi": kendall_tau(attn_last, absphi),
            }

            k = int(min(5, L))
            top_attn = np.argsort(-attn_last)[:k].tolist()
            top_absphi = np.argsort(-absphi)[:k].tolist()
            top_phi_pos = np.argsort(-phi)[:k].tolist()

            overlap_abs = len(set(top_attn).intersection(set(top_absphi)))
            overlap_pos = len(set(top_attn).intersection(set(top_phi_pos)))

            attn_align.update({
                "topk": k,
                "topk_attn_indices": top_attn,
                "topk_absphi_indices": top_absphi,
                "topk_phi_positive_indices": top_phi_pos,
                "topk_overlap_attn_absphi": overlap_abs,
                "topk_overlap_ratio_attn_absphi": float(overlap_abs) / float(k) if k > 0 else 0.0,
                "topk_overlap_attn_phi_positive": overlap_pos,
                "topk_overlap_ratio_attn_phi_positive": float(overlap_pos) / float(k) if k > 0 else 0.0,
                "topk_attn_features": [feature_names[i] for i in top_attn],
                "topk_absphi_features": [feature_names[i] for i in top_absphi],
            })

        except Exception as e:
            attn_align = {"attention_available": False, "error": str(e)}
    else:
        attn_align = {"attention_available": False, "error": f"Missing file: {attn_npy_path}"}

    # 10) Build output dictionary
    out = {
        "dataset": dataset,
        "batch_index": batch_index,
        "case_id": case_id,
        "prefix_activities": prefix_activities,
        "prefix_length": L,
        "stored_predicted_label": stored_pred_label,
        "stored_predicted_prob": stored_pred_prob,
        "model_predicted_label": pred_label,
        "model_predicted_prob": pred_prob,
        "predicted_class_index": class_idx,
        "feature_names": feature_names,
        "shap_values": phi.tolist(),
        "base_value": float(base_val),
        "f_full": float(fx_full_true),
        "verification": verif,
        "verification_loo": loo,
        "verification_topk": topk_results,
        "attention_comparison": attn_align,
        "settings": {
            "explain_logit": explain_logit,
            "n_permutations": n_permutations,
            "random_seed": random_seed,
        }
    }

    # 11) Save JSON
    with open(os.path.join(out_dir, "shap_explanation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # 12) Generate plots
    explanation = shap.Explanation(
        values=np.array(phi, dtype=float),
        base_values=float(base_val),
        data=np.array(prefix_activities, dtype=object),
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Deletion Waterfall, pred={pred_label} (batch={batch_index})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_waterfall.png"), dpi=200)
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Deletion Bar, pred={pred_label} (batch={batch_index})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_bar.png"), dpi=200)
    plt.close()

    # Attention vs SHAP comparison plot
    if attn_align.get("attention_available") and "attn_last_row_mean_heads" in attn_align:
        attn = np.array(attn_align["attn_last_row_mean_heads"])
        absphi = np.abs(phi)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        axes[0].scatter(attn, absphi, alpha=0.7, s=60)
        for i in range(L):
            axes[0].annotate(f'E{i+1}', (attn[i], absphi[i]), fontsize=8, alpha=0.7)
        axes[0].set_xlabel("Attention Weight")
        axes[0].set_ylabel("|SHAP Value|")
        axes[0].set_title(
            f"Attention vs |SHAP|\n"
            f"Pearson={attn_align['pearson_attn_vs_absphi']:.3f}, "
            f"Spearman={attn_align['spearman_attn_vs_absphi']:.3f}"
        )

        # Side-by-side bar comparison
        x = np.arange(L)
        width = 0.35
        attn_norm = attn / attn.max() if attn.max() > 0 else attn
        absphi_norm = absphi / absphi.max() if absphi.max() > 0 else absphi
        axes[1].bar(x - width/2, attn_norm, width, label="Attention (normalized)")
        axes[1].bar(x + width/2, absphi_norm, width, label="|SHAP| (normalized)")
        axes[1].set_xlabel("Event Position")
        axes[1].set_ylabel("Normalized Importance")
        axes[1].set_title("Attention vs |SHAP| by Position")
        axes[1].legend()
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"E{i+1}" for i in range(L)], rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "attn_vs_shap.png"), dpi=200)
        plt.close()

    # 13) Print summary
    print("\n" + "=" * 80)
    print(f"Saved outputs to: {out_dir}")
    print(f"True base v(empty): {base_val_true:.6f}")
    print(f"SHAP base value:    {base_val:.6f}  (diff={base_diff:.3e})")
    print(f"True full v(full):  {fx_full_true:.6f}")
    print(f"Reconstructed:      {fx_recon:.6f}")
    print(f"Additivity error:   {add_err:.3e}")
    if loo:
        print(f"LOO corr phi vs delta: pearson={loo['pearson_phi_vs_loo']:.3f}, spearman={loo['spearman_phi_vs_loo']:.3f}")
        print(f"LOO corr abs:         pearson={loo['pearson_absphi_vs_absloo']:.3f}, spearman={loo['spearman_absphi_vs_absloo']:.3f}")
    if topk_results:
        print(f"Top-1 positive: {topk_results['top1_positive_feature']}")
        print(f"Top-1 abs:      {topk_results['top1_abs_feature']}")
    if attn_align.get("attention_available"):
        print(f"Attn vs phi:      pearson={attn_align['pearson_attn_vs_phi']:.3f}, spearman={attn_align['spearman_attn_vs_phi']:.3f}")
        print(f"Attn vs abs(phi): pearson={attn_align['pearson_attn_vs_absphi']:.3f}, spearman={attn_align['spearman_attn_vs_absphi']:.3f}")
        print(f"Top-{attn_align['topk']} overlap (attn vs |phi|): {attn_align['topk_overlap_attn_absphi']}/{attn_align['topk']}  "
              f"(ratio={attn_align['topk_overlap_ratio_attn_absphi']:.2f})")
        print("Top attn:", attn_align["topk_attn_features"])
        print("Top abs(phi):", attn_align["topk_absphi_features"])
    else:
        print("Attention alignment skipped:", attn_align.get("error"))
    print("=" * 80 + "\n")

    return out


# =============================================================================
# Batch Analysis
# =============================================================================
def run_batch_analysis(
    dataset: str,
    start_idx: int,
    end_idx: int,
    repo_root: str = ".",
    **kwargs
) -> dict:
    """Run analysis for multiple samples and aggregate results."""
    all_results = []
    failed_indices = []

    for batch_index in range(start_idx, end_idx):
        print(f"\nProcessing batch_index={batch_index}...")
        try:
            result = run_shap_deletion_analysis(
                dataset=dataset,
                batch_index=batch_index,
                repo_root=repo_root,
                **kwargs
            )
            all_results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")
            failed_indices.append(batch_index)

    # Aggregate statistics
    summary = compute_batch_summary(all_results, dataset, start_idx, end_idx, failed_indices)

    # Save batch results
    out_dir = os.path.join(repo_root, "outputs", dataset, "shap_deletion_batch_analysis")
    os.makedirs(out_dir, exist_ok=True)

    batch_output = {
        "summary": summary,
        "individual_results": all_results
    }

    with open(os.path.join(out_dir, "shap_analysis_batch.json"), "w", encoding="utf-8") as f:
        json.dump(batch_output, f, indent=2, default=str)

    print(f"\nBatch results saved to: {out_dir}")
    return batch_output


def compute_batch_summary(results: list, dataset: str, start_idx: int, end_idx: int, failed_indices: list) -> dict:
    """Compute summary statistics from batch results."""
    if not results:
        return {"error": "No successful results", "failed_indices": failed_indices}

    # Collect metrics
    additivity_errors = []
    loo_pearsons = []
    loo_spearmans = []
    attn_vs_absphi_pearsons = []
    attn_vs_absphi_spearmans = []
    attn_vs_phi_pearsons = []
    top5_overlaps = []
    prefix_lengths = []

    for r in results:
        prefix_lengths.append(r["prefix_length"])
        additivity_errors.append(abs(r["verification"]["additivity_error_vs_true_full"]))

        if r.get("verification_loo"):
            loo_pearsons.append(r["verification_loo"]["pearson_phi_vs_loo"])
            loo_spearmans.append(r["verification_loo"]["spearman_phi_vs_loo"])

        ac = r.get("attention_comparison", {})
        if ac.get("attention_available"):
            attn_vs_absphi_pearsons.append(ac["pearson_attn_vs_absphi"])
            attn_vs_absphi_spearmans.append(ac["spearman_attn_vs_absphi"])
            attn_vs_phi_pearsons.append(ac["pearson_attn_vs_phi"])
            if "topk_overlap_ratio_attn_absphi" in ac:
                top5_overlaps.append(ac["topk_overlap_ratio_attn_absphi"])

    def safe_stats(arr):
        arr = [x for x in arr if not np.isnan(x)]
        if not arr:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr)
        }

    return {
        "dataset": dataset,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_successful": len(results),
        "n_failed": len(failed_indices),
        "failed_indices": failed_indices,
        "prefix_length": safe_stats(prefix_lengths),
        "additivity_error_abs": safe_stats(additivity_errors),
        "loo_pearson": safe_stats(loo_pearsons),
        "loo_spearman": safe_stats(loo_spearmans),
        "attn_vs_absphi_pearson": safe_stats(attn_vs_absphi_pearsons),
        "attn_vs_absphi_spearman": safe_stats(attn_vs_absphi_spearmans),
        "attn_vs_phi_pearson": safe_stats(attn_vs_phi_pearsons),
        "top5_overlap_ratio": safe_stats(top5_overlaps),
    }


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SHAP Deletion-Only Analysis for ProcessTransformer"
    )
    parser.add_argument("--dataset", default="BPIC2012-O", help="Dataset name")
    parser.add_argument("--batch_index", type=int, default=0, help="Batch index for single analysis")
    parser.add_argument("--batch_mode", action="store_true", help="Run batch analysis")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for batch mode")
    parser.add_argument("--end_idx", type=int, default=32, help="End index for batch mode")
    parser.add_argument("--repo_root", default=".", help="Repository root path")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Number of SHAP permutations")
    parser.add_argument("--explain_probs", action="store_true", help="Explain probabilities instead of logits")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--no_loo", action="store_true", help="Skip LOO verification")
    parser.add_argument("--no_topk", action="store_true", help="Skip top-k verification")
    parser.add_argument("--topk", type=int, default=10, help="Max k for top-k checks")

    args = parser.parse_args()

    print(f"SHAP Deletion-Only Analysis for {args.dataset}")
    print(f"Repository root: {args.repo_root}")
    print()

    if args.batch_mode:
        run_batch_analysis(
            dataset=args.dataset,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            repo_root=args.repo_root,
            explain_logit=not args.explain_probs,
            n_permutations=args.n_permutations,
            random_seed=args.seed,
            do_loo_check=not args.no_loo,
            do_topk_check=not args.no_topk,
            topk=args.topk,
        )
    else:
        run_shap_deletion_analysis(
            dataset=args.dataset,
            batch_index=args.batch_index,
            repo_root=args.repo_root,
            out_dir=args.out_dir,
            explain_logit=not args.explain_probs,
            n_permutations=args.n_permutations,
            random_seed=args.seed,
            do_loo_check=not args.no_loo,
            do_topk_check=not args.no_topk,
            topk=args.topk,
        )


if __name__ == "__main__":
    main()
