#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis for ProcessTransformer Next Activity Prediction.

This script computes SHAP values using proper deletion semantics (mask-space)
and compares them with attention scores. Supports both single-sample and
batch analysis modes.

Key features:
- True Shapley value semantics (deletion, not in-place masking)
- Proper left-padding preservation for right-aligned sequences
- Built-in attention vs SHAP comparison with correlation metrics
- Comprehensive verification (LOO, top-k removal, additivity)
- Batch processing for statistical analysis across multiple samples

Usage:
    # Single sample analysis
    python shap_analysis.py --dataset BPIC2012-O --batch_index 13

    # Batch analysis (multiple samples)
    python shap_analysis.py --dataset BPIC2012-O --batch_mode --start_idx 0 --end_idx 32
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "explain_logit": True,           # Explain logits (recommended) vs probabilities
    "n_permutations": 1000,          # Number of permutations for SHAP
    "random_seed": 123,              # For reproducibility
    "do_loo_check": True,            # Leave-one-out verification
    "do_topk_check": True,           # Top-k removal verification
    "topk_max": 10,                  # Maximum k for top-k checks
}


# =============================================================================
# Utilities
# =============================================================================

def resolve_ckpt_prefix(models_root: str, dataset: str) -> str:
    """Resolve the checkpoint path for a dataset."""
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


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def pad_left_to_max(tokens: np.ndarray, max_len: int, pad_id: int) -> np.ndarray:
    """
    Left-pad tokens to max_len, preserving right-alignment.

    This matches the deletion semantics: kept tokens maintain their
    relative order and are right-aligned in the output.
    """
    out = np.full((max_len,), pad_id, dtype=np.int32)
    if tokens.size == 0:
        return out
    if tokens.size > max_len:
        tokens = tokens[-max_len:]
    out[-tokens.size:] = tokens.astype(np.int32)
    return out


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        import pandas as pd
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        return pearson_corr(ra, rb)
    except ImportError:
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


def load_prefix_from_batch_predictions(path: str, batch_index: int) -> Tuple[Any, List[str], str, float]:
    """Load prefix data from batch_predictions.json."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    preds = obj["predictions"]
    for item in preds:
        if int(item.get("batch_index", -1)) == int(batch_index):
            return (
                item["case_id"],
                item["prefix_activities"],
                item.get("predicted_label"),
                item.get("pred_prob")
            )

    raise ValueError(f"batch_index={batch_index} not found in {path}")


def load_attention_for_sample(
    attn_path: str,
    batch_index: int,
    prefix_len: int
) -> Optional[np.ndarray]:
    """
    Load and normalize attention weights for the last query position.

    Returns normalized attention vector of shape (L,) or None if unavailable.
    """
    if not os.path.exists(attn_path):
        return None

    try:
        A = np.load(attn_path)  # (N, H, T, T)
        if batch_index < 0 or batch_index >= A.shape[0]:
            return None

        sample = A[batch_index]  # (H, T, T)
        H, T, _ = sample.shape

        if prefix_len > T:
            return None

        # Extract last row of the real token region
        start_pos = T - prefix_len
        last_pos = T - 1

        # Average across heads, get attention from last position to all positions
        attn_last_per_head = sample[:, last_pos, start_pos:T]  # (H, L)
        attn_last = attn_last_per_head.mean(axis=0)  # (L,)

        # Normalize to sum to 1
        s = float(attn_last.sum())
        if s > 0:
            attn_last = attn_last / s

        return attn_last
    except Exception:
        return None


# =============================================================================
# SHAP Analysis Core
# =============================================================================

class SHAPAnalyzer:
    """
    Analyzer for computing SHAP values and comparing with attention.

    Uses mask-space SHAP computation with proper deletion semantics.
    """

    def __init__(
        self,
        model,
        x_word_dict: dict,
        y_word_dict: dict,
        max_case_length: int,
        config: dict = None
    ):
        self.model = model
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.max_case_length = max_case_length
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        self.pad_id = int(x_word_dict.get("[PAD]") or x_word_dict.get("<pad>") or 0)
        self.inv_x = {v: k for k, v in x_word_dict.items()}
        self.inv_y = {v: k for k, v in y_word_dict.items()}

    def analyze_single(
        self,
        prefix_activities: List[str],
        case_id: Any,
        attention_vector: Optional[np.ndarray] = None,
        stored_pred_label: str = None,
        stored_pred_prob: float = None
    ) -> dict:
        """
        Analyze a single prefix using SHAP.

        Parameters
        ----------
        prefix_activities : list of str
            Activity names in the prefix
        case_id : any
            Case identifier
        attention_vector : np.ndarray, optional
            Normalized attention weights from model (shape: L,)
        stored_pred_label : str, optional
            Stored prediction label (for verification)
        stored_pred_prob : float, optional
            Stored prediction probability (for verification)

        Returns
        -------
        dict
            Analysis results including SHAP values, verification, and comparison metrics
        """
        # Tokenize prefix
        x_unpadded = np.array(
            [self.x_word_dict[a] for a in prefix_activities],
            dtype=np.int32
        )
        L = len(x_unpadded)
        feature_names = [
            f"E{i+1}:{self.inv_x.get(int(t), str(int(t)))}"
            for i, t in enumerate(x_unpadded)
        ]

        # Get model prediction on full prefix
        x_full_padded = pad_left_to_max(
            x_unpadded, self.max_case_length, self.pad_id
        ).reshape(1, -1)

        logits_full = self.model.predict(x_full_padded, verbose=0)
        probs_full = softmax(logits_full, axis=-1)
        class_idx = int(np.argmax(probs_full[0]))
        pred_label = self.inv_y.get(class_idx, str(class_idx))
        pred_prob = float(probs_full[0, class_idx])

        # Define value function on mask vectors
        explain_logit = self.config["explain_logit"]

        def f_from_masks(masks: np.ndarray) -> np.ndarray:
            masks = np.asarray(masks)
            if masks.ndim == 1:
                masks = masks.reshape(1, -1)

            keep = masks > 0.5
            n = keep.shape[0]

            X_batch = np.zeros((n, self.max_case_length), dtype=np.int32)
            for r in range(n):
                kept_tokens = x_unpadded[keep[r]]
                X_batch[r] = pad_left_to_max(
                    kept_tokens, self.max_case_length, self.pad_id
                )

            logits = self.model.predict(X_batch, verbose=0)
            if explain_logit:
                return logits[:, class_idx].astype(np.float64)

            probs = softmax(logits, axis=-1)
            return probs[:, class_idx].astype(np.float64)

        # Compute baseline values
        mask_empty = np.zeros((1, L), dtype=np.int8)
        base_val_true = float(f_from_masks(mask_empty)[0])

        mask_full = np.ones((1, L), dtype=np.int8)
        fx_full_true = float(f_from_masks(mask_full)[0])

        # Run SHAP permutation explainer
        background = np.zeros((1, L), dtype=np.int8)
        masker = shap.maskers.Independent(background)

        n_perms = self.config["n_permutations"]
        max_evals = int((L + 1) * n_perms)

        try:
            explainer = shap.Explainer(f_from_masks, masker, algorithm="permutation")
            exp = explainer(mask_full, max_evals=max_evals)
        except TypeError:
            # Fallback for older SHAP versions
            explainer = shap.Explainer(f_from_masks, masker, algorithm="permutation")
            exp = explainer(mask_full)
        except Exception:
            explainer = shap.explainers.Permutation(f_from_masks, masker)
            try:
                exp = explainer(mask_full, max_evals=max_evals)
            except TypeError:
                exp = explainer(mask_full)

        phi = np.array(exp.values[0], dtype=np.float64)
        base_val = float(np.array(exp.base_values).reshape(-1)[0])

        # Verification metrics
        fx_recon = float(base_val + phi.sum())
        add_err = float(fx_recon - fx_full_true)
        base_diff = float(base_val - base_val_true)

        verification = {
            "base_value_true_empty": base_val_true,
            "base_value_shap": base_val,
            "base_value_diff": base_diff,
            "f_full_true": fx_full_true,
            "f_reconstructed": fx_recon,
            "additivity_error": add_err,
            "n_permutations": n_perms,
            "max_evals": max_evals,
            "explain_logit": explain_logit,
        }

        # LOO verification
        loo = {}
        if self.config["do_loo_check"]:
            deltas = np.zeros((L,), dtype=np.float64)
            for i in range(L):
                m = np.ones((1, L), dtype=np.int8)
                m[0, i] = 0
                deltas[i] = fx_full_true - float(f_from_masks(m)[0])

            loo = {
                "pearson_phi_vs_loo": pearson_corr(phi, deltas),
                "spearman_phi_vs_loo": spearman_corr(phi, deltas),
                "kendall_phi_vs_loo": kendall_tau(phi, deltas),
                "pearson_absphi_vs_absloo": pearson_corr(np.abs(phi), np.abs(deltas)),
                "spearman_absphi_vs_absloo": spearman_corr(np.abs(phi), np.abs(deltas)),
                "loo_deltas": deltas.tolist(),
            }

        # Top-k verification
        topk = {}
        if self.config["do_topk_check"]:
            kmax = int(min(self.config["topk_max"], L))
            order_pos = np.argsort(-phi)
            order_abs = np.argsort(-np.abs(phi))
            order_neg = np.argsort(phi)

            drops_pos, drops_abs, increases_neg = [], [], []

            for k in range(1, kmax + 1):
                # Remove top-k by positive SHAP
                m = np.ones((1, L), dtype=np.int8)
                m[0, order_pos[:k]] = 0
                drops_pos.append(float(fx_full_true - f_from_masks(m)[0]))

                # Remove top-k by absolute SHAP
                m = np.ones((1, L), dtype=np.int8)
                m[0, order_abs[:k]] = 0
                drops_abs.append(float(fx_full_true - f_from_masks(m)[0]))

                # Remove top-k by negative SHAP (should increase score)
                m = np.ones((1, L), dtype=np.int8)
                m[0, order_neg[:k]] = 0
                increases_neg.append(float(f_from_masks(m)[0] - fx_full_true))

            topk = {
                "kmax": kmax,
                "top1_positive_feature": feature_names[int(order_pos[0])],
                "top1_abs_feature": feature_names[int(order_abs[0])],
                "drops_topk_positive": drops_pos,
                "drops_topk_abs": drops_abs,
                "increases_remove_topk_negative": increases_neg,
            }

        # Attention comparison
        attention_comparison = {"attention_available": False}
        if attention_vector is not None and len(attention_vector) == L:
            absphi = np.abs(phi)

            attention_comparison = {
                "attention_available": True,
                "attention_weights": attention_vector.tolist(),
                # Correlation metrics
                "pearson_attn_vs_phi": pearson_corr(attention_vector, phi),
                "spearman_attn_vs_phi": spearman_corr(attention_vector, phi),
                "kendall_attn_vs_phi": kendall_tau(attention_vector, phi),
                "pearson_attn_vs_absphi": pearson_corr(attention_vector, absphi),
                "spearman_attn_vs_absphi": spearman_corr(attention_vector, absphi),
                "kendall_attn_vs_absphi": kendall_tau(attention_vector, absphi),
            }

            # Top-k overlap analysis
            for k in [3, 5, min(10, L)]:
                if k > L:
                    continue
                top_attn = set(np.argsort(-attention_vector)[:k].tolist())
                top_absphi = set(np.argsort(-absphi)[:k].tolist())
                top_phi_pos = set(np.argsort(-phi)[:k].tolist())

                overlap_abs = len(top_attn & top_absphi)
                overlap_pos = len(top_attn & top_phi_pos)

                attention_comparison[f"top{k}_overlap_attn_absphi"] = overlap_abs
                attention_comparison[f"top{k}_overlap_ratio_attn_absphi"] = overlap_abs / k
                attention_comparison[f"top{k}_overlap_attn_phi_positive"] = overlap_pos
                attention_comparison[f"top{k}_overlap_ratio_attn_phi_positive"] = overlap_pos / k

            # Feature-level comparison
            k = min(5, L)
            top_attn_idx = np.argsort(-attention_vector)[:k].tolist()
            top_absphi_idx = np.argsort(-absphi)[:k].tolist()

            attention_comparison["top_attn_features"] = [feature_names[i] for i in top_attn_idx]
            attention_comparison["top_absphi_features"] = [feature_names[i] for i in top_absphi_idx]

        # Build result
        result = {
            "case_id": case_id,
            "prefix_activities": prefix_activities,
            "prefix_length": L,
            "feature_names": feature_names,
            "stored_predicted_label": stored_pred_label,
            "stored_predicted_prob": stored_pred_prob,
            "model_predicted_label": pred_label,
            "model_predicted_prob": pred_prob,
            "predicted_class_index": class_idx,
            "shap_values": phi.tolist(),
            "base_value": base_val,
            "f_full": fx_full_true,
            "verification": verification,
            "verification_loo": loo,
            "verification_topk": topk,
            "attention_comparison": attention_comparison,
        }

        return result


def run_single_analysis(
    dataset: str,
    batch_index: int,
    repo_root: str,
    config: dict = None
) -> dict:
    """
    Run SHAP analysis for a single sample.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., "BPIC2012-O")
    batch_index : int
        Index in the batch to analyze
    repo_root : str
        Repository root path
    config : dict, optional
        Configuration overrides

    Returns
    -------
    dict
        Analysis results
    """
    from processtransformer.data import loader
    from processtransformer.models import transformer
    from processtransformer import constants

    config = {**DEFAULT_CONFIG, **(config or {})}

    # Set random seeds
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    # Paths
    batch_pred_path = os.path.join(repo_root, "outputs", dataset, "batch_predictions.json")
    attn_path = os.path.join(repo_root, "outputs", dataset, "block_mha_scores.npy")
    models_root = os.path.join(repo_root, "models")

    # Load prefix data
    case_id, prefix_activities, stored_pred_label, stored_pred_prob = \
        load_prefix_from_batch_predictions(batch_pred_path, batch_index)

    L = len(prefix_activities)

    # Load model
    dl = loader.LogsDataLoader(name=dataset)
    _, _, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = \
        dl.load_data(constants.Task.NEXT_ACTIVITY)

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
    model.load_weights(ckpt).expect_partial()

    # Load attention
    attention_vector = load_attention_for_sample(attn_path, batch_index, L)

    # Run analysis
    analyzer = SHAPAnalyzer(
        model=model,
        x_word_dict=x_word_dict,
        y_word_dict=y_word_dict,
        max_case_length=max_case_length,
        config=config
    )

    result = analyzer.analyze_single(
        prefix_activities=prefix_activities,
        case_id=case_id,
        attention_vector=attention_vector,
        stored_pred_label=stored_pred_label,
        stored_pred_prob=stored_pred_prob
    )

    result["dataset"] = dataset
    result["batch_index"] = batch_index

    return result


def run_batch_analysis(
    dataset: str,
    start_idx: int,
    end_idx: int,
    repo_root: str,
    config: dict = None
) -> dict:
    """
    Run SHAP analysis for multiple samples and aggregate results.

    Parameters
    ----------
    dataset : str
        Dataset name
    start_idx : int
        Starting batch index (inclusive)
    end_idx : int
        Ending batch index (exclusive)
    repo_root : str
        Repository root path
    config : dict, optional
        Configuration overrides

    Returns
    -------
    dict
        Aggregated analysis results with summary statistics
    """
    from processtransformer.data import loader
    from processtransformer.models import transformer
    from processtransformer import constants

    config = {**DEFAULT_CONFIG, **(config or {})}

    # Set random seeds
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    # Paths
    batch_pred_path = os.path.join(repo_root, "outputs", dataset, "batch_predictions.json")
    attn_path = os.path.join(repo_root, "outputs", dataset, "block_mha_scores.npy")
    models_root = os.path.join(repo_root, "models")

    # Load model once
    dl = loader.LogsDataLoader(name=dataset)
    _, _, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = \
        dl.load_data(constants.Task.NEXT_ACTIVITY)

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
    model.load_weights(ckpt).expect_partial()

    analyzer = SHAPAnalyzer(
        model=model,
        x_word_dict=x_word_dict,
        y_word_dict=y_word_dict,
        max_case_length=max_case_length,
        config=config
    )

    # Run analysis for each sample
    all_results = []
    failed_indices = []

    for batch_index in range(start_idx, end_idx):
        print(f"Processing batch_index={batch_index}...")
        try:
            # Load prefix data
            case_id, prefix_activities, stored_pred_label, stored_pred_prob = \
                load_prefix_from_batch_predictions(batch_pred_path, batch_index)

            L = len(prefix_activities)
            attention_vector = load_attention_for_sample(attn_path, batch_index, L)

            result = analyzer.analyze_single(
                prefix_activities=prefix_activities,
                case_id=case_id,
                attention_vector=attention_vector,
                stored_pred_label=stored_pred_label,
                stored_pred_prob=stored_pred_prob
            )
            result["dataset"] = dataset
            result["batch_index"] = batch_index
            all_results.append(result)

        except Exception as e:
            print(f"  Failed: {e}")
            failed_indices.append(batch_index)

    # Aggregate statistics
    summary = compute_batch_summary(all_results)
    summary["dataset"] = dataset
    summary["start_idx"] = start_idx
    summary["end_idx"] = end_idx
    summary["n_successful"] = len(all_results)
    summary["n_failed"] = len(failed_indices)
    summary["failed_indices"] = failed_indices

    return {
        "summary": summary,
        "individual_results": all_results
    }


def compute_batch_summary(results: List[dict]) -> dict:
    """Compute summary statistics from batch results."""
    if not results:
        return {"error": "No successful results"}

    # Collect metrics
    additivity_errors = []
    loo_pearsons = []
    loo_spearmans = []
    attn_vs_absphi_pearsons = []
    attn_vs_absphi_spearmans = []
    attn_vs_phi_pearsons = []
    top3_overlaps = []
    top5_overlaps = []
    prefix_lengths = []

    for r in results:
        prefix_lengths.append(r["prefix_length"])
        additivity_errors.append(abs(r["verification"]["additivity_error"]))

        if r["verification_loo"]:
            loo_pearsons.append(r["verification_loo"]["pearson_phi_vs_loo"])
            loo_spearmans.append(r["verification_loo"]["spearman_phi_vs_loo"])

        ac = r["attention_comparison"]
        if ac.get("attention_available"):
            attn_vs_absphi_pearsons.append(ac["pearson_attn_vs_absphi"])
            attn_vs_absphi_spearmans.append(ac["spearman_attn_vs_absphi"])
            attn_vs_phi_pearsons.append(ac["pearson_attn_vs_phi"])

            if "top3_overlap_ratio_attn_absphi" in ac:
                top3_overlaps.append(ac["top3_overlap_ratio_attn_absphi"])
            if "top5_overlap_ratio_attn_absphi" in ac:
                top5_overlaps.append(ac["top5_overlap_ratio_attn_absphi"])

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

    summary = {
        "prefix_length": safe_stats(prefix_lengths),
        "additivity_error_abs": safe_stats(additivity_errors),
        "loo_pearson": safe_stats(loo_pearsons),
        "loo_spearman": safe_stats(loo_spearmans),
        "attn_vs_absphi_pearson": safe_stats(attn_vs_absphi_pearsons),
        "attn_vs_absphi_spearman": safe_stats(attn_vs_absphi_spearmans),
        "attn_vs_phi_pearson": safe_stats(attn_vs_phi_pearsons),
        "top3_overlap_ratio": safe_stats(top3_overlaps),
        "top5_overlap_ratio": safe_stats(top5_overlaps),
    }

    return summary


def save_results(result: dict, out_dir: str, batch_index: int = None):
    """Save analysis results to files."""
    os.makedirs(out_dir, exist_ok=True)

    # Save JSON
    if batch_index is not None:
        json_path = os.path.join(out_dir, f"shap_analysis_batch_{batch_index}.json")
    else:
        json_path = os.path.join(out_dir, "shap_analysis_batch.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Saved results to: {json_path}")

    # Generate plots for single sample
    if "shap_values" in result:
        generate_plots(result, out_dir, batch_index)


def generate_plots(result: dict, out_dir: str, batch_index: int = None):
    """Generate SHAP visualization plots."""
    phi = np.array(result["shap_values"])
    base_val = result["base_value"]
    prefix = result["prefix_activities"]
    feature_names = result["feature_names"]
    pred_label = result["model_predicted_label"]
    L = len(phi)

    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=phi,
        base_values=base_val,
        data=np.array(prefix, dtype=object),
        feature_names=feature_names,
    )

    suffix = f"_batch_{batch_index}" if batch_index is not None else ""

    # Waterfall plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Waterfall - pred={pred_label}{suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_waterfall{suffix}.png"), dpi=200)
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Bar - pred={pred_label}{suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_bar{suffix}.png"), dpi=200)
    plt.close()

    # Attention vs SHAP comparison plot (if available)
    ac = result.get("attention_comparison", {})
    if ac.get("attention_available") and "attention_weights" in ac:
        attn = np.array(ac["attention_weights"])
        absphi = np.abs(phi)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        axes[0].scatter(attn, absphi, alpha=0.7)
        axes[0].set_xlabel("Attention Weight")
        axes[0].set_ylabel("|SHAP Value|")
        axes[0].set_title(
            f"Attention vs |SHAP|\n"
            f"Pearson={ac['pearson_attn_vs_absphi']:.3f}, "
            f"Spearman={ac['spearman_attn_vs_absphi']:.3f}"
        )

        # Side-by-side bar comparison
        x = np.arange(L)
        width = 0.35
        axes[1].bar(x - width/2, attn / attn.max(), width, label="Attention (normalized)")
        axes[1].bar(x + width/2, absphi / absphi.max(), width, label="|SHAP| (normalized)")
        axes[1].set_xlabel("Event Position")
        axes[1].set_ylabel("Normalized Importance")
        axes[1].set_title("Attention vs |SHAP| by Position")
        axes[1].legend()
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"E{i+1}" for i in range(L)], rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"attn_vs_shap{suffix}.png"), dpi=200)
        plt.close()


def print_summary(result: dict):
    """Print analysis summary to console."""
    print("\n" + "=" * 80)

    if "summary" in result:
        # Batch results
        s = result["summary"]
        print(f"BATCH ANALYSIS SUMMARY: {s['dataset']}")
        print(f"Samples: {s['n_successful']} successful, {s['n_failed']} failed")
        print(f"Indices: {s['start_idx']} to {s['end_idx']}")
        print("-" * 40)

        def print_metric(name, stats):
            if stats["n"] > 0:
                print(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")

        print_metric("Prefix length", s["prefix_length"])
        print_metric("Additivity error (abs)", s["additivity_error_abs"])
        print_metric("LOO Pearson (phi vs delta)", s["loo_pearson"])
        print_metric("LOO Spearman (phi vs delta)", s["loo_spearman"])
        print("-" * 40)
        print("ATTENTION vs SHAP COMPARISON:")
        print_metric("Pearson (attn vs |phi|)", s["attn_vs_absphi_pearson"])
        print_metric("Spearman (attn vs |phi|)", s["attn_vs_absphi_spearman"])
        print_metric("Pearson (attn vs phi)", s["attn_vs_phi_pearson"])
        print_metric("Top-3 overlap ratio", s["top3_overlap_ratio"])
        print_metric("Top-5 overlap ratio", s["top5_overlap_ratio"])
    else:
        # Single result
        print(f"SINGLE SAMPLE ANALYSIS: {result['dataset']} batch_index={result['batch_index']}")
        print(f"Case ID: {result['case_id']}")
        print(f"Prefix length: {result['prefix_length']}")
        print(f"Prediction: {result['model_predicted_label']} (p={result['model_predicted_prob']:.4f})")
        print("-" * 40)

        v = result["verification"]
        print(f"Additivity error: {v['additivity_error']:.2e}")
        print(f"Base value (true): {v['base_value_true_empty']:.4f}")
        print(f"Base value (SHAP): {v['base_value_shap']:.4f}")

        if result["verification_loo"]:
            loo = result["verification_loo"]
            print(f"LOO correlation: Pearson={loo['pearson_phi_vs_loo']:.3f}, Spearman={loo['spearman_phi_vs_loo']:.3f}")

        ac = result["attention_comparison"]
        if ac.get("attention_available"):
            print("-" * 40)
            print("ATTENTION vs SHAP COMPARISON:")
            print(f"Pearson (attn vs |phi|): {ac['pearson_attn_vs_absphi']:.3f}")
            print(f"Spearman (attn vs |phi|): {ac['spearman_attn_vs_absphi']:.3f}")
            print(f"Pearson (attn vs phi): {ac['pearson_attn_vs_phi']:.3f}")
            if "top5_overlap_ratio_attn_absphi" in ac:
                print(f"Top-5 overlap ratio: {ac['top5_overlap_ratio_attn_absphi']:.2f}")
            print(f"Top attention features: {ac.get('top_attn_features', [])}")
            print(f"Top |SHAP| features: {ac.get('top_absphi_features', [])}")

    print("=" * 80 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SHAP Analysis for ProcessTransformer"
    )
    parser.add_argument("--dataset", default="BPIC2012-O", help="Dataset name")
    parser.add_argument("--batch_index", type=int, default=0, help="Batch index for single analysis")
    parser.add_argument("--batch_mode", action="store_true", help="Run batch analysis")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for batch mode")
    parser.add_argument("--end_idx", type=int, default=32, help="End index for batch mode")
    parser.add_argument("--repo_root", default=".", help="Repository root path")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: outputs/{dataset}/shap)")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Number of SHAP permutations")
    parser.add_argument("--explain_probs", action="store_true", help="Explain probabilities instead of logits")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()

    # Build config
    config = {
        "explain_logit": not args.explain_probs,
        "n_permutations": args.n_permutations,
        "random_seed": args.seed,
        "do_loo_check": True,
        "do_topk_check": True,
        "topk_max": 10,
    }

    # Default output directory
    if args.out_dir is None:
        if args.batch_mode:
            args.out_dir = os.path.join(args.repo_root, "outputs", args.dataset, "shap_batch_analysis")
        else:
            args.out_dir = os.path.join(args.repo_root, "outputs", args.dataset, f"shap_batch_{args.batch_index}")

    print(f"SHAP Analysis for {args.dataset}")
    print(f"Output directory: {args.out_dir}")
    print(f"Config: {config}")
    print()

    if args.batch_mode:
        # Batch analysis
        result = run_batch_analysis(
            dataset=args.dataset,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            repo_root=args.repo_root,
            config=config
        )
        save_results(result, args.out_dir)
        print_summary(result)
    else:
        # Single sample analysis
        result = run_single_analysis(
            dataset=args.dataset,
            batch_index=args.batch_index,
            repo_root=args.repo_root,
            config=config
        )
        save_results(result, args.out_dir, args.batch_index)
        print_summary(result)

        # Save raw arrays for further analysis
        np.save(
            os.path.join(args.out_dir, f"phi_values_batch_{args.batch_index}.npy"),
            np.array(result["shap_values"])
        )
        if result["attention_comparison"].get("attention_available"):
            np.save(
                os.path.join(args.out_dir, f"attention_batch_{args.batch_index}.npy"),
                np.array(result["attention_comparison"]["attention_weights"])
            )


if __name__ == "__main__":
    main()
