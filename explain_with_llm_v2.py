"""
Hybrid Explanation Generator (v3 - Fixed Prediction Source)
============================================================
IMPORTANT FIX: Reads prediction from SHAP output file, not batch_predictions.json
The SHAP computation re-runs the model and has the correct prediction.

Usage:
    from explain_with_llm import explain_prefix

    result = explain_prefix(
        dataset_name="BPIC2012-O",
        out_dir="./outputs",
        prefix_index=0,
        backend="gpt"
    )
    print(result["explanation"])
"""

import os
import json
from typing import List, Tuple, Any, Optional, Dict

import numpy as np

from llm_integration_v2 import generate_hybrid_explanation


# ---------------------------------------------------------------------------
# Attention summarization
# ---------------------------------------------------------------------------
def summarize_attention_for_prefix(
        scores_one: np.ndarray,
        prefix_len: int,
        top_k: int = 3,
) -> List[Tuple[int, float]]:
    """
    Summarize attention for the last prefix event.
    """
    H, Tq, Tk = scores_one.shape
    if prefix_len <= 0:
        return []

    L = prefix_len
    offset = Tq - L

    # Remove PAD dimensions -> [H, L, L]
    S = scores_one[:, offset:, :][:, :, offset:]

    # Head-averaged matrix [L, L]
    S_avg = S.mean(axis=0)

    # Last row = attention from last event to all events
    last_row = S_avg[L - 1, :].astype("float64")

    # Normalize to sum to 1
    total = float(last_row.sum())
    if total > 0.0:
        last_row = last_row / total
    else:
        last_row = np.ones_like(last_row) / max(L, 1)

    # Top-k positions by attention weight
    order = np.argsort(last_row)[::-1][:top_k]
    return [(int(pos), float(last_row[pos])) for pos in order]


# ---------------------------------------------------------------------------
# SHAP summarization
# ---------------------------------------------------------------------------
def summarize_shap_for_prefix(
        shap_values: np.ndarray,
        prefix_len: int,
        top_k: int = 3,
) -> List[Tuple[int, float]]:
    """
    Summarize SHAP values for a prefix.
    """
    if prefix_len <= 0 or shap_values is None:
        return []

    T = len(shap_values)
    if T > prefix_len:
        offset = T - prefix_len
        shap_real = shap_values[offset:]
    else:
        shap_real = shap_values[:prefix_len]

    L = len(shap_real)

    # Sort by absolute value (most impactful first)
    abs_shap = np.abs(shap_real)
    order = np.argsort(abs_shap)[::-1][:top_k]

    return [(int(pos), float(shap_real[pos])) for pos in order]


# ---------------------------------------------------------------------------
# Load SHAP data AND prediction from SHAP output file
# ---------------------------------------------------------------------------
def load_shap_data(run_dir: str, prefix_index: int) -> Dict[str, Any]:
    """
    Load SHAP values AND prediction from the SHAP output file.

    The SHAP computation re-runs the model, so it has the CORRECT prediction.

    Returns dict with:
        - shap_values: np.ndarray or None
        - predicted_label: str or None (from model re-run)
        - predicted_prob: float or None (from model re-run)
        - case_id: Any or None
        - prefix_activities: List[str] or None
    """
    result = {
        "shap_values": None,
        "predicted_label": None,
        "predicted_prob": None,
        "case_id": None,
        "prefix_activities": None,
    }

    # Primary source: shap_explanation.json from deletion-only pipeline
    shap_json_path = os.path.join(
        run_dir,
        f"shap_deletion_only_shap_pkg_batch_{prefix_index}",
        "shap_explanation.json",
    )

    if os.path.exists(shap_json_path):
        try:
            with open(shap_json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            # SHAP values
            phi = obj.get("phi_deletion_only", None)
            if phi is not None:
                result["shap_values"] = np.array(phi, dtype="float64")

            # CORRECT prediction from model re-run
            result["predicted_label"] = obj.get("model_predicted_label")
            result["predicted_prob"] = obj.get("model_predicted_prob")
            result["case_id"] = obj.get("case_id")
            result["prefix_activities"] = obj.get("prefix_activities")

            return result

        except Exception as e:
            print(f"Warning: Could not load SHAP JSON: {e}")

    # Fallback: try other formats (legacy)
    for fname in os.listdir(run_dir):
        if fname.startswith("shap_values") and fname.endswith(".npy"):
            path = os.path.join(run_dir, fname)
            try:
                data = np.load(path)
                if data.ndim == 2 and prefix_index < len(data):
                    result["shap_values"] = data[prefix_index]
                elif data.ndim == 1:
                    result["shap_values"] = data
                break
            except Exception:
                continue

    return result


# ---------------------------------------------------------------------------
# Load basic prediction data from batch_predictions.json (fallback only)
# ---------------------------------------------------------------------------
def load_batch_prediction(run_dir: str, prefix_index: int) -> Dict[str, Any]:
    """
    Load prediction data from batch_predictions.json.

    NOTE: This may have STALE predictions if the model was retrained.
    Prefer load_shap_data() which re-runs the model.
    """
    preds_path = os.path.join(run_dir, "batch_predictions.json")
    if not os.path.exists(preds_path):
        return {}

    with open(preds_path, "r", encoding="utf-8") as f:
        preds_data = json.load(f)

    predictions = preds_data.get("predictions", [])

    for pred in predictions:
        if pred.get("batch_index") == prefix_index:
            return {
                "case_id": pred.get("case_id"),
                "prefix_activities": pred.get("prefix_activities"),
                "prefix_len": pred.get("prefix_len"),
                "predicted_label": pred.get("predicted_label"),
                "predicted_prob": pred.get("pred_prob"),
                "top_alternatives": [
                    (alt["label"], float(alt["prob"]))
                    for alt in pred.get("top_alternatives", [])
                ],
            }

    return {}


# ---------------------------------------------------------------------------
# Main explanation function
# ---------------------------------------------------------------------------
def explain_prefix(
        dataset_name: str,
        out_dir: str,
        prefix_index: int,
        top_k_attention: int = 3,
        top_k_shap: int = 3,
        backend: str = "gpt",
        llm_model_name: Optional[str] = None,
        shap_values: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Generate a hybrid explanation for one prefix.

    IMPORTANT: This function now reads the prediction from the SHAP output file,
    which re-runs the model and has the CORRECT prediction.

    Parameters
    ----------
    dataset_name : str
        Dataset name, e.g., "BPIC2012-O" or "Helpdesk"
    out_dir : str
        Base output directory
    prefix_index : int
        Index in the batch (0-based)
    top_k_attention : int
        Number of top events by attention
    top_k_shap : int
        Number of top events by |SHAP|
    backend : str
        "gpt" or "local" for LLM backend
    llm_model_name : Optional[str]
        Override model name
    shap_values : Optional[np.ndarray]
        Pre-computed SHAP values (if None, loads from files)

    Returns
    -------
    Dict with explanation, confidence_level, alignment_analysis, etc.
    """
    run_dir = os.path.join(out_dir, dataset_name)

    # 1) Load SHAP data (includes CORRECT prediction from model re-run)
    shap_data = load_shap_data(run_dir, prefix_index)

    # 2) Load batch prediction data (for alternatives and fallback)
    batch_data = load_batch_prediction(run_dir, prefix_index)

    # 3) Use SHAP prediction (correct) over batch prediction (may be stale)
    predicted_activity = shap_data.get("predicted_label") or batch_data.get("predicted_label")
    pred_prob = shap_data.get("predicted_prob") or batch_data.get("predicted_prob") or 0.0
    case_id = shap_data.get("case_id") or batch_data.get("case_id")
    prefix_activities = shap_data.get("prefix_activities") or batch_data.get("prefix_activities") or []
    prefix_len = len(prefix_activities)

    # Debug: show which prediction source we're using
    if shap_data.get("predicted_label"):
        print(f"  [Using prediction from SHAP output: {predicted_activity} (p={pred_prob:.4f})]")
    else:
        print(f"  [Warning: Using prediction from batch_predictions.json - may be stale]")

    if not predicted_activity:
        raise ValueError(f"No prediction found for prefix_index {prefix_index}")

    # 4) Get alternatives from batch data
    top_alternatives = batch_data.get("top_alternatives", [])

    # 5) Load attention scores
    scores_path = os.path.join(run_dir, "block_mha_scores.npy")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Attention scores not found at {scores_path}")
    scores_all = np.load(scores_path)  # [B, H, Tq, Tk]

    if prefix_index >= scores_all.shape[0]:
        raise ValueError(f"prefix_index {prefix_index} out of range for attention scores")

    # 6) Summarize attention
    scores_one = scores_all[prefix_index]
    attn_summary = summarize_attention_for_prefix(
        scores_one, prefix_len, top_k=top_k_attention
    )

    attention_events: List[Tuple[int, str, float]] = [
        (pos, prefix_activities[pos], weight)
        for pos, weight in attn_summary
        if 0 <= pos < len(prefix_activities)
    ]

    # 7) Get SHAP values and summarize
    if shap_values is None:
        shap_values = shap_data.get("shap_values")

    if shap_values is not None:
        shap_summary = summarize_shap_for_prefix(
            shap_values, prefix_len, top_k=top_k_shap
        )
        shap_events: List[Tuple[int, str, float]] = [
            (pos, prefix_activities[pos], shap_val)
            for pos, shap_val in shap_summary
            if 0 <= pos < len(prefix_activities)
        ]
    else:
        shap_events = []
        print(f"  [Warning: No SHAP values found for prefix_index {prefix_index}]")

    # 8) Generate explanation
    result = generate_hybrid_explanation(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives,
        attention_events=attention_events,
        shap_events=shap_events,
        backend=backend,
        llm_model_name=llm_model_name,
    )

    # Add metadata
    result["attention_events"] = attention_events
    result["shap_events"] = shap_events
    result["case_id"] = case_id
    result["predicted_activity"] = predicted_activity
    result["pred_prob"] = pred_prob
    result["prefix_len"] = prefix_len
    result["prefix_activities"] = prefix_activities

    return result


# ---------------------------------------------------------------------------
# Convenience function for direct signal input
# ---------------------------------------------------------------------------
def explain_with_signals(
        dataset_name: str,
        case_id: Any,
        prefix_activities: List[str],
        predicted_activity: str,
        pred_prob: float,
        attention_weights: List[float],
        shap_values: List[float],
        top_alternatives: Optional[List[Tuple[str, float]]] = None,
        top_k: int = 3,
        backend: str = "gpt",
        llm_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate explanation from pre-computed attention and SHAP arrays.
    """
    L = len(prefix_activities)

    # Build attention events (top by weight)
    attn_indexed = [(i, attention_weights[i]) for i in range(min(L, len(attention_weights)))]
    attn_sorted = sorted(attn_indexed, key=lambda x: x[1], reverse=True)[:top_k]
    attention_events = [
        (pos, prefix_activities[pos], weight)
        for pos, weight in attn_sorted
    ]

    # Build SHAP events (top by absolute value)
    shap_indexed = [(i, shap_values[i]) for i in range(min(L, len(shap_values)))]
    shap_sorted = sorted(shap_indexed, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    shap_events = [
        (pos, prefix_activities[pos], val)
        for pos, val in shap_sorted
    ]

    return generate_hybrid_explanation(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives or [],
        attention_events=attention_events,
        shap_events=shap_events,
        backend=backend,
        llm_model_name=llm_model_name,
    )


if __name__ == "__main__":
    # Example
    result = explain_with_signals(
        dataset_name="BPIC2012-O",
        case_id="case_12345",
        prefix_activities=[
            "o_selected", "o_created", "o_sent", "o_cancelled",
            "o_selected", "o_created"
        ],
        predicted_activity="o_sent",
        pred_prob=0.96,
        attention_weights=[0.05, 0.10, 0.08, 0.12, 0.25, 0.40],
        shap_values=[0.3, 1.2, 0.5, -0.8, 0.9, 2.1],
        backend="gpt",
    )
    print(result["explanation"])