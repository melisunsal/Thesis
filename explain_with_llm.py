# explain_with_llm.py
import os
import json
from typing import List, Tuple, Any, Optional

import numpy as np

from llm_integration import generate_llm_explanation


def summarize_attention_for_prefix(
    scores_one: np.ndarray,
    prefix_len: int,
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    """Summarize attention for the last prefix event (including self).

    scores_one: [H, Tq, Tk] attention tensor for one sample, with padding.
    prefix_len: number of real events in the prefix (L).
    Returns: list of (position_index, normalized_importance) for top_k events,
             where position_index is 0-based into prefix_activities.
    """
    H, Tq, Tk = scores_one.shape
    if prefix_len <= 0:
        return []

    L = prefix_len

    # Assume left-padding: real tokens sit in the last L positions of length Tq
    # (this matches how the visualization right-aligns labels)
    offset = Tq - L

    # Remove PAD dimensions on both axes -> [H, L, L]
    S = scores_one[:, offset:, :][:, :, offset:]

    # Head-averaged matrix [L, L]
    S_avg = S.mean(axis=0)

    # Last row = last real event in the prefix (same as bottom row of the heatmap)
    last_row = S_avg[L - 1, :]  # shape [L]

    # Normalize to sum to 1 (for interpretability)
    last_row = last_row.astype("float64")
    total = float(last_row.sum())
    if total > 0.0:
        last_row = last_row / total
    else:
        last_row = np.ones_like(last_row) / max(L, 1)

    # Pick top-k positions (including self)
    order = np.argsort(last_row)[::-1][:top_k]
    result = [(int(pos), float(last_row[pos])) for pos in order]
    return result




def explain_prefix(
    dataset_name: str,
    out_dir: str,
    prefix_index: int,
    top_k_events: int = 3,
    backend: str = "gpt",
    llm_model_name: Optional[str] = None,
) -> str:
    """High-level helper to get an explanation for one prefix.

    This function:
      1. Loads attention scores and predictions from the given run directory.
      2. Builds a small attention summary for the chosen prefix.
      3. Calls the LLM explainer (GPT or LLMama) with the structured prompt.
      4. Returns the explanation text.

    Parameters
    ----------
    dataset_name:
        Dataset name, e.g. "BPIC2012-O" or "Helpdesk".
    out_dir:
        Base output directory used by get_attention_hooked.py, e.g. "./outputs".
    prefix_index:
        Index in the explained mini-batch (0-based).
    top_k_events:
        Number of most important past events (via attention) to surface in the prompt.
    backend:
        "gpt" or "llmama". See llm_integration.generate_llm_explanation.
    llm_model_name:
        Optional model name override for the chosen backend.
    """
    run_dir = os.path.join(out_dir, dataset_name)

    # 1) Load attention scores
    scores_path = os.path.join(run_dir, "block_mha_scores.npy")
    scores_all = np.load(scores_path)  # [B, H, Tq, Tk]

    # 2) Load batch predictions (prefix text, case ids, probabilities)
    preds_path = os.path.join(run_dir, "batch_predictions.json")
    with open(preds_path, "r", encoding="utf-8") as f:
        preds_data = json.load(f)

    predictions = preds_data["predictions"]
    B = scores_all.shape[0]

    idx = prefix_index
    if idx < 0 or idx >= B or idx >= len(predictions):
        raise ValueError(
            f"prefix_index {idx} is out of range for batch size {B} "
            f"and prediction list length {len(predictions)}."
        )

    pred = predictions[idx]
    prefix_activities: List[str] = pred["prefix_activities"]
    prefix_len = int(pred["prefix_len"])
    case_id: Any = pred["case_id"]
    predicted_activity: str = pred["predicted_label"]
    pred_prob: float = float(pred["pred_prob"])
    top_alternatives: List[Tuple[str, float]] = [
        (alt["label"], float(alt["prob"])) for alt in pred["top_alternatives"]
    ]

    # 3) Summarize attention for this prefix
    scores_one = scores_all[idx]  # [H, Tq, Tk]
    attn_top = summarize_attention_for_prefix(
        scores_one, prefix_len, top_k=top_k_events
    )

    important_events: List[Tuple[int, str, float]] = [
        (pos, prefix_activities[pos], weight)
        for pos, weight in attn_top
        if 0 <= pos < len(prefix_activities)
    ]

    # 4) Call LLM through the shared integration
    explanation = generate_llm_explanation(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives,
        important_events=important_events,
        backend=backend,
        llm_model_name=llm_model_name,
    )

    return explanation
