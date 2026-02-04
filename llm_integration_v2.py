"""
LLM Integration for Hybrid Explanations (v6 - Rich Non-Aligned Explanations)
=============================================================================
Changes in v6:
- When alignment is weak, provide detailed analysis of what EACH signal tells us
- Explain the meaning of high-attention events (context, recency, etc.)
- Explain the meaning of high-SHAP events (what drives the score)
"""

import os
from typing import List, Tuple, Any, Optional, Dict

from openai import OpenAI
import torch
from transformers import pipeline


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You explain predictions of a business process prediction model to an end user.
You provide local explanations for a single running case, not global descriptions.

KEY GUIDELINES:
1. Explain what both signals mean:
   - ATTENTION: Where the model focused when processing this case
   - SHAP: How the prediction score changes when events are removed
2. When signals don't align, explain what EACH signal tells us separately
3. Use clear business language - no ML jargon

IMPORTANT FORMATTING RULES:
- Do NOT include numeric probabilities (no "96%" or "0.35")
- Use confidence levels: "very high/high/moderate/low confidence"
- Refer to events by number and name: "Event 5 (o_created)"

Your answer must be a single coherent explanation block (no numbered sections).
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _confidence_level(prob: float) -> str:
    if prob >= 0.95:
        return "very high confidence"
    elif prob >= 0.80:
        return "high confidence"
    elif prob >= 0.60:
        return "moderate confidence"
    elif prob >= 0.40:
        return "low confidence"
    else:
        return "very low confidence"


def _get_process_context(dataset_name: str) -> str:
    name = dataset_name.lower()
    if "helpdesk" in name:
        return "Process: Helpdesk incident management."
    if "bpic2012" in name or "bpi2012" in name:
        if "_o" in name or "-o" in name:
            return "Process: Offer handling subprocess (BPI Challenge 2012)."
        return "Process: Loan application handling (BPI Challenge 2012)."
    return "Process: Business process event log."


# ---------------------------------------------------------------------------
# Analyze alignment
# ---------------------------------------------------------------------------
def analyze_signal_alignment(
    attention_events: List[Tuple[int, str, float]],
    shap_events: List[Tuple[int, str, float]],
    top_k: int = 3,
) -> Dict[str, Any]:
    attn_top = {pos: (act, weight) for pos, act, weight in attention_events[:top_k]}
    shap_top = {pos: (act, val) for pos, act, val in shap_events[:top_k]}

    attn_positions = set(attn_top.keys())
    shap_positions = set(shap_top.keys())

    aligned_positions = attn_positions & shap_positions
    attn_only_positions = attn_positions - shap_positions
    shap_only_positions = shap_positions - attn_positions

    aligned_events = []
    for pos in aligned_positions:
        act, attn_weight = attn_top[pos]
        _, shap_val = shap_top[pos]
        aligned_events.append({
            "position": pos,
            "activity": act,
            "attention_weight": attn_weight,
            "shap_value": shap_val,
        })
    aligned_events.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    attention_only = [
        {"position": pos, "activity": attn_top[pos][0], "attention_weight": attn_top[pos][1]}
        for pos in sorted(attn_only_positions)
    ]

    shap_only = [
        {"position": pos, "activity": shap_top[pos][0], "shap_value": shap_top[pos][1]}
        for pos in sorted(shap_only_positions)
    ]

    overlap_count = len(aligned_positions)
    if overlap_count >= 2:
        alignment_strength = "strong"
    elif overlap_count == 1:
        alignment_strength = "partial"
    else:
        alignment_strength = "weak"

    return {
        "aligned_events": aligned_events,
        "attention_only": attention_only,
        "shap_only": shap_only,
        "overlap_count": overlap_count,
        "alignment_strength": alignment_strength,
    }


# ---------------------------------------------------------------------------
# Analyze prefix patterns
# ---------------------------------------------------------------------------
def analyze_prefix_patterns(
    prefix_activities: List[str],
    predicted_activity: str,
) -> Dict[str, Any]:
    L = len(prefix_activities)
    if L == 0:
        return {"patterns_found": False, "specific_observations": []}

    last_activity = prefix_activities[-1]
    specific_observations = []

    # Where does predicted activity appear?
    pred_positions = [i + 1 for i, act in enumerate(prefix_activities) if act == predicted_activity]
    if pred_positions:
        specific_observations.append({
            "type": "predicted_in_prefix",
            "description": f"The predicted activity '{predicted_activity}' already appears at Event(s) {pred_positions}."
        })

    # Where does last_activity → predicted_activity appear?
    sequence_positions = []
    for i in range(L - 1):
        if prefix_activities[i] == last_activity and prefix_activities[i + 1] == predicted_activity:
            sequence_positions.append(i + 1)

    if sequence_positions:
        specific_observations.append({
            "type": "sequence_match",
            "description": f"The sequence '{last_activity} → {predicted_activity}' appears at Event(s) {sequence_positions}. The prefix ends with '{last_activity}' (Event {L})."
        })

    # What follows last_activity?
    followers = []
    for i in range(L - 1):
        if prefix_activities[i] == last_activity:
            followers.append((i + 1, prefix_activities[i + 1]))

    if followers:
        follower_summary = [f"Event {pos}: {act}" for pos, act in followers]
        matches = sum(1 for _, act in followers if act == predicted_activity)
        specific_observations.append({
            "type": "follower_analysis",
            "description": f"In this prefix, '{last_activity}' is followed by: {follower_summary}. {matches} of these match '{predicted_activity}'.",
            "match_count": matches,
            "total_count": len(followers)
        })

    # Recent context
    recent = prefix_activities[-min(5, L):]
    specific_observations.append({
        "type": "recent_context",
        "description": f"Recent sequence: {' → '.join(recent)}"
    })

    return {
        "patterns_found": len(specific_observations) > 1,
        "specific_observations": specific_observations,
        "last_activity": last_activity,
        "prefix_length": L,
    }


# ---------------------------------------------------------------------------
# Analyze what attention-only events tell us
# ---------------------------------------------------------------------------
def analyze_attention_meaning(
    attention_only_events: List[Dict],
    prefix_activities: List[str],
    prefix_len: int,
) -> str:
    """
    Analyze what the high-attention (but not high-SHAP) events tell us.
    """
    if not attention_only_events:
        return ""

    lines = []
    lines.append("WHAT THE HIGH-ATTENTION EVENTS TELL US:")
    lines.append("The model focused on these events when processing the case, but removing them doesn't strongly change the prediction score.")
    lines.append("")

    for evt in attention_only_events:
        pos = evt["position"]
        act = evt["activity"]

        # Analyze position context
        is_recent = (prefix_len - pos) <= 3
        is_first = pos <= 2

        position_context = []
        if is_recent:
            position_context.append("recent event")
        if is_first:
            position_context.append("early in the case")

        # Count how often this activity appears
        count = sum(1 for a in prefix_activities if a == act)
        if count > 1:
            position_context.append(f"'{act}' appears {count} times in prefix")

        context_str = f" ({', '.join(position_context)})" if position_context else ""

        lines.append(f"  → Event {pos + 1} ({act}){context_str}")
        lines.append(f"    The model looked at this event for context. It may help the model understand")
        lines.append(f"    the case state, but the prediction doesn't depend strongly on this specific event.")

    lines.append("")
    lines.append("  Interpretation: These events provide context - the model uses them to understand")
    lines.append("  where the case is in the process, even though they don't directly drive the prediction score.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyze what SHAP-only events tell us
# ---------------------------------------------------------------------------
def analyze_shap_meaning(
    shap_only_events: List[Dict],
    prefix_activities: List[str],
    predicted_activity: str,
    prefix_len: int,
) -> str:
    """
    Analyze what the high-SHAP (but not high-attention) events tell us.

    IMPORTANT: Lead with what the event DOES, not the removal effect.
    The removal effect is confusing (double-negative reasoning).
    """
    if not shap_only_events:
        return ""

    lines = []
    lines.append("WHAT THE HIGH-SHAP EVENTS TELL US:")
    lines.append("These events strongly affect the prediction score, but the model didn't focus attention on them.")
    lines.append("")

    for evt in shap_only_events:
        pos = evt["position"]
        act = evt["activity"]
        shap = evt["shap_value"]

        # Lead with what the event DOES (positive = supports, negative = opposes)
        if shap > 0:
            impact = "POSITIVE impact - supports the prediction"
        else:
            impact = "NEGATIVE impact - pushes away from the prediction"

        # Analyze relationship to predicted activity
        relationship = []
        if act == predicted_activity:
            relationship.append("same activity as predicted")

        # Check if this activity precedes predicted activity elsewhere
        for i in range(prefix_len - 1):
            if prefix_activities[i] == act and prefix_activities[i + 1] == predicted_activity:
                relationship.append(f"followed by '{predicted_activity}' at Event {i + 2}")
                break

        rel_str = f" [{', '.join(relationship)}]" if relationship else ""

        lines.append(f"  → Event {pos + 1} ({act}): {impact}{rel_str}")

    lines.append("")
    lines.append("  Interpretation: The model learned patterns connecting these events to the prediction,")
    lines.append("  even without explicitly focusing on them.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------
def _build_user_prompt(
    dataset_name: str,
    case_id: Any,
    prefix_activities: List[str],
    predicted_activity: str,
    pred_prob: float,
    top_alternatives: List[Tuple[str, float]],
    attention_events: List[Tuple[int, str, float]],
    shap_events: List[Tuple[int, str, float]],
    dataset_statistics: Optional[Dict[str, Any]] = None,
) -> str:

    alignment = analyze_signal_alignment(attention_events, shap_events)
    patterns = analyze_prefix_patterns(prefix_activities, predicted_activity)

    prefix_len = len(prefix_activities)

    # Prepare prefix block
    if prefix_activities:
        prefix_lines = [f"{i + 1}. {act}" for i, act in enumerate(prefix_activities)]
        prefix_block = "\n".join(prefix_lines)
    else:
        prefix_block = "(prefix is empty)"

    confidence = _confidence_level(pred_prob)
    confidence_is_high = pred_prob >= 0.80

    # Alternatives
    if top_alternatives:
        meaningful_alts = []
        for label, prob in top_alternatives:
            if prob > 0.10:
                meaningful_alts.append(f"{label} (moderate likelihood)")
            elif prob > 0.05:
                meaningful_alts.append(f"{label} (low likelihood)")
        alt_block = ", ".join(meaningful_alts) if meaningful_alts else "None with meaningful likelihood"
    else:
        alt_block = "Not available"

    # Signal explanation
    signal_explanation = """
WHAT THE SIGNALS MEAN:

ATTENTION: Shows where the model focused when processing this case.
High attention = the model weighted this event heavily when building its understanding.
Think of it as "what the model looked at most."

SHAP (deletion-based): Shows each event's impact on the prediction score.
- POSITIVE impact = event supports the prediction (makes it more likely)
- NEGATIVE impact = event pushes away from the prediction (makes it less likely)
Think of it as "what helps or hurts the prediction."
"""

    # Build KEY EVENTS section
    key_events_lines = []

    # ALIGNED EVENTS
    if alignment["aligned_events"]:
        key_events_lines.append("ALIGNED EVENTS (high attention AND high SHAP):")
        key_events_lines.append("These are the most reliable indicators - the model focused on them AND they affect the score.")
        key_events_lines.append("")

        for evt in alignment["aligned_events"]:
            pos = evt["position"]
            act = evt["activity"]
            shap = evt["shap_value"]
            # Lead with what the event DOES
            if shap > 0:
                impact = "POSITIVE impact (supports prediction)"
            else:
                impact = "NEGATIVE impact (pushes away from prediction)"
            key_events_lines.append(f"  → Event {pos + 1} ({act}): {impact}")
        key_events_lines.append("")

    key_events_block = "\n".join(key_events_lines)

    # Build detailed analysis for NON-ALIGNED events
    attention_analysis = analyze_attention_meaning(
        alignment["attention_only"],
        prefix_activities,
        prefix_len
    )

    shap_analysis = analyze_shap_meaning(
        alignment["shap_only"],
        prefix_activities,
        predicted_activity,
        prefix_len
    )

    # Build ALIGNMENT SUMMARY
    if alignment["alignment_strength"] == "strong":
        alignment_summary = (
            f"SIGNAL ALIGNMENT: Strong\n"
            f"{alignment['overlap_count']} events appear in both rankings. "
            f"The model's focus aligns with what drives the prediction."
        )
    elif alignment["alignment_strength"] == "partial":
        evt = alignment["aligned_events"][0]
        alignment_summary = (
            f"SIGNAL ALIGNMENT: Partial\n"
            f"Event {evt['position'] + 1} ({evt['activity']}) appears in both rankings - this is the most reliable indicator.\n"
            f"Other events contribute through either attention (context) or SHAP (score impact) separately."
        )
    else:
        if confidence_is_high:
            alignment_summary = (
                f"SIGNAL ALIGNMENT: Different perspectives, but {confidence}\n\n"
                f"The attention and SHAP rankings highlight different events, yet the model is confident.\n"
                f"This means: the model gathers context from one set of events (attention) while\n"
                f"the prediction score is driven by a different set (SHAP). Both contribute to the prediction\n"
                f"in different ways. See the detailed analysis below for what each set tells us."
            )
        else:
            alignment_summary = (
                f"SIGNAL ALIGNMENT: Different perspectives with {confidence}\n\n"
                f"The signals point to different events and confidence is not high.\n"
                f"The prediction is less certain."
            )

    process_context = _get_process_context(dataset_name)

    # =========================================================================
    # PROMPT
    # =========================================================================
    user_prompt = f"""
[CASE INFORMATION]
Dataset: {dataset_name}
Case ID: {case_id}
Process context: {process_context}

[PREDICTION]
Predicted next activity: {predicted_activity}
Model confidence: {confidence}
Alternatives: {alt_block}

[CASE PREFIX]
Prefix length: {prefix_len} events
{prefix_block}

{signal_explanation}

[KEY EVENTS]
{key_events_block}
{alignment_summary}

{attention_analysis}

{shap_analysis}

[EXPLANATION TASK]
Write a clear explanation for a process analyst.

1. State the prediction and confidence level.

2. For ALIGNED events: Explain they are reliable because the model focused on them AND they affect the score.
   Example: "Event 14 (o_created) has positive impact - it supports the prediction of o_sent."

3. For HIGH-ATTENTION-ONLY events: Explain what context they provide.
   Example: "The model focused on Event 6 (o_created) for context. While this helps 
   understand the case flow, it doesn't strongly affect the prediction score."

4. For HIGH-SHAP-ONLY events: Explain their impact directly.
   Example (positive): "Event 12 (o_sent) has positive impact - it supports the prediction."
   Example (negative): "Event 10 (o_cancelled) has negative impact - it pushes away from the prediction."
   
   IMPORTANT: Do NOT say "removing X would increase/decrease the score" - this is confusing.
   Instead, directly say "X has positive/negative impact" or "X supports/opposes the prediction."


RULES:
- NO numeric values
- Be specific about THIS prefix
- Lead with what each event DOES (positive/negative impact, supports/opposes)
- Do NOT use "removing would increase/decrease" phrasing
- Explain what BOTH signals tell us, even when they don't align
"""
    return user_prompt


# ---------------------------------------------------------------------------
# LLM Clients
# ---------------------------------------------------------------------------
_gpt_client: Optional[OpenAI] = None

def _get_gpt_client() -> OpenAI:
    global _gpt_client
    if _gpt_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set.")
        _gpt_client = OpenAI(api_key=api_key)
    return _gpt_client

def _call_gpt(prompt: str, model_name: Optional[str]) -> str:
    client = _get_gpt_client()
    model = model_name or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return resp.choices[0].message.content

_local_llm_pipe = None

def _get_local_llm_pipe(model_name: Optional[str]):
    global _local_llm_pipe
    if _local_llm_pipe is not None:
        return _local_llm_pipe
    model_id = model_name or "meta-llama/Llama-3.2-3B-Instruct"
    _local_llm_pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    return _local_llm_pipe

def _call_local_llm(prompt: str, model_name: Optional[str]) -> str:
    pipe = _get_local_llm_pipe(model_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(messages, max_new_tokens=800)
    generated_messages = outputs[0]["generated_text"]
    last_msg = generated_messages[-1]
    if isinstance(last_msg, dict) and "content" in last_msg:
        return last_msg["content"].strip()
    return str(last_msg).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_hybrid_explanation(
    dataset_name: str,
    case_id: Any,
    prefix_activities: List[str],
    predicted_activity: str,
    pred_prob: float,
    top_alternatives: List[Tuple[str, float]],
    attention_events: List[Tuple[int, str, float]],
    shap_events: List[Tuple[int, str, float]],
    backend: str = "gpt",
    llm_model_name: Optional[str] = None,
    dataset_statistics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate explanation with detailed analysis of non-aligned events.
    """
    alignment_analysis = analyze_signal_alignment(attention_events, shap_events)

    prompt = _build_user_prompt(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives,
        attention_events=attention_events,
        shap_events=shap_events,
        dataset_statistics=dataset_statistics,
    )

    b = backend.lower()
    if b == "gpt":
        explanation = _call_gpt(prompt, llm_model_name)
    elif b in ("local", "llama", "llmama"):
        explanation = _call_local_llm(prompt, llm_model_name)
    else:
        raise ValueError(f"Unknown backend '{backend}'.")

    return {
        "explanation": explanation,
        "confidence_level": _confidence_level(pred_prob),
        "alignment_analysis": alignment_analysis,
        "prompt": prompt,
    }


def generate_llm_explanation(
    dataset_name: str,
    case_id: Any,
    prefix_activities: List[str],
    predicted_activity: str,
    pred_prob: float,
    top_alternatives: List[Tuple[str, float]],
    important_events: List[Tuple[int, str, float]],
    backend: str = "gpt",
    llm_model_name: Optional[str] = None,
) -> str:
    """Backward-compatible wrapper."""
    result = generate_hybrid_explanation(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives,
        attention_events=important_events,
        shap_events=[],
        backend=backend,
        llm_model_name=llm_model_name,
    )
    return result["explanation"]