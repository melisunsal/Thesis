"""
LLM Integration for Hybrid Explanations
========================================
Combines attention-based focus and SHAP-based causal attribution
to generate natural language explanations for process predictions.

Prompt template follows the 6-part structure documented in the thesis:
1. Meta information
2. Task in one sentence
3. Purpose and goals
4. Process context
5. Model input and signals
6. Output format and instructions

Based on thesis findings:
- Attention = computational focus (where the model "looks")
- SHAP = causal contribution (what drives the prediction under omission)
- When they align → mutual validation
- When they diverge → attention shows context, SHAP shows drivers
"""

import os
from typing import List, Tuple, Any, Optional, Dict

from openai import OpenAI
import torch
from transformers import pipeline


# ---------------------------------------------------------------------------
# System prompt: instructs the LLM on its role and constraints
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You explain predictions of a business process prediction model to an end user.
You always provide local explanations for a single running case and its current
event prefix, not global descriptions of the model or process.
The end user has no machine learning background. Use short, concrete sentences and
business language (activities, cases, approvals, waiting, rework). Do not talk about
tokens, embeddings, or hidden states. Base everything only on the information in
the user message. If something is uncertain, say that it is uncertain, and do not
speculate beyond what can be inferred from this specific case prefix.
Your answer must be a single coherent explanation block (no numbered sections
or headings). Start by briefly summarising the prediction and which past activities
were most influential, then continue with more detailed event-level explanations
and any remaining uncertainty in the same block.
"""


# ---------------------------------------------------------------------------
# Confidence level helper
# ---------------------------------------------------------------------------
def _confidence_level(prob: float) -> str:
    """Convert probability to human-readable confidence level."""
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


# ---------------------------------------------------------------------------
# Process context per dataset (Section 4)
# ---------------------------------------------------------------------------
def _get_process_context(dataset_name: str) -> str:
    """Short, human-readable description per dataset for Section 4."""
    name = dataset_name.lower()

    if "helpdesk" in name:
        return (
            "Process: helpdesk incident management. Each case is one ticket opened by a "
            "customer and handled by support until it is resolved or closed. Typical "
            "activities include ticket assignment, seriousness classification, waiting "
            "for customer response, taking charge of the ticket, and resolution."
        )

    if "bpic2012" in name or "bpi2012" in name:
        if "_o" in name or "-o" in name:
            return (
                "Process: offer handling subprocess from a loan application process "
                "(BPI Challenge 2012). Each case tracks offers that are selected, created, "
                "sent to customers, and either accepted or cancelled. The process is cyclic: "
                "offers may be sent back and recreated multiple times before final resolution."
            )
        return (
            "Process: loan or application handling from the BPI Challenge 2012 logs. Each "
            "case is one application that goes through submission, validation, assessment, "
            "and final decision (acceptance or rejection)."
        )

    # Generic fallback
    return (
        "Process: generic business process recorded as an event log. Each case is one "
        "process instance with a sequence of activities over time."
    )


# ---------------------------------------------------------------------------
# Build the 6-part prompt template
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
) -> str:
    """
    Build the 6-part structured prompt template.

    Parameters
    ----------
    attention_events : List[Tuple[int, str, float]]
        Top events by attention weight: (position, activity_name, normalized_weight)
        These indicate where the model focused during computation.

    shap_events : List[Tuple[int, str, float]]
        Top events by |SHAP value|: (position, activity_name, shap_value)
        Positive SHAP = pushes toward prediction; negative = pushes away.
        These indicate what causally drives the prediction.
    """

    # Prepare prefix block
    if prefix_activities:
        prefix_lines = [f"{i + 1}. {act}" for i, act in enumerate(prefix_activities)]
        prefix_block = "\n".join(prefix_lines)
        prefix_len = len(prefix_activities)
    else:
        prefix_block = "(prefix is empty)"
        prefix_len = 0

    # Confidence level
    confidence = _confidence_level(pred_prob)

    # Alternatives block
    if top_alternatives:
        alt_lines = [f"- {label}" for (label, prob) in top_alternatives if prob > 0.05]
        if alt_lines:
            alt_block = "\n".join(alt_lines)
        else:
            alt_block = "No strong alternative predictions."
    else:
        alt_block = "No alternative predictions available."

    # Attention-based importance block
    if attention_events:
        attn_lines = []
        for pos, act, weight in attention_events:
            attn_lines.append(
                f"- Event {pos + 1}: {act} (attention weight: {weight:.0%})"
            )
        attention_block = "\n".join(attn_lines)
    else:
        attention_block = "No attention information available for this prefix."

    # SHAP-based importance block
    if shap_events:
        shap_lines = []
        for pos, act, shap_val in shap_events:
            if shap_val > 0:
                direction = "supports the prediction"
            else:
                direction = "opposes the prediction"
            shap_lines.append(
                f"- Event {pos + 1}: {act} ({direction}, impact score: {shap_val:+.2f})"
            )
        shap_block = "\n".join(shap_lines)
    else:
        shap_block = "No SHAP information available for this prefix."

    # Check signal alignment for interpretation guidance
    if attention_events and shap_events:
        attn_positions = set(pos for pos, _, _ in attention_events[:3])
        shap_positions = set(pos for pos, _, _ in shap_events[:3])
        overlap = len(attn_positions & shap_positions)
        if overlap >= 2:
            alignment_note = (
                "The attention and SHAP signals largely agree: the events the model "
                "focused on are also the events that causally drive the prediction."
            )
        elif overlap >= 1:
            alignment_note = (
                "The attention and SHAP signals partially overlap: some events the model "
                "focused on are causal drivers, while others may provide context without "
                "directly influencing the outcome."
            )
        else:
            alignment_note = (
                "The attention and SHAP signals highlight different events: the model "
                "attended to certain events for context, but different events actually "
                "drive the prediction. This is common in sequential processes where early "
                "events set context and late events trigger outcomes."
            )
    else:
        alignment_note = ""

    process_context = _get_process_context(dataset_name)

    # =========================================================================
    # 6-PART PROMPT TEMPLATE
    # =========================================================================
    user_prompt = f"""
[SECTION 1 - META INFORMATION]
Dataset: {dataset_name}
Case ID: {case_id}
User type: Process analyst (no machine learning background)

[SECTION 2 - TASK IN ONE SENTENCE]
Explain the model's local decision behaviour for this specific case prefix by
describing how its attention to past events and the causal importance of those
events (measured by SHAP) lead it to favour the predicted next activity over
alternative activities.

[SECTION 3 - PURPOSE AND GOALS]
The explanation should focus on local model behaviour for this single case prefix.
More concretely, it should:
- show which past events in this prefix were most influential according to both
  attention-based focus and SHAP-based causal contribution,
- explain how these locally important events support (or weaken) the predicted
  next activity from a process point of view,
- and mention any remaining uncertainty or plausible alternative next activities.

The goal is to make the model's behaviour on this specific prefix understandable
to a process analyst, not to describe the model in general.

[SECTION 4 - PROCESS CONTEXT]
{process_context}

[SECTION 5 - MODEL INPUT AND SIGNALS]
Prediction task: next activity prediction
Model: ProcessTransformer
Prefix length: {prefix_len} events
Predicted next activity: {predicted_activity}
Model certainty: {confidence}

Top alternative next activities (if any):
{alt_block}

Case prefix (oldest to newest):
{prefix_block}

Attention-based focus (which past events the model looked at most):
{attention_block}

SHAP-based omission impact (deletion masking, logit scale):
{shap_block}

Signal interpretation:
{alignment_note}

[SECTION 6 - OUTPUT FORMAT AND INSTRUCTIONS]
Write a single coherent explanation as one block of text.

Start with a short summary in two or three sentences that:
- states the predicted next activity and the model's certainty level (use terms
  like "high confidence" or "moderate confidence", not exact numbers),
- briefly describes which one to three past events were most influential based on
  the SHAP causal contributions, and why this makes the prediction plausible.

Then continue in the same block with a more detailed explanation that:
- refers to influential events by their index and activity name
  (for example "Event 5: resolve_ticket"),
- explains how these events influence the prediction, distinguishing between
  events that causally drive the prediction (high SHAP) and events the model
  attended to for context (high attention but lower SHAP),
- relates the explanation back to the process context when possible,
- and comments on uncertainty or plausible alternative next activities if the
  model certainty is moderate or low.

Do not introduce headings or numbered sections in your output. Avoid machine
learning jargon. Base all statements only on the information given in this prompt,
and say explicitly when evidence for an interpretation is weak or ambiguous.
"""
    return user_prompt


# ---------------------------------------------------------------------------
# GPT client (OpenAI)
# ---------------------------------------------------------------------------
_gpt_client: Optional[OpenAI] = None
OPENAI_API=""


def _get_gpt_client() -> OpenAI:
    global _gpt_client
    if _gpt_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set to use backend 'gpt'.")
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
        max_tokens=600,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Local LLM (TinyLlama or similar)
# ---------------------------------------------------------------------------
_local_llm_pipe = None


def _get_local_llm_pipe(model_name: Optional[str]):
    """Create or reuse a local LLM chat pipeline."""
    global _local_llm_pipe

    if _local_llm_pipe is not None:
        return _local_llm_pipe

    model_id = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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

    outputs = pipe(
        messages,
        max_new_tokens=500,
    )

    generated_messages = outputs[0]["generated_text"]
    last_msg = generated_messages[-1]

    if isinstance(last_msg, dict) and "content" in last_msg:
        return last_msg["content"].strip()
    else:
        return str(last_msg).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_hybrid_explanation2(
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
) -> Dict[str, Any]:
    """
    Generate a hybrid explanation combining attention and SHAP signals.

    Uses the 6-part prompt template documented in the thesis:
    1. Meta information
    2. Task in one sentence
    3. Purpose and goals
    4. Process context
    5. Model input and signals
    6. Output format and instructions

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., "BPIC2012-O", "Helpdesk")
    case_id : Any
        Identifier for the case being explained
    prefix_activities : List[str]
        List of activity names in the prefix (oldest to newest)
    predicted_activity : str
        The model's predicted next activity
    pred_prob : float
        Probability of the predicted activity (used for confidence level)
    top_alternatives : List[Tuple[str, float]]
        Alternative predictions: [(activity_name, probability), ...]
    attention_events : List[Tuple[int, str, float]]
        Top events by attention: [(position, activity_name, weight), ...]
        Position is 0-based index into prefix_activities
    shap_events : List[Tuple[int, str, float]]
        Top events by |SHAP|: [(position, activity_name, shap_value), ...]
        Positive SHAP supports prediction, negative opposes
    backend : str
        "gpt" for OpenAI, "local" for local LLM
    llm_model_name : Optional[str]
        Override model name for the backend

    Returns
    -------
    Dict containing:
        - explanation: str (the generated text)
        - confidence_level: str
        - signals_aligned: bool (whether attention and SHAP agree)
        - prompt: str (the constructed prompt, for debugging)
    """
    prompt = _build_user_prompt(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives,
        attention_events=attention_events,
        shap_events=shap_events,
    )

    b = backend.lower()
    if b == "gpt":
        explanation = _call_gpt(prompt, llm_model_name)
    elif b in ("local", "llama", "llmama"):
        explanation = _call_local_llm(prompt, llm_model_name)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'gpt' or 'local'.")

    # Determine signal alignment
    attn_positions = set(pos for pos, _, _ in attention_events[:3]) if attention_events else set()
    shap_positions = set(pos for pos, _, _ in shap_events[:3]) if shap_events else set()
    overlap = len(attn_positions & shap_positions)
    signals_aligned = overlap >= 2 if (attn_positions and shap_positions) else None

    return {
        "explanation": explanation,
        # "confidence_level": _confidence_level(pred_prob),
        # "signals_aligned": signals_aligned,
        # "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Backward compatibility wrapper (attention-only mode)
# ---------------------------------------------------------------------------
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
    """
    Backward-compatible wrapper that uses attention-only (legacy behavior).

    For new code, use generate_hybrid_explanation() instead.
    """
    result = generate_hybrid_explanation(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives,
        attention_events=important_events,
        shap_events=[],  # No SHAP in legacy mode
        backend=backend,
        llm_model_name=llm_model_name,
    )
    return result["explanation"]
