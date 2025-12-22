import os
from typing import List, Tuple, Any, Optional

from openai import OpenAI
import torch
from transformers import pipeline


# ---------------------------------------------------------------------------
# System prompt: common for both GPT and LLMama
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You explain predictions of a business process prediction model to an end user.\n"
    "You always provide local explanations for a single running case and its current\n"
    "event prefix, not global descriptions of the model or process.\n"
    "The end user has no machine learning background. Use short, concrete sentences and\n"
    "business language (activities, cases, approvals, waiting, rework). Do not talk about\n"
    "tokens, embeddings, or hidden states. Base everything only on the information in\n"
    "the user message. If something is uncertain, say that it is uncertain, and do not\n"
    "speculate beyond what can be inferred from this specific case prefix.\n"
    "Your answer must be a single coherent explanation block (no numbered sections\n"
    "or headings). Start by briefly summarising the prediction and which past activities\n"
    "the model attended to most, then continue with more detailed event level\n"
    "explanations and any remaining uncertainty in the same block."
)


_gpt_client: Optional[OpenAI] = None
_llmama_pipe = None  # global cache



# ---------------------------------------------------------------------------
# Small helper: dataset specific process context
# ---------------------------------------------------------------------------
def _get_process_context(dataset_name: str) -> str:
    """Short, human readable description per dataset."""
    name = dataset_name.lower()

    if "helpdesk" in name:
        return (
            "Process: helpdesk incident management. Each case is one ticket opened by a\n"
            "customer and handled by support until it is resolved or closed."
        )

    if "bpic2012" in name or "bpi2012" in name:
        return (
            "Process: loan or application handling from the BPI Challenge 2012 logs. Each\n"
            "case is one application that is checked and then accepted or rejected."
        )

    # Generic fallback
    return (
        "Process: generic business process recorded as an event log. Each case is one\n"
        "process instance with a sequence of activities over time."
    )


def _build_user_prompt(
    dataset_name: str,
    case_id: Any,
    prefix_activities: List[str],
    predicted_activity: str,
    pred_prob: float,
    top_alternatives: List[Tuple[str, float]],
    important_events: List[Tuple[int, str, float]],
) -> str:
    # Prefix as numbered list (1 based indices, easier for humans)
    if prefix_activities:
        prefix_lines = [f"{i + 1}. {act}" for i, act in enumerate(prefix_activities)]
        prefix_block = "\n".join(prefix_lines)
        prefix_len = len(prefix_activities)
    else:
        prefix_block = "(prefix is empty)"
        prefix_len = 0

    # Alternatives as bullet list
    if top_alternatives:
        alt_lines = [f"- {label} (p={prob:.2f})" for (label, prob) in top_alternatives]
        alt_block = "\n".join(alt_lines)
    else:
        alt_block = "No alternative predictions are available."

    # Attention based importance
    if important_events:
        imp_lines = []
        for pos, act, score in important_events:
            # pos is zero based, show 1 based index to the user
            imp_lines.append(
                f"- Event {pos + 1}: {act} (relative importance {score:.2f})"
            )
        importance_block = "\n".join(imp_lines)
    else:
        importance_block = "No attention information is available for this prefix."

    process_context = _get_process_context(dataset_name)

    user_prompt = f"""
[SECTION 1 - META INFORMATION]
Dataset: {dataset_name}
Case ID: {case_id}
User type: Process analyst (no ML background)

[SECTION 2 - TASK IN ONE SENTENCE]
Explain the model's local decision behaviour for this specific case prefix by
describing how its attention to past events and the given prediction scores lead
it to favour the predicted next activity over the alternative activities.

[SECTION 3 - PURPOSE AND GOALS]
The explanation should focus on local model behaviour for this single case prefix.
More concretely, it should:
- show which past events in this prefix were most influential according to the
  attention-based importance scores,
- explain how these locally important events support (or weaken) the predicted
  next activity from a process point of view,
- and mention any remaining uncertainty or plausible alternative next activities,
  if the probabilities suggest that.
The goal is to make the model's behaviour on this specific prefix understandable
to a process analyst, not to describe the model in general.

[SECTION 4 - PROCESS CONTEXT]
{process_context}

[SECTION 5 - MODEL INPUT AND SIGNALS]
Prediction task: next activity prediction
Model: ProcessTransformer
Prefix length: {prefix_len} events
Predicted next activity: {predicted_activity} (p={pred_prob:.2f})

Top alternative next activities (if any):
{alt_block}

Case prefix (oldest to newest):
{prefix_block}

Attention-based importance (most influential past events):
{importance_block}

[SECTION 6 - OUTPUT FORMAT AND INSTRUCTIONS]
Write a single coherent explanation as one block of text.

Start with a short summary in two or three sentences that:
- states the predicted next activity and its probability, and
- briefly describes which one to three past events the model attended to most
  according to the attention based importance list, and why this makes the
  prediction plausible for this case.

Then continue in the same block with a more detailed explanation that:
- refers to influential events by their index and activity name
  (for example "Event 19: o_created"),
- explains how these events influence the prediction, linking back to the
  attention based relative importance scores in the prompt,
- and comments on uncertainty or plausible alternative next activities if the
  probabilities or attention pattern suggest them.

Do not introduce headings or numbered sections. Avoid machine learning jargon.
Base all statements only on the information given in the prompt, and say
explicitly when evidence for an interpretation is weak or ambiguous.
"""
    return user_prompt


# ---------------------------------------------------------------------------
# Backend 1: GPT (OpenAI)
# ---------------------------------------------------------------------------
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
    model = model_name or os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def _get_llmama_pipe(model_name: Optional[str]):
    """Create or reuse a local LLaMA chat pipeline."""
    global _llmama_pipe

    if _llmama_pipe is not None:
        return _llmama_pipe

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


    _llmama_pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",           # CPU
        torch_dtype=torch.float32,   # safe on CPU
    )
    return _llmama_pipe


def _call_llmama(prompt: str, model_name: Optional[str]) -> str:
    pipe = _get_llmama_pipe(model_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=400,
    )

    generated_messages = outputs[0]["generated_text"]
    last_msg = generated_messages[-1]

    if isinstance(last_msg, dict) and "content" in last_msg:
        return last_msg["content"].strip()
    else:
        return str(last_msg).strip()

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
    """Build the structured prompt and call the selected LLM backend.

    backend:
        "gpt"    -> OpenAI models, using OPENAI_API_KEY and OPENAI_MODEL_NAME
        "llmama" -> LLMama server, using LLMAMA_BASE_URL, LLMAMA_API_KEY, LLMAMA_MODEL_NAME
    """
    prompt = _build_user_prompt(
        dataset_name=dataset_name,
        case_id=case_id,
        prefix_activities=prefix_activities,
        predicted_activity=predicted_activity,
        pred_prob=pred_prob,
        top_alternatives=top_alternatives,
        important_events=important_events,
    )

    b = backend.lower()
    if b == "gpt":
        return _call_gpt(prompt, llm_model_name)
    if b == "llmama":
        return _call_llmama(prompt, llm_model_name)

    raise ValueError(f"Unknown backend '{backend}'. Use 'gpt' or 'llmama'.")
