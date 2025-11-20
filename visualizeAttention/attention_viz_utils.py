#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for saving batch metadata and visualizing attention maps.

Used by:
- get_attention_hooked.py  -> save_batch_metadata(...)
- visualization.py         -> render_attention_entry(...)
"""

from typing import List, Optional, Union, Sequence
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(p: Union[str, Path]) -> Path:
    """Create directory if it does not exist and return it as Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _right_align(labels: Sequence[str], L: int, pad_token: str) -> List[str]:
    """
    Left-pad labels with pad_token so that len(labels) == L.
    If labels is longer than L, keep only the last L entries.
    """
    labels = list(map(str, labels))
    if len(labels) >= L:
        return labels[-L:]
    pad_len = L - len(labels)
    return [pad_token] * pad_len + labels


def _load_labels(batch_txt: Optional[str], sample_idx: int) -> Optional[List[str]]:
    """
    Load the token sequence for a single sample from batch_prefixes.txt.

    Each line in batch_prefixes.txt is expected to be a whitespace-separated list
    of activity labels for that sample (no padding needed).
    """
    if batch_txt is None:
        return None
    path = Path(batch_txt)
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    if not (0 <= sample_idx < len(lines)):
        return None

    line = lines[sample_idx].strip()
    if not line:
        return None
    return line.split()


# ---------------------------------------------------------------------------
# 1) Metadata saving
# ---------------------------------------------------------------------------

def save_batch_metadata(
    out_dir: Union[str, Path],
    prefix_texts: Sequence[str],
    case_ids: Sequence[Union[str, int]],
    pad_token: str = "[PAD]",
    align_right: bool = True,
) -> None:
    """
    Save human-readable metadata for the current batch.

    Parameters
    ----------
    out_dir:
        Directory where metadata files will be written.
    prefix_texts:
        List of strings, each the decoded prefix tokens for one sample
        (already space-separated, e.g. "a_submitted a_partlysubmitted").
    case_ids:
        List of case ids aligned with prefix_texts.
    pad_token, align_right:
        Kept for backward compatibility. Padding/alignment is handled
        during visualization.
    """
    out = _ensure_dir(out_dir)

    # One prefix per line
    prefixes_path = out / "batch_prefixes.txt"
    with prefixes_path.open("w", encoding="utf-8") as f:
        for p in prefix_texts:
            f.write(str(p).strip() + "\n")

    # One case id per line
    case_ids_path = out / "batch_case_ids.txt"
    with case_ids_path.open("w", encoding="utf-8") as f:
        for cid in case_ids:
            f.write(str(cid) + "\n")

    # Optional JSON with both together (handy for debugging)
    meta = [
        {"index": i, "case_id": str(cid), "prefix": str(p).strip()}
        for i, (p, cid) in enumerate(zip(prefix_texts, case_ids))
    ]
    meta_path = out / "batch_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 2) Attention visualization
# ---------------------------------------------------------------------------

def render_attention_entry(
    scores_np: np.ndarray,
    out_dir: Union[str, Path],
    sample_idx: int,
    batch_txt: Optional[str] = None,
    pad_token: str = "[PAD]",
    align_right: bool = True,
    title_prefix: str = "",
) -> None:
    """
    Plot attention maps for a single sample.

    Parameters
    ----------
    scores_np:
        Numpy array with shape [B, H, T, T].
    out_dir:
        Directory where plots will be stored.
    sample_idx:
        Index in the batch (0-based).
    batch_txt:
        Path to batch_prefixes.txt. If provided, labels will be taken from here
        and used both for axis ticks and to crop away PAD rows/columns.
    pad_token:
        String used as PAD label in plots (default "[PAD]").
    align_right:
        If True, labels will be right-aligned to the sequence length T, with
        pad_token padded on the left. This assumes left-padding in X.
    title_prefix:
        Optional prefix string for the plot titles.
    """
    out = _ensure_dir(out_dir)

    if scores_np.ndim != 4:
        raise ValueError(f"Expected scores_np with 4 dims [B,H,T,T], got shape {scores_np.shape}")

    B, H, Tq, Tk = scores_np.shape
    if Tq != Tk:
        raise ValueError(f"Only square attention maps supported, got Tq={Tq}, Tk={Tk}")
    if not (0 <= sample_idx < B):
        raise IndexError(f"sample_idx {sample_idx} is out of range for batch size {B}")

    # Slice scores for this sample: [H, T, T]
    S = np.array(scores_np[sample_idx], copy=True)

    # Prepare labels and which indices correspond to real tokens
    labels: Optional[List[str]] = _load_labels(batch_txt, sample_idx) if batch_txt else None

    if labels is not None:
        if align_right:
            labels = _right_align(labels, Tq, pad_token)
        else:
            labels = labels[:Tq]

        # Keep only non-PAD positions on both axes
        keep_idx = [i for i, lab in enumerate(labels) if lab != pad_token]
        if keep_idx:
            S = S[:, keep_idx, :][:, :, keep_idx]
            labels = [labels[i] for i in keep_idx]
        else:
            # all PAD; fall back to full matrix without labels
            labels = None

    # Head-averaged matrix: [L_eff, L_eff]
    S_avg = S.mean(axis=0)
    L_eff = S_avg.shape[0]
    H_eff = S.shape[0]

    prefix = title_prefix or ""

    # --- Head-averaged heatmap ---
    plt.figure(figsize=(7, 5))
    plt.imshow(S_avg, aspect="auto")
    plt.title(f"{prefix}Head-averaged (sample {sample_idx})")
    plt.xlabel("Key index j (past events)")
    plt.ylabel("Query index i (current step)")
    if labels is not None:
        ax = plt.gca()
        ax.set_xticks(range(L_eff))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(L_eff))
        ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(Path(out) / f"sample{sample_idx}_avg_heatmap.png", dpi=160)
    plt.close()

    # --- Per-head heatmaps ---
    for h in range(H_eff):
        plt.figure(figsize=(7, 5))
        plt.imshow(S[h], aspect="auto")
        plt.title(f"{prefix}Sample {sample_idx} Head {h}")
        plt.xlabel("Key index j (past events)")
        plt.ylabel("Query index i (current step)")
        if labels is not None:
            ax = plt.gca()
            ax.set_xticks(range(L_eff))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticks(range(L_eff))
            ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(Path(out) / f"sample{sample_idx}_head{h}_heatmap.png", dpi=160)
        plt.close()
