#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (content shortened in this retry to avoid environment flakiness)
from typing import List, Optional, Union, Sequence
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _right_align(labels, L, pad_token):
    labels = list(map(str, labels))
    return ([pad_token] * (L - len(labels)) + labels) if len(labels) < L else labels[-L:]

def save_batch_metadata(out_dir, prefix_texts=None, case_ids=None, pad_token="<pad>", align_right=True):
    out_dir = _ensure_dir(out_dir)
    if prefix_texts is not None:
        lines = []
        for item in prefix_texts:
            if isinstance(item, str):
                lines.append(item.strip())
            else:
                lines.append(" ".join(map(str, item)))
        (out_dir / "batch_prefixes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if case_ids is not None:
        ids = list(map(str, case_ids))
        (out_dir / "batch_case_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    meta = {"pad_token": pad_token, "align_right": bool(align_right)}
    (out_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def _labels_from_txt(batch_txt, B, L, pad_token, align_right):
    p = Path(batch_txt)
    if not p.exists(): return None
    lines = p.read_text(encoding="utf-8").splitlines()
    labs = []
    for s in lines[:B]:
        toks = s.strip().split()
        labs.append(_right_align(toks, L, pad_token) if align_right else toks[:L] + [pad_token]*(L-len(toks)))
    if len(labs) < B: labs += [[pad_token]*L for _ in range(B-len(labs))]
    return labs

def _labels_from_csv(labels_csv, B, L, pad_token, align_right):
    p = Path(labels_csv)
    if not p.exists(): return None
    import pandas as pd
    df = pd.read_csv(p)
    col = "prefix" if "prefix" in df.columns else df.columns[0]
    lines = df[col].astype(str).tolist()
    labs = []
    for s in lines[:B]:
        toks = s.strip().split()
        labs.append(_right_align(toks, L, pad_token) if align_right else toks[:L] + [pad_token]*(L-len(toks)))
    if len(labs) < B: labs += [[pad_token]*L for _ in range(B-len(labs))]
    return labs

def choose_labels_source(B, L, batch_txt=None, labels_csv=None, pad_token="<pad>", align_right=True):
    labs = None
    if batch_txt: labs = _labels_from_txt(batch_txt, B, L, pad_token, align_right)
    if labs is None and labels_csv:
        labs = _labels_from_csv(labels_csv, B, L, pad_token, align_right)
    return labs

def render_attention(scores_np, out_dir, sample_idx=0, labels=None, title_prefix=""):
    out_dir = _ensure_dir(out_dir)
    B,H,L,L2 = scores_np.shape
    assert L==L2 and 0<=sample_idx<B
    S = scores_np[sample_idx]      # [H,L,L]
    Savg = S.mean(axis=0)          # [L,L]

    plt.figure(figsize=(7,5))
    plt.imshow(Savg, aspect="auto")
    plt.title(f"{title_prefix}Head-averaged (sample {sample_idx})")
    plt.xlabel("Key index j (past events)"); plt.ylabel("Query index i (current step)")
    if labels is not None:
        ax = plt.gca()
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(); plt.tight_layout(); plt.savefig(Path(out_dir)/f"sample{sample_idx}_avg_heatmap.png", dpi=160); plt.close()

    H = S.shape[0]
    for h in range(H):
        plt.figure(figsize=(7,5))
        plt.imshow(S[h], aspect="auto")
        plt.title(f"{title_prefix}Sample {sample_idx} Head {h}")
        plt.xlabel("Key index j (past events)"); plt.ylabel("Query index i (current step)")
        if labels is not None:
            ax = plt.gca()
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(); plt.tight_layout(); plt.savefig(Path(out_dir)/f"sample{sample_idx}_head{h}_heatmap.png", dpi=160); plt.close()

    # Savg: [L, L] (head-averaged attention matrix)
    L = Savg.shape[0]

    # heatmap'te kullandığın x-ekseni etiketlerini L uzunluğunda üret:
    # align_right=True ise: [PAD,...,PAD, token1, token2, ..., token_t]
    tokens = [lbl for lbl in labels if lbl != "[PAD]"]  # PAD'siz gerçek tokenlar
    t = len(tokens)
    xlabels_full = (["[PAD]"] * (L - t)) + tokens  # uzunluk L

    # Son query satırı (en alt satır)
    last_row = Savg[-1]  # shape [L]

    # Geçerli kolon aralığı: sağdan t sütun
    start = L - t
    cols = list(range(start, L))

    # Bar için değerler ve etiketler (HEATMAP İLE AYNI SIRA)
    vals = last_row[start:L].astype(float)  # shape [t]
    # İstersen normalize et:
    s = float(vals.sum());
    vals = vals / (s + 1e-12)

    bar_labels = xlabels_full[start:L]  # örn: ['register','classify','analyze']

    # Debug: sayıları da yaz
    print("Last-step (i=L-1) attention:")
    for lbl, v in zip(bar_labels, vals):
        print(f"  {lbl:>10s}: {v:.4f}")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(vals)), vals)
    ax = plt.gca()
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(bar_labels, rotation=90, fontsize=10)
    plt.title(f"{title_prefix}Last step attention (sample {sample_idx})")
    plt.xlabel("Key index j (past events)")
    plt.ylabel("Attention weight")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"sample{sample_idx}_last_step_bar.png", dpi=160)
    plt.close()

    return str(out_dir)

def render_attention_entry(scores_np, out_dir, sample_idx=0, batch_txt=None, labels_csv=None, pad_token="<pad>", align_right=True, title_prefix=""):
    B,H,L,L2 = scores_np.shape
    labels_list = choose_labels_source(B, L, batch_txt=batch_txt, labels_csv=labels_csv, pad_token=pad_token, align_right=align_right)
    labels = labels_list[sample_idx] if labels_list is not None else None
    return render_attention(scores_np, out_dir, sample_idx=sample_idx, labels=labels, title_prefix=title_prefix)

def render_each_head(scores_np, out_dir, sample_idx=0, labels=None, title_prefix=""):
    import matplotlib.pyplot as plt
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    S = scores_np[sample_idx]  # [H, L, L]
    H, L, _ = S.shape
    for h in range(H):
        plt.figure(figsize=(6,4))
        plt.imshow(S[h], aspect="auto")
        plt.title(f"{title_prefix}Head {h} | sample {sample_idx}")
        plt.xlabel("Key j"); plt.ylabel("Query i")
        if labels is not None:
            ax = plt.gca()
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        plt.tight_layout()
        plt.savefig(out / f"sample{sample_idx}_head{h}.png", dpi=160)
        plt.close()

