#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

# İstersen attention'ın label okuma yardımcılarını kullanabilirsin:
# from attention_viz_utils import choose_labels_source  # repo'nda zaten var

def _read_lines(p: Path) -> Optional[list]:
    if not p.exists(): return None
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def _labels_from_prefixes_txt(prefixes_txt: Optional[Path], i: int, L: int) -> Optional[List[str]]:
    if prefixes_txt is None or not prefixes_txt.exists(): return None
    lines = _read_lines(prefixes_txt)
    if not lines or i >= len(lines): return None
    toks = lines[i].split()
    # sağa hizalı label’lar (attention’daki ile aynı görünsün)
    return (["[PAD]"] * (L - len(toks)) + toks) if len(toks) < L else toks[-L:]

def _labels_from_ids(x_ids_row: np.ndarray, L: int, inv_x: Optional[dict], pad_id: int = 0) -> List[str]:
    row = x_ids_row[:L]
    if inv_x is None:
        # id → string
        return [str(int(t)) for t in row]
    out = []
    for t in row:
        t = int(t)
        out.append("[PAD]" if t == pad_id else inv_x.get(t, f"<{t}>"))
    return out

def render_shap_single(
    shap_values_row: np.ndarray,      # (L,) veya (maxlen,) -- L kadarını kullanacağız
    x_ids_row: np.ndarray,            # (maxlen,)
    L: int,                           # prefix length (trim sonrası uzunluk)
    out_dir: str,
    *,
    title_prefix: str = "",
    tokens: Optional[List[str]] = None,
    cmap: str = "coolwarm",
) -> str:
    """Tek örnek SHAP ısı haritası (1 satır)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vals = shap_values_row[:L]
    toks = tokens if tokens is not None else [str(i) for i in range(L)]

    data = vals[None, :]  # (1, L)
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(toks)), 1.8))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_yticks([])  # tek satır
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks, rotation=45, ha="right")
    ax.set_title(f"{title_prefix}SHAP (L={L})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png = Path(out_dir) / "shap_single.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return str(out_png)
