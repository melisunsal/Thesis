#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attention verification script (PAD-aware).

Checks, for a given run directory containing:
 - block_mha_scores.npy  (shape [B, H, T, T])
 - batch_prefixes.txt    (one space-separated prefix per line)

Metrics:
 1) In non-PAD region (cropped L×L block):
    - max row-sum deviation from 1
    - total future-leak mass (i<j)

 2) For non-PAD queries to PAD keys (on full T×T scores):
    - total mass
    - max mass per row

Assumes left-padding: real tokens are right-aligned, prefixes in
batch_prefixes.txt contain only real activities (no PAD tokens).
"""

import sys
from pathlib import Path
import numpy as np


def verify_attention(run_dir, verbose=True):
    run_dir = Path(run_dir)
    scores_path = run_dir / "block_mha_scores.npy"
    prefixes_path = run_dir / "batch_prefixes.txt"

    if not scores_path.exists():
        raise FileNotFoundError(f"Could not find {scores_path}")
    if not prefixes_path.exists():
        raise FileNotFoundError(f"Could not find {prefixes_path}")

    scores = np.load(scores_path)  # [B, H, T, T]
    lines = prefixes_path.read_text(encoding="utf-8").splitlines()

    B, H, Tq, Tk = scores.shape
    assert Tq == Tk, f"Attention maps must be square, got {scores.shape}"
    if len(lines) < B:
        raise ValueError(
            f"batch_prefixes.txt has fewer lines ({len(lines)}) than batch size B={B}"
        )

    if verbose:
        print(f"[verify] Run dir: {run_dir}")
        print(f"[verify] scores shape: B={B}, H={H}, T={Tq}")

    # --- 1) Non-PAD bölge (cropped L×L blok) ---
    max_row_sum_dev_nonpad = 0.0
    total_future_leak_nonpad = 0.0
    counted_rows = 0

    # --- 2) Non-PAD query -> PAD key mass (full T×T) ---
    total_nonpad_to_pad_mass = 0.0
    max_row_nonpad_to_pad_mass = 0.0

    for b in range(B):
        line = lines[b].strip()
        if not line:
            continue
        labels = line.split()
        L = len(labels)

        # Güvenlik: prefix uzunluğu T'den büyükse, son T token'i al
        if L > Tq:
            labels = labels[-Tq:]
            L = len(labels)

        # Left padding: gerçek token indeksleri [start .. Tq-1]
        start = Tq - L
        real_idx = np.arange(start, Tq)

        # -------------------------------
        # 1) Non-PAD blok: [H, L, L]
        # -------------------------------
        S_block = scores[b][:, real_idx][:, :, real_idx]  # [H, L, L]

        # Satır toplamları (her head + her query için)
        row_sums = S_block.sum(axis=-1)  # [H, L]
        dev = np.abs(row_sums - 1.0)
        max_row_sum_dev_nonpad = max(max_row_sum_dev_nonpad, float(dev.max()))
        counted_rows += H * L

        # Geleceğe kaçak: j > i (üst üçgen, diagonal hariç)
        L_eff = L
        triu_mask = np.triu(np.ones((L_eff, L_eff), dtype=bool), k=1)  # (i<j)
        future_leak = S_block[:, triu_mask].sum()
        total_future_leak_nonpad += float(future_leak)

        # -------------------------------
        # 2) Non-PAD query -> PAD key (full T×T)
        # -------------------------------
        if start > 0:
            S_full = scores[b]  # [H, T, T]
            pad_cols = np.arange(0, start)  # PAD key indeksleri

            # her head, her non-pad query satırı için PAD kolonlarına giden mass
            for h in range(H):
                for i in real_idx:
                    row_pad_mass = float(S_full[h, i, pad_cols].sum())
                    total_nonpad_to_pad_mass += row_pad_mass
                    if row_pad_mass > max_row_nonpad_to_pad_mass:
                        max_row_nonpad_to_pad_mass = row_pad_mass

    # ------------ Sonuçları yazdır ------------
    print(f"[verify] Max row-sum deviation (non-pad L×L): {max_row_sum_dev_nonpad:.3e}")
    print(f"[verify] Total future-leak mass (non-pad L×L): {total_future_leak_nonpad:.6f}")
    print(f"[verify] Total mass non-pad query -> PAD keys: {total_nonpad_to_pad_mass:.6f}")
    print(f"[verify] Max row mass non-pad -> PAD keys:     {max_row_nonpad_to_pad_mass:.6f}")

    # Basit karar
    ok_row = max_row_sum_dev_nonpad < 1e-5
    ok_future = total_future_leak_nonpad < 1e-5
    ok_pad = total_nonpad_to_pad_mass < 1e-5 and max_row_nonpad_to_pad_mass < 1e-5

    if ok_row and ok_future and ok_pad:
        print("[verify][OK] Non-pad region is causal & normalized; non-pad -> PAD mass ~ 0.")
    else:
        print("[verify][WARN] Some attention anomalies detected, inspect the metrics above.")

    return {
        "max_row_sum_dev_nonpad": max_row_sum_dev_nonpad,
        "total_future_leak_nonpad": total_future_leak_nonpad,
        "total_nonpad_to_pad_mass": total_nonpad_to_pad_mass,
        "max_row_nonpad_to_pad_mass": max_row_nonpad_to_pad_mass,
    }


if __name__ == "__main__":
    # Kullanım:
    #   python verify_attention.py ../outputs/BPIC2012-O
    # veya argüman vermezsen varsayılan klasör:
    default_dir = "../outputs/BPIC2012-O"

    if len(sys.argv) > 1:
        run_dir_arg = sys.argv[1]
    else:
        run_dir_arg = default_dir

    verify_attention(run_dir_arg)
