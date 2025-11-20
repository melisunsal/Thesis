#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from attention_viz_utils import render_attention_entry

def main():
    # Sabit dizin
    base_dir = Path("./outputs/BPIC2012-O")
    scores_path = base_dir / "block_mha_scores.npy"
    batch_txt_path = base_dir / "batch_prefixes.txt"
    case_ids_path = base_dir / "batch_case_ids.txt"
    output_dir = Path("./outputs/BPIC2012-O/attentionMaps")
    # Yükle
    scores = np.load(scores_path)
    print(f"Loaded attention scores: shape={scores.shape}")

    # Case id’ler (varsa)
    case_ids = case_ids_path.read_text(encoding="utf-8").splitlines() if case_ids_path.exists() else None
    prefixes = batch_txt_path.read_text(encoding="utf-8").splitlines() if batch_txt_path.exists() else None

    # Kullanıcıdan hangi sample’ı göstereceğini iste
    try:
        sample_idx = int(input(f"Enter sample index [0-{scores.shape[0]-1}]: "))
    except ValueError:
        print("Invalid input. Exiting.")
        return
    if sample_idx < 0 or sample_idx >= scores.shape[0]:
        print("Out of range.")
        return

    # Başlık: Case ID + Prefix
    title_prefix = ""
    if case_ids and 0 <= sample_idx < len(case_ids):
        title_prefix = f"Case {case_ids[sample_idx]} – "
    if prefixes and 0 <= sample_idx < len(prefixes):
        print(f"\nPrefix for sample {sample_idx}: {prefixes[sample_idx]}\n")

    # Görselleştir (aynı klasöre kaydedilecek)
    render_attention_entry(
        scores_np=scores,
        out_dir=str(output_dir),
        sample_idx=sample_idx,
        batch_txt=str(batch_txt_path) if batch_txt_path.exists() else None,
        pad_token="[PAD]",
        align_right=True,
        title_prefix=title_prefix,
    )
    print(f"✅ Saved plots to: {base_dir}")

if __name__ == "__main__":
    main()
