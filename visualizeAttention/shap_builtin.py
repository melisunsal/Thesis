#!/usr/bin/env python3
import argparse, os, numpy as np, shap, matplotlib.pyplot as plt
from pathlib import Path


def load_arrays(run_dir):
    sv   = np.load(os.path.join(run_dir, "shap_values_batch.npy"))
    X    = np.load(os.path.join(run_dir, "x_ids_batch.npy"))
    base = np.load(os.path.join(run_dir, "base_values.npy"))
    return sv, X, base

def main(run_dir):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", default=True, help="save figs next to arrays")
    args = ap.parse_args()



    sv, X, base = load_arrays(run_dir)
    B, T = sv.shape

    try:
        sample_idx = int(input(f"Enter sample index [0-{B-1}]: "))
    except ValueError:
        print("Invalid input. Exiting.")
        return
    if sample_idx < 0 or sample_idx >= B:
        print("Out of range.")
        return
    print(X[sample_idx])
    # Build a minimal Explanation for selected class
    feat_names = [f"pos_{j-(T-1):+d}" for j in range(T)]  # e.g., pos_-13 ... pos_0 (last)
    exp = shap.Explanation(values=sv, base_values=base, data=X, feature_names=feat_names)

    PAD_ID = 0  # change if needed
    row = X[sample_idx]
    L = int((row != PAD_ID).sum())
    start = T - L  # right-aligned active zone

    # try to read token strings saved by attention pipeline
    base_dir = Path(run_dir).parent
    tokens_txt = base_dir / "batch_prefixes.txt"
    tokens = None
    if tokens_txt.exists():
        lines = tokens_txt.read_text(encoding="utf-8").splitlines()
        if 0 <= sample_idx < len(lines):
            toks_raw = lines[sample_idx].strip().split()
            # right-align to length L
            tokens = (["[PAD]"] * (L - len(toks_raw)) + toks_raw) if len(toks_raw) < L else toks_raw[-L:]

    # build per-sample feature names over observed positions only
    pos_names = [f"pos_{k}" for k in range(-L, 0)]
    if tokens is not None:
        feat_names_row = [f"{tok} @ {p}" for tok, p in zip(tokens, pos_names)]
    else:
        feat_names_row = pos_names

    # build a one-row Explanation trimmed to observed positions
    exp_row = shap.Explanation(
        values=sv[sample_idx, start:],  # (L,)
        base_values=base[sample_idx],  # scalar for this class
        data=row[start:],  # (L,)
        feature_names=feat_names_row
    )

    # 1) bar
    plt.figure()
    shap.plots.bar(exp_row, max_display=L, show=False)
    plt.savefig(os.path.join(run_dir, f"shap_bar_idx{sample_idx}.png"), bbox_inches="tight", dpi=160)

    # 2) waterfall
    plt.figure()
    shap.plots.waterfall(exp_row, max_display=L, show=False)
    plt.savefig(os.path.join(run_dir, f"shap_waterfall_idx{sample_idx}.png"), bbox_inches="tight", dpi=160)

    # 3) Overall (summary: beeswarm)
    plt.figure()
    shap.summary_plot(sv, features=None, feature_names=feat_names, show=False)
    if args.save:
        plt.savefig(os.path.join(run_dir, "shap_summary_beeswarm.png"), bbox_inches="tight", dpi=160)
    else:
        plt.show()

    # 4) Overall (summary: bar of mean |SHAP|)
    plt.figure()
    shap.summary_plot(sv, features=None, feature_names=feat_names, plot_type="bar", show=False)
    if args.save:
        plt.savefig(os.path.join(run_dir, "shap_summary_bar.png"), bbox_inches="tight", dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    # main("../outputs/BPIC2012-A/shap")
    # main("../outputs/BPIC2012-A/shap_small_background")
    # main("../outputs/BPIC2012-A/shap_too_much_sample")
    # main("../outputs/BPIC2012-A/shap_small_background_big_sample")
    #
    main("../outputs/BPIC2012-O/shap")
    main("../outputs/BPIC2012-O/shap_small_background")
    # main("../outputs/BPIC2012-O/shap_too_much_sample")
    # main("../outputs/BPIC2012-O/shap_small_background_big_sample")

    # main("../outputs/BPIC2012-W/shap")
    # main("../outputs/BPIC2012-W/shap_small_background")
    # main("../outputs/BPIC2012-W/shap_too_much_sample")
    # main("../outputs/BPIC2012-W/shap_small_background_big_sample")




