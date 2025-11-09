#!/usr/bin/env python3
import argparse, os, numpy as np, shap, matplotlib.pyplot as plt

def load_arrays(run_dir):
    sv   = np.load(os.path.join(run_dir, "shap_values_batch.npy"))
    X    = np.load(os.path.join(run_dir, "x_ids_batch.npy"))
    base = np.load(os.path.join(run_dir, "base_values.npy"))
    return sv, X, base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="../outputs/Helpdesk/shap")
    ap.add_argument("--save", action="store_true", default=True, help="save figs next to arrays")
    args = ap.parse_args()



    sv, X, base = load_arrays(args.run_dir)
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



    # 1) One sample (bar)
    plt.figure()
    shap.plots.bar(exp[sample_idx], max_display=T, show=False)
    if args.save:
        plt.savefig(os.path.join(args.run_dir, f"shap_bar_idx{sample_idx}.png"), bbox_inches="tight", dpi=160)
    else:
        plt.show()

    # 2) One sample (waterfall)
    plt.figure()
    shap.plots.waterfall(exp[sample_idx], max_display=T, show=False)
    if args.save:
        plt.savefig(os.path.join(args.run_dir, f"shap_waterfall_idx{sample_idx}.png"), bbox_inches="tight", dpi=160)
    else:
        plt.show()

    # 3) Overall (summary: beeswarm)
    plt.figure()
    shap.summary_plot(sv, features=None, feature_names=feat_names, show=False)
    if args.save:
        plt.savefig(os.path.join(args.run_dir, "shap_summary_beeswarm.png"), bbox_inches="tight", dpi=160)
    else:
        plt.show()

    # 4) Overall (summary: bar of mean |SHAP|)
    plt.figure()
    shap.summary_plot(sv, features=None, feature_names=feat_names, plot_type="bar", show=False)
    if args.save:
        plt.savefig(os.path.join(args.run_dir, "shap_summary_bar.png"), bbox_inches="tight", dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    main()
