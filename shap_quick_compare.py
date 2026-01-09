"""
Quick SHAP Configuration Comparison
===================================

Simplified script to quickly compare key SHAP configurations.
Focuses on the most impactful design choices.

Usage:
    python shap_quick_compare.py --dataset BPIC2012-W --prefix_index 12
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def pad_left(tokens, max_len, pad_id):
    result = np.full(max_len, pad_id, dtype=np.int32)
    L = len(tokens)
    if L > 0:
        result[-L:] = tokens[-L:]
    return result


# =============================================================================
# MASKING FUNCTIONS
# =============================================================================

def deletion_mask(tokens, mask, max_len, pad_id):
    """Your current: delete and re-pad."""
    kept = tokens[mask.astype(bool)]
    return pad_left(kept, max_len, pad_id)


def pad_replace_mask(tokens, mask, max_len, pad_id):
    """Attention Please style: PAD in place."""
    result = pad_left(tokens, max_len, pad_id)
    L = len(tokens)
    start = max_len - L
    for i, m in enumerate(mask):
        if m < 0.5:
            result[start + i] = pad_id
    return result


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def output_logit(logits, target_class):
    """Your current: raw logit."""
    return logits[target_class]


def output_probability(logits, target_class):
    """Alternative: softmax probability."""
    probs = softmax(logits)
    return probs[target_class]


def output_log_odds(logits, target_class):
    """Alternative: log odds vs second best."""
    probs = softmax(logits)
    sorted_idx = np.argsort(-probs)
    second = sorted_idx[1] if sorted_idx[0] == target_class else sorted_idx[0]
    eps = 1e-10
    return np.log(probs[target_class] + eps) - np.log(probs[second] + eps)


# =============================================================================
# SHAP COMPUTATION
# =============================================================================

def compute_shap(model, tokens, max_len, pad_id, target_class,
                 mask_fn, output_fn, n_perm=500):
    """
    Compute SHAP values with specified masking and output functions.
    """
    L = len(tokens)
    phi = np.zeros(L)
    
    # Value function
    def f(mask):
        X = mask_fn(tokens, mask, max_len, pad_id).reshape(1, -1)
        logits = model.predict(X, verbose=0)[0]
        return output_fn(logits, target_class)
    
    # Permutation SHAP
    for _ in range(n_perm):
        perm = np.random.permutation(L)
        mask = np.zeros(L)
        prev_val = f(mask)
        
        for idx in perm:
            mask[idx] = 1.0
            curr_val = f(mask)
            phi[idx] += curr_val - prev_val
            prev_val = curr_val
    
    phi /= n_perm
    
    # Get baseline and full values
    base_val = f(np.zeros(L))
    full_val = f(np.ones(L))
    
    return {
        "phi": phi,
        "base_value": base_val,
        "full_value": full_val,
        "reconstructed": base_val + phi.sum(),
        "additivity_error": abs(base_val + phi.sum() - full_val)
    }


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(model, tokens, max_len, pad_id, target_class, 
                   attention=None, n_perm=500):
    """
    Run 4 key configurations and compare.
    """
    configs = {
        "current (deletion+logit)": {
            "mask_fn": deletion_mask,
            "output_fn": output_logit
        },
        "pad_replace+logit": {
            "mask_fn": pad_replace_mask,
            "output_fn": output_logit
        },
        "deletion+probability": {
            "mask_fn": deletion_mask,
            "output_fn": output_probability
        },
        "deletion+log_odds": {
            "mask_fn": deletion_mask,
            "output_fn": output_log_odds
        }
    }
    
    results = {}
    
    for name, cfg in configs.items():
        print(f"\nComputing: {name}...")
        
        result = compute_shap(
            model=model,
            tokens=tokens,
            max_len=max_len,
            pad_id=pad_id,
            target_class=target_class,
            mask_fn=cfg["mask_fn"],
            output_fn=cfg["output_fn"],
            n_perm=n_perm
        )
        
        # Add attention correlation if available
        if attention is not None:
            L = len(tokens)
            T = attention.shape[1]
            start = T - L
            attn = attention[:, T-1, start:].mean(axis=0)
            attn = attn / (attn.sum() + 1e-10)
            
            abs_phi = np.abs(result["phi"])
            result["attn_pearson"] = float(np.corrcoef(attn, abs_phi)[0, 1])
            
            # Top-5 overlap
            k = min(5, L)
            top_attn = set(np.argsort(-attn)[:k])
            top_phi = set(np.argsort(-abs_phi)[:k])
            result["top5_overlap"] = len(top_attn & top_phi) / k
        
        results[name] = result
        
        print(f"  Additivity error: {result['additivity_error']:.2e}")
        if "attn_pearson" in result:
            print(f"  Attention correlation: {result['attn_pearson']:.3f}")
    
    return results


def plot_comparison(results, feature_names, save_path=None):
    """
    Create comparison visualization.
    """
    configs = list(results.keys())
    n_configs = len(configs)
    L = len(feature_names)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        phi = result["phi"]
        
        # Color by sign
        bar_colors = ['#e74c3c' if p < 0 else '#2ecc71' for p in phi]
        
        ax.barh(range(L), phi, color=bar_colors, alpha=0.8)
        ax.set_yticks(range(L))
        ax.set_yticklabels([f"E{i+1}" for i in range(L)], fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel("SHAP Value")
        ax.set_title(f"{name}\n(add_err={result['additivity_error']:.1e})")
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {save_path}")
    
    plt.close()
    
    # Also create ranking comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Get top-5 for each config
    top_features = {}
    for name, result in results.items():
        phi = np.array(result["phi"])
        top_idx = np.argsort(-np.abs(phi))[:5]
        top_features[name] = [(feature_names[i], phi[i]) for i in top_idx]
    
    # Print ranking comparison
    print("\n" + "="*70)
    print("TOP-5 FEATURE RANKING COMPARISON")
    print("="*70)
    
    for rank in range(5):
        print(f"\nRank #{rank+1}:")
        for name in configs:
            feat, val = top_features[name][rank]
            print(f"  {name:<25}: {feat:<30} (φ={val:+.3f})")


def print_summary_table(results):
    """Print summary comparison table."""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Configuration':<30} {'Add.Error':<12} {'Attn.Corr':<12} {'Top5 Ovlp':<12}")
    print("-"*66)
    
    for name, result in results.items():
        add_err = result['additivity_error']
        attn_corr = result.get('attn_pearson', float('nan'))
        top5 = result.get('top5_overlap', float('nan'))
        
        print(f"{name:<30} {add_err:<12.2e} {attn_corr:<12.3f} {top5:<12.1%}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BPIC2012-W")
    parser.add_argument("--prefix_index", type=int, default=12)
    parser.add_argument("--n_perm", type=int, default=500, help="Permutations per config")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    
    # Import your modules
    from processtransformer import constants
    from processtransformer.data import loader
    from processtransformer.models import transformer
    
    print(f"\n{'='*60}")
    print(f"SHAP CONFIGURATION COMPARISON")
    print(f"Dataset: {args.dataset}, Prefix Index: {args.prefix_index}")
    print(f"{'='*60}")
    
    # Load data
    dl = loader.LogsDataLoader(name=args.dataset)
    _, _, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = \
        dl.load_data(constants.Task.NEXT_ACTIVITY)
    
    pad_id = int(x_word_dict.get("[PAD]", x_word_dict.get("<pad>", 0)))
    inv_x = {v: k for k, v in x_word_dict.items()}
    inv_y = {v: k for k, v in y_word_dict.items()}
    
    # Load model
    model = transformer.get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size,
        output_dim=num_output
    )
    model.load_weights(f"./models/{args.dataset}/next_activity_ckpt").expect_partial()
    
    # Load prefix
    outputs_dir = Path(f"./outputs/{args.dataset}")
    with open(outputs_dir / "batch_predictions.json") as f:
        predictions = json.load(f)["predictions"]
    
    prefix_info = predictions[args.prefix_index]
    prefix_activities = prefix_info["prefix_activities"]
    tokens = np.array([x_word_dict[a] for a in prefix_activities], dtype=np.int32)
    
    print(f"\nPrefix ({len(tokens)} events):")
    print(f"  {' → '.join(prefix_activities[:6])}{'...' if len(prefix_activities) > 6 else ''}")
    
    # Get prediction
    X = pad_left(tokens, max_case_length, pad_id).reshape(1, -1)
    logits = model.predict(X, verbose=0)[0]
    target_class = int(np.argmax(logits))
    prob = softmax(logits)[target_class]
    print(f"\nPrediction: {inv_y.get(target_class)} (p={prob:.3f})")
    
    # Load attention
    attn_path = outputs_dir / "block_mha_scores.npy"
    attention = np.load(attn_path)[args.prefix_index] if attn_path.exists() else None
    
    # Run comparison
    np.random.seed(42)
    results = run_comparison(
        model=model,
        tokens=tokens,
        max_len=max_case_length,
        pad_id=pad_id,
        target_class=target_class,
        attention=attention,
        n_perm=args.n_perm
    )
    
    # Create feature names
    feature_names = [f"E{i+1}:{inv_x.get(t, t)}" for i, t in enumerate(tokens)]
    
    # Print summary
    print_summary_table(results)
    
    # Plot comparison
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = outputs_dir
    
    plot_path = out_dir / f"shap_comparison_batch_{args.prefix_index}.png"
    plot_comparison(results, feature_names, save_path=plot_path)
    
    # Save results
    results_path = out_dir / f"shap_comparison_batch_{args.prefix_index}.json"
    
    # Convert numpy to lists for JSON
    json_results = {}
    for name, result in results.items():
        json_results[name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in result.items()
        }
    
    with open(results_path, 'w') as f:
        json.dump({
            "dataset": args.dataset,
            "prefix_index": args.prefix_index,
            "prefix_activities": prefix_activities,
            "prediction": inv_y.get(target_class),
            "n_permutations": args.n_perm,
            "results": json_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Final recommendation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Find best additivity
    best_add = min(results.items(), key=lambda x: x[1]['additivity_error'])
    print(f"\n✓ Best additivity: {best_add[0]} (error={best_add[1]['additivity_error']:.2e})")
    
    # Check if rankings are consistent
    current_top3 = set(np.argsort(-np.abs(results["current (deletion+logit)"]["phi"]))[:3])
    
    consistent = True
    for name, result in results.items():
        other_top3 = set(np.argsort(-np.abs(result["phi"]))[:3])
        if len(current_top3 & other_top3) < 2:
            consistent = False
            break
    
    if consistent:
        print("✓ Top-3 features are CONSISTENT across configurations")
        print("  → Your methodology is robust!")
    else:
        print("⚠ Top-3 features DIFFER across configurations")
        print("  → Results depend on methodology choice. Discuss in thesis.")
