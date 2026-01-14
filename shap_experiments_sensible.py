import os
import json
import argparse
import math
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from abc import ABC, abstractmethod

import scipy.stats

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class SHAPConfig:
    name: str
    masking_strategy: str   # deletion | pad_replace | mean_replace
    baseline_type: str      # empty | training_mean | frequent_prefix
    explainer_type: str     # permutation | kernel | exact
    n_samples: int = 500
    description: str = ""


# =============================================================================
# Maskers
# =============================================================================

class Masker(ABC):
    @abstractmethod
    def apply(self, prefix_tokens: np.ndarray, mask: np.ndarray,
              max_length: int, pad_id: int) -> np.ndarray:
        pass


class DeletionMasker(Masker):
    def apply(self, prefix_tokens: np.ndarray, mask: np.ndarray,
              max_length: int, pad_id: int) -> np.ndarray:
        kept = prefix_tokens[mask.astype(bool)]
        out = np.full((max_length,), pad_id, dtype=np.int32)
        Lk = int(len(kept))
        if Lk > 0:
            out[-Lk:] = kept
        return out


class PadReplaceMasker(Masker):
    def apply(self, prefix_tokens: np.ndarray, mask: np.ndarray,
              max_length: int, pad_id: int) -> np.ndarray:
        L = int(len(prefix_tokens))
        out = np.full((max_length,), pad_id, dtype=np.int32)
        start = max_length - L
        out[start:] = prefix_tokens
        # replace masked positions with PAD
        for i in range(L):
            if mask[i] < 0.5:
                out[start + i] = pad_id
        return out


class MeanReplaceMasker(Masker):
    def __init__(self, replacement_token_id: int):
        self.replacement_token_id = int(replacement_token_id)

    def apply(self, prefix_tokens: np.ndarray, mask: np.ndarray,
              max_length: int, pad_id: int) -> np.ndarray:
        L = int(len(prefix_tokens))
        out = np.full((max_length,), pad_id, dtype=np.int32)
        start = max_length - L
        out[start:] = prefix_tokens
        for i in range(L):
            if mask[i] < 0.5:
                out[start + i] = self.replacement_token_id
        return out


# =============================================================================
# Baselines
# =============================================================================

class Baseline(ABC):
    @abstractmethod
    def base_value(
        self,
        model,
        target_class: int,
        score_fn: Callable[[np.ndarray, int], float],
        predict_batch_size: int
    ) -> float:
        pass


class EmptyBaseline(Baseline):
    def __init__(self, max_length: int, pad_id: int):
        self.X = np.full((1, max_length), pad_id, dtype=np.int32)

    def base_value(self, model, target_class: int,
                   score_fn: Callable[[np.ndarray, int], float],
                   predict_batch_size: int) -> float:
        out = model.predict(self.X, verbose=0, batch_size=predict_batch_size)[0]
        return float(score_fn(out, target_class))


class BackgroundMeanBaseline(Baseline):
    """
    Approximates "training_mean" using a background of real prefixes
    from batch_predictions.json, because encoded train_X is not always accessible.
    """
    def __init__(self, model, bg_X: np.ndarray, target_class: int,
                 score_fn: Callable[[np.ndarray, int], float],
                 predict_batch_size: int):
        outs = model.predict(bg_X, verbose=0, batch_size=predict_batch_size)
        vals = np.array([score_fn(outs[i], target_class) for i in range(outs.shape[0])], dtype=np.float64)
        self.mean_val = float(vals.mean())

    def base_value(self, model, target_class: int,
                   score_fn: Callable[[np.ndarray, int], float],
                   predict_batch_size: int) -> float:
        return float(self.mean_val)


class FrequentPrefixBaseline(Baseline):
    """
    Build a baseline prefix as the most frequent suffix-of-length-k among background prefixes.
    """
    def __init__(self, model, bg_prefix_tokens: List[np.ndarray], max_length: int, pad_id: int,
                 target_class: int, score_fn: Callable[[np.ndarray, int], float],
                 k: int, predict_batch_size: int):
        k = max(1, int(k))
        counts: Dict[tuple, int] = {}

        for seq in bg_prefix_tokens:
            if seq.size == 0:
                continue
            suf = seq[-k:] if seq.size >= k else seq
            key = tuple(suf.astype(int).tolist())
            counts[key] = counts.get(key, 0) + 1

        if not counts:
            # fallback: empty
            frequent = np.array([], dtype=np.int32)
        else:
            frequent = np.array(max(counts.items(), key=lambda kv: kv[1])[0], dtype=np.int32)

        X = np.full((1, max_length), pad_id, dtype=np.int32)
        if frequent.size > 0:
            X[0, -frequent.size:] = frequent

        out = model.predict(X, verbose=0, batch_size=predict_batch_size)[0]
        self.val = float(score_fn(out, target_class))

    def base_value(self, model, target_class: int,
                   score_fn: Callable[[np.ndarray, int], float],
                   predict_batch_size: int) -> float:
        return float(self.val)


# =============================================================================
# Explainers
# =============================================================================

class Explainer(ABC):
    @abstractmethod
    def explain(self, value_fn: Callable[[np.ndarray], np.ndarray], L: int, n: int,
                base_value: float) -> np.ndarray:
        pass


class PermutationMCExplainer(Explainer):
    """
    Efficient permutation Shapley:
    For each permutation, evaluate all cumulative masks in one batched model call.
    """
    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def explain(self, value_fn: Callable[[np.ndarray], np.ndarray], L: int, n: int,
                base_value: float) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        phi = np.zeros(L, dtype=np.float64)

        n = int(max(1, n))
        for _ in range(n):
            perm = rng.permutation(L)

            masks = np.zeros((L + 1, L), dtype=np.float32)  # m0..mL
            cur = np.zeros(L, dtype=np.float32)
            masks[0] = cur
            for t, j in enumerate(perm, start=1):
                cur = cur.copy()
                cur[j] = 1.0
                masks[t] = cur

            vals = value_fn(masks).astype(np.float64)  # [L+1]
            deltas = vals[1:] - vals[:-1]              # [L]
            for t, j in enumerate(perm):
                phi[j] += deltas[t]

        phi /= float(n)
        return phi


class KernelMaskExplainer(Explainer):
    """
    KernelSHAP-like regression on coalition masks.
    We regress y = f(mask) - base_value against X=mask, then solve for phi.
    """
    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def explain(self, value_fn: Callable[[np.ndarray], np.ndarray], L: int, n: int,
                base_value: float) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = int(max(50, n))

        # include empty and full
        masks = np.zeros((n + 2, L), dtype=np.float32)
        masks[0] = 0.0
        masks[1] = 1.0

        for i in range(2, n + 2):
            if L == 1:
                masks[i, 0] = 1.0
                continue
            size = int(rng.integers(1, L))
            idx = rng.choice(L, size=size, replace=False)
            masks[i, idx] = 1.0

        values = value_fn(masks).astype(np.float64)
        y = values - float(base_value)

        # Shapley kernel weights
        w = np.zeros(masks.shape[0], dtype=np.float64)
        for i in range(masks.shape[0]):
            s = int(masks[i].sum())
            if s == 0 or s == L:
                w[i] = 1e6
            else:
                w[i] = (L - 1) / (math.comb(L, s) * s * (L - s))

        X = masks.astype(np.float64)
        lam = 1e-6

        XtW = X.T * w
        A = XtW @ X + lam * np.eye(L)
        b = XtW @ y
        phi = np.linalg.solve(A, b)
        return phi


class ExactShapleyExplainer(Explainer):
    def explain(self, value_fn: Callable[[np.ndarray], np.ndarray], L: int, n: int,
                base_value: float) -> np.ndarray:
        if L > 12:
            raise ValueError(f"Exact Shapley too expensive for L={L} (>12)")

        from itertools import combinations
        fact = math.factorial

        # precompute all coalition values
        coalition_val: Dict[tuple, float] = {}
        masks = []
        keys = []
        for size in range(L + 1):
            for subset in combinations(range(L), size):
                mask = np.zeros(L, dtype=np.float32)
                if size > 0:
                    mask[list(subset)] = 1.0
                masks.append(mask)
                keys.append(subset)

        vals = value_fn(np.array(masks, dtype=np.float32)).astype(np.float64)
        for k, v in zip(keys, vals):
            coalition_val[k] = float(v)

        phi = np.zeros(L, dtype=np.float64)
        denom = fact(L)
        all_idx = set(range(L))

        for i in range(L):
            others = list(all_idx - {i})
            for s in range(L):
                for subset in combinations(others, s):
                    S = tuple(sorted(subset))
                    S_i = tuple(sorted(subset + (i,)))
                    weight = (fact(s) * fact(L - s - 1)) / denom
                    phi[i] += weight * (coalition_val[S_i] - coalition_val[S])

        return phi


# =============================================================================
# Attention alignment (optional)
# =============================================================================

def aggregate_attention_last_query_mean(attention: np.ndarray, L: int) -> np.ndarray:
    # attention can be [H,T,T] or [B,H,T,T], average blocks if needed
    attn = attention
    if attn.ndim == 4:
        attn = attn.mean(axis=0)

    H, T, _ = attn.shape
    start = T - L
    last = T - 1
    vec = attn[:, last, start:].mean(axis=0)
    s = float(vec.sum())
    if s > 0:
        vec = vec / s
    return vec.astype(np.float64)


def alignment_metrics(attn: np.ndarray, phi: np.ndarray, k: int = 5) -> Dict[str, float]:
    abs_phi = np.abs(phi).astype(np.float64)
    if abs_phi.sum() > 0:
        abs_phi = abs_phi / abs_phi.sum()

    pearson = float(np.corrcoef(attn, abs_phi)[0, 1])
    spearman = float(scipy.stats.spearmanr(attn, abs_phi).correlation)

    kk = int(min(k, len(phi)))
    top_attn = set(np.argsort(-attn)[:kk].tolist())
    top_phi = set(np.argsort(-np.abs(phi))[:kk].tolist())
    overlap = float(len(top_attn & top_phi) / max(1, kk))

    return {"pearson": pearson, "spearman": spearman, "top_k_overlap": overlap}


# =============================================================================
# Build background from batch_predictions.json
# =============================================================================

def load_background_from_predictions(
    predictions: List[dict],
    x_word_dict: dict,
    max_length: int,
    pad_id: int,
    max_bg: int = 512
) -> Dict[str, Any]:
    """
    Returns:
      bg_prefix_tokens: list of [Li] arrays (unpadded)
      bg_X: [N, T] padded token ids
      mean_token_id: most frequent non-pad token in background
    """
    bg_prefix_tokens: List[np.ndarray] = []
    bg_X_list: List[np.ndarray] = []

    for row in predictions[:max_bg]:
        acts = row.get("prefix_activities", [])
        toks = np.array([x_word_dict[a] for a in acts], dtype=np.int32)
        bg_prefix_tokens.append(toks)

        X = np.full((max_length,), pad_id, dtype=np.int32)
        L = int(toks.size)
        if L > 0:
            X[-L:] = toks
        bg_X_list.append(X)

    bg_X = np.stack(bg_X_list, axis=0) if bg_X_list else np.full((1, max_length), pad_id, dtype=np.int32)

    flat = bg_X.reshape(-1)
    flat = flat[flat != pad_id]
    if flat.size == 0:
        mean_token_id = int(pad_id)
    else:
        mean_token_id = int(np.bincount(flat.astype(np.int64)).argmax())

    return {"bg_prefix_tokens": bg_prefix_tokens, "bg_X": bg_X, "mean_token_id": mean_token_id}


# =============================================================================
# Sensible config groups
# =============================================================================

def build_sensible_configs(n_samples: int) -> Dict[str, List[SHAPConfig]]:
    """
    Returns group_name -> list of configs.
    """
    # Aliases for readability
    def C(name, masking, baseline, explainer, desc):
        return SHAPConfig(
            name=name,
            masking_strategy=masking,
            baseline_type=baseline,
            explainer_type=explainer,
            n_samples=n_samples,
            description=desc,
        )

    # Core set: what you will most likely keep
    core = [
        C("current", "deletion", "empty", "permutation",
          "Baseline reference: deletion + empty + permutation"),
        C("pad_perm_empty", "pad_replace", "empty", "permutation",
          "Position-preserving masking under empty baseline"),
        C("pad_perm_mean", "pad_replace", "training_mean", "permutation",
          "Recommended: pad_replace + background-mean baseline + permutation"),
        C("mean_perm_mean", "mean_replace", "training_mean", "permutation",
          "Mean-replace + background-mean baseline, permutation estimator"),
    ]

    # Focused comparisons
    masking = [
        C("del_perm_empty", "deletion", "empty", "permutation", "Masking compare (deletion)"),
        C("pad_perm_empty", "pad_replace", "empty", "permutation", "Masking compare (pad_replace)"),
        C("mean_perm_empty", "mean_replace", "empty", "permutation", "Masking compare (mean_replace)"),
    ]

    baseline = [
        C("del_empty_perm", "deletion", "empty", "permutation", "Baseline compare (empty)"),
        C("del_mean_perm", "deletion", "training_mean", "permutation", "Baseline compare (background mean)"),
        C("del_freq_perm", "deletion", "frequent_prefix", "permutation", "Baseline compare (frequent prefix)"),
    ]

    # Kernel: only with fixed-position masking
    kernel = [
        C("pad_mean_kernel", "pad_replace", "training_mean", "kernel",
          "Kernel under pad_replace + background mean baseline"),
        C("mean_mean_kernel", "mean_replace", "training_mean", "kernel",
          "Kernel under mean_replace + background mean baseline"),
    ]

    sanity_exact = [
        C("pad_empty_exact", "pad_replace", "empty", "exact",
          "Exact Shapley sanity check (only if L<=12)"),
    ]

    # Union
    def uniq(cfgs: List[SHAPConfig]) -> List[SHAPConfig]:
        seen = set()
        out = []
        for c in cfgs:
            if c.name not in seen:
                out.append(c)
                seen.add(c.name)
        return out

    all_sensible = uniq(core + masking + baseline + kernel + sanity_exact)

    return {
        "core": core,
        "masking": masking,
        "baseline": baseline,
        "kernel": kernel,
        "sanity_exact": sanity_exact,
        "all": all_sensible,
    }


# =============================================================================
# Main run
# =============================================================================

def run_configs(
    model,
    prefix_tokens: np.ndarray,
    max_length: int,
    pad_id: int,
    target_class: int,
    score_fn: Callable[[np.ndarray, int], float],
    configs: List[SHAPConfig],
    background: Dict[str, Any],
    attention: Optional[np.ndarray],
    frequent_prefix_k: int,
    predict_batch_size: int,
    seed: int = 42
) -> Dict[str, Any]:
    prefix_tokens = np.asarray(prefix_tokens, dtype=np.int32)
    L = int(prefix_tokens.size)

    # full value
    X_full = np.full((1, max_length), pad_id, dtype=np.int32)
    if L > 0:
        X_full[0, -L:] = prefix_tokens
    out_full = model.predict(X_full, verbose=0, batch_size=predict_batch_size)[0]
    full_value = float(score_fn(out_full, target_class))

    bg_X = background["bg_X"]
    bg_prefix_tokens = background["bg_prefix_tokens"]
    mean_token_id = background["mean_token_id"]

    results: Dict[str, Any] = {
        "L": L,
        "target_class": int(target_class),
        "full_value": full_value,
        "configs": {}
    }

    def make_value_fn(masker: Masker) -> Callable[[np.ndarray], np.ndarray]:
        def value_fn(masks: np.ndarray) -> np.ndarray:
            masks = np.asarray(masks, dtype=np.float32)
            N = masks.shape[0]
            X = np.zeros((N, max_length), dtype=np.int32)
            for i in range(N):
                X[i] = masker.apply(prefix_tokens, masks[i], max_length, pad_id)
            outs = model.predict(X, verbose=0, batch_size=predict_batch_size)
            vals = np.array([score_fn(outs[i], target_class) for i in range(N)], dtype=np.float64)
            return vals
        return value_fn

    for cfg in configs:
        try:
            # masker
            if cfg.masking_strategy == "deletion":
                masker = DeletionMasker()
            elif cfg.masking_strategy == "pad_replace":
                masker = PadReplaceMasker()
            elif cfg.masking_strategy == "mean_replace":
                masker = MeanReplaceMasker(mean_token_id)
            else:
                raise ValueError(f"Unknown masking_strategy: {cfg.masking_strategy}")

            # baseline
            if cfg.baseline_type == "empty":
                baseline = EmptyBaseline(max_length=max_length, pad_id=pad_id)
            elif cfg.baseline_type == "training_mean":
                baseline = BackgroundMeanBaseline(
                    model=model,
                    bg_X=bg_X,
                    target_class=target_class,
                    score_fn=score_fn,
                    predict_batch_size=predict_batch_size
                )
            elif cfg.baseline_type == "frequent_prefix":
                baseline = FrequentPrefixBaseline(
                    model=model,
                    bg_prefix_tokens=bg_prefix_tokens,
                    max_length=max_length,
                    pad_id=pad_id,
                    target_class=target_class,
                    score_fn=score_fn,
                    k=frequent_prefix_k,
                    predict_batch_size=predict_batch_size
                )
            else:
                raise ValueError(f"Unknown baseline_type: {cfg.baseline_type}")

            base_val = float(baseline.base_value(model, target_class, score_fn, predict_batch_size))

            # explainer
            if cfg.explainer_type == "permutation":
                explainer = PermutationMCExplainer(seed=seed)
            elif cfg.explainer_type == "kernel":
                # Only sensible with fixed-position masking, but we assume configs already pruned
                explainer = KernelMaskExplainer(seed=seed)
            elif cfg.explainer_type == "exact":
                explainer = ExactShapleyExplainer()
            else:
                raise ValueError(f"Unknown explainer_type: {cfg.explainer_type}")

            # skip exact if too large
            if cfg.explainer_type == "exact" and L > 12:
                raise ValueError(f"Exact skipped for L={L} (>12)")

            value_fn = make_value_fn(masker)
            phi = explainer.explain(value_fn, L=L, n=cfg.n_samples, base_value=base_val)
            phi = np.asarray(phi, dtype=np.float64)

            reconstructed = float(base_val + float(phi.sum()))
            add_err = float(abs(reconstructed - full_value))

            out_cfg: Dict[str, Any] = {
                "description": cfg.description,
                "settings": {
                    "masking": cfg.masking_strategy,
                    "baseline": cfg.baseline_type,
                    "explainer": cfg.explainer_type,
                    "n_samples": int(cfg.n_samples),
                },
                "base_value": base_val,
                "phi": phi.tolist(),
                "reconstructed": reconstructed,
                "additivity_error": add_err,
            }

            if attention is not None:
                attn_vec = aggregate_attention_last_query_mean(attention, L=L)
                out_cfg["alignment"] = alignment_metrics(attn_vec, phi, k=5)

            results["configs"][cfg.name] = out_cfg

            print(f"  {cfg.name}: add_err={add_err:.2e}")

        except Exception as e:
            results["configs"][cfg.name] = {
                "error": str(e),
                "settings": {
                    "masking": cfg.masking_strategy,
                    "baseline": cfg.baseline_type,
                    "explainer": cfg.explainer_type,
                    "n_samples": int(cfg.n_samples),
                }
            }
            print(f"  {cfg.name}: ERROR {e}")

    return results


def print_summary(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("SENSIBLE SHAP CONFIG SUMMARY")
    print("=" * 90)
    print(f"L={results.get('L')} target_class={results.get('target_class')} full_value={results.get('full_value'):.4f}")
    print("")
    print(f"{'config':<22} {'add_err':>12} {'pearson':>9} {'spearman':>10} {'top5':>8}")
    print("-" * 90)

    for name, r in results.get("configs", {}).items():
        if "error" in r:
            print(f"{name:<22} ERROR: {r['error']}")
            continue
        add_err = r.get("additivity_error", float("nan"))
        align = r.get("alignment", {})
        pearson = align.get("pearson", float("nan"))
        spearman = align.get("spearman", float("nan"))
        top5 = align.get("top_k_overlap", float("nan"))
        print(f"{name:<22} {add_err:>12.2e} {pearson:>9.3f} {spearman:>10.3f} {top5:>8.2f}")


# =============================================================================
# CLI (mirrors your previous shap_experiments style)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensible SHAP Configuration Experiments (PPM)")
    parser.add_argument("--dataset", default="BPIC2012-W", help="Dataset name")
    parser.add_argument("--prefix_index", type=int, default=12, help="Prefix index")
    parser.add_argument("--experiment", default="all",
                        help="Group to run: core, masking, baseline, kernel, sanity_exact, all")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--n_samples", type=int, default=500, help="Samples per config (perm count or kernel samples)")
    parser.add_argument("--max_bg", type=int, default=512, help="Number of background prefixes from batch_predictions.json")
    parser.add_argument("--frequent_k", type=int, default=3, help="Suffix length for frequent_prefix baseline")
    parser.add_argument("--predict_batch_size", type=int, default=256, help="batch_size for model.predict")

    args = parser.parse_args()

    # Import your utilities
    from processtransformer import constants
    from processtransformer.data import loader
    from processtransformer.models import transformer

    print(f"\nLoading data for {args.dataset}...")

    dl = loader.LogsDataLoader(name=args.dataset)
    train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = \
        dl.load_data(constants.Task.NEXT_ACTIVITY)

    pad_token = getattr(constants, "PAD_TOKEN", "[PAD]")
    pad_id = int(x_word_dict.get(pad_token, x_word_dict.get("<pad>", 0)))

    model = transformer.get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size,
        output_dim=num_output
    )

    ckpt_path = f"./models/{args.dataset}/next_activity_ckpt"
    model.load_weights(ckpt_path).expect_partial()
    print(f"Loaded model from {ckpt_path}")

    outputs_dir = Path(f"./outputs/{args.dataset}")

    with open(outputs_dir / "batch_predictions.json", encoding="utf-8") as f:
        predictions = json.load(f)["predictions"]

    prefix_info = predictions[args.prefix_index]
    prefix_activities = prefix_info["prefix_activities"]
    prefix_tokens = np.array([x_word_dict[a] for a in prefix_activities], dtype=np.int32)
    print(f"Prefix ({len(prefix_activities)} events): {' -> '.join(prefix_activities[:5])}...")

    attention_path = outputs_dir / "block_mha_scores.npy"
    attention = np.load(attention_path)[args.prefix_index] if attention_path.exists() else None

    # Background from predictions
    background = load_background_from_predictions(
        predictions=predictions,
        x_word_dict=x_word_dict,
        max_length=max_case_length,
        pad_id=pad_id,
        max_bg=args.max_bg
    )

    # Score function: logit for target class (keeps additivity meaningful)
    def score_fn(model_out: np.ndarray, cls: int) -> float:
        return float(model_out[int(cls)])

    # target class taken from full prediction
    X_full = np.full((1, max_case_length), pad_id, dtype=np.int32)
    X_full[0, -len(prefix_tokens):] = prefix_tokens
    full_out = model.predict(X_full, verbose=0, batch_size=args.predict_batch_size)[0]
    target_class = int(np.argmax(full_out))

    groups = build_sensible_configs(n_samples=args.n_samples)
    if args.experiment not in groups:
        raise ValueError(f"Unknown experiment group: {args.experiment}. Choose from: {list(groups.keys())}")

    configs_to_run = groups[args.experiment]

    print(f"\nRunning experiment group: {args.experiment}")
    for c in configs_to_run:
        print(f"  - {c.name}: {c.description}")

    results = run_configs(
        model=model,
        prefix_tokens=prefix_tokens,
        max_length=max_case_length,
        pad_id=pad_id,
        target_class=target_class,
        score_fn=score_fn,
        configs=configs_to_run,
        background=background,
        attention=attention,
        frequent_prefix_k=args.frequent_k,
        predict_batch_size=args.predict_batch_size,
        seed=42
    )

    print_summary(results)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = outputs_dir / f"shap_experiments_sensible_{args.experiment}_batch_{args.prefix_index}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
