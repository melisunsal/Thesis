"""
SHAP Configuration Experiments for Process Mining
=================================================

This module provides alternative SHAP configurations to compare against
your current implementation. Each configuration changes one design choice.

Experiments:
1. Masking Strategy: Deletion vs PAD-replacement vs Mean-replacement
2. Baseline Choice: Empty vs Training-mean vs Most-frequent-prefix
3. Output Type: Logit vs Probability vs Log-odds
4. Explainer Type: Permutation vs Kernel vs Exact (small L)
5. Attention Aggregation: Mean vs Max vs Weighted

Usage:
    python shap_experiments.py --dataset BPIC2012-W --prefix_index 12 --experiment all
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.stats

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class SHAPConfig:
    """Configuration for SHAP computation."""
    name: str
    masking_strategy: str  # "deletion", "pad_replace", "mean_replace"
    baseline_type: str     # "empty", "training_mean", "frequent_prefix"
    output_type: str       # "logit", "probability", "log_odds"
    explainer_type: str    # "permutation", "kernel", "exact"
    n_permutations: int = 1000
    description: str = ""


# =============================================================================
# MASKING STRATEGIES
# =============================================================================

class MaskingStrategy(ABC):
    """Abstract base class for masking strategies."""
    
    @abstractmethod
    def apply_mask(self, tokens: np.ndarray, mask: np.ndarray, 
                   max_length: int, pad_id: int) -> np.ndarray:
        """Apply mask to tokens and return padded sequence."""
        pass


class DeletionMasking(MaskingStrategy):
    """
    Deletion masking: Remove masked tokens, collapse sequence, re-pad.
    
    Original: [A, B, C, D, E]
    Mask:     [1, 0, 1, 0, 1]
    Result:   [PAD, PAD, A, C, E]  (right-aligned)
    
    This is your current implementation.
    """
    
    def apply_mask(self, tokens: np.ndarray, mask: np.ndarray,
                   max_length: int, pad_id: int) -> np.ndarray:
        kept_tokens = tokens[mask.astype(bool)]
        result = np.full(max_length, pad_id, dtype=np.int32)
        L = len(kept_tokens)
        if L > 0:
            result[-L:] = kept_tokens
        return result


class PADReplaceMasking(MaskingStrategy):
    """
    PAD replacement: Replace masked tokens with PAD in-place.
    
    Original: [PAD, PAD, A, B, C, D, E]
    Mask:     [-, -, 1, 0, 1, 0, 1]
    Result:   [PAD, PAD, A, PAD, C, PAD, E]
    
    Preserves positional information but introduces PAD in middle.
    Similar to "Attention Please" Experiment 2.
    """
    
    def apply_mask(self, tokens: np.ndarray, mask: np.ndarray,
                   max_length: int, pad_id: int) -> np.ndarray:
        # First, create full padded sequence
        result = np.full(max_length, pad_id, dtype=np.int32)
        L = len(tokens)
        start_pos = max_length - L
        result[start_pos:] = tokens
        
        # Then replace masked positions with PAD
        for i, m in enumerate(mask):
            if m < 0.5:  # masked out
                result[start_pos + i] = pad_id
        
        return result


class MeanReplaceMasking(MaskingStrategy):
    """
    Mean replacement: Replace masked tokens with most frequent token.
    
    Original: [PAD, PAD, A, B, C, D, E]
    Mask:     [-, -, 1, 0, 1, 0, 1]
    Result:   [PAD, PAD, A, <MEAN>, C, <MEAN>, E]
    
    Where <MEAN> is the most frequent activity in training data.
    """
    
    def __init__(self, mean_token_id: int):
        self.mean_token_id = mean_token_id
    
    def apply_mask(self, tokens: np.ndarray, mask: np.ndarray,
                   max_length: int, pad_id: int) -> np.ndarray:
        result = np.full(max_length, pad_id, dtype=np.int32)
        L = len(tokens)
        start_pos = max_length - L
        result[start_pos:] = tokens
        
        for i, m in enumerate(mask):
            if m < 0.5:
                result[start_pos + i] = self.mean_token_id
        
        return result


# =============================================================================
# BASELINE STRATEGIES
# =============================================================================

class BaselineStrategy(ABC):
    """Abstract base class for baseline computation."""
    
    @abstractmethod
    def get_baseline(self, model, max_length: int, pad_id: int,
                     target_class: int) -> float:
        """Get baseline value for SHAP."""
        pass
    
    @abstractmethod
    def get_baseline_input(self, max_length: int, pad_id: int) -> np.ndarray:
        """Get baseline input sequence."""
        pass


class EmptyBaseline(BaselineStrategy):
    """
    Empty sequence baseline: All PAD tokens.
    
    Baseline = f([PAD, PAD, ..., PAD])
    
    This is your current implementation.
    """
    
    def get_baseline_input(self, max_length: int, pad_id: int) -> np.ndarray:
        return np.full((1, max_length), pad_id, dtype=np.int32)
    
    def get_baseline(self, model, max_length: int, pad_id: int,
                     target_class: int) -> float:
        X = self.get_baseline_input(max_length, pad_id)
        logits = model.predict(X, verbose=0)[0]
        return logits[target_class]


class TrainingMeanBaseline(BaselineStrategy):
    """
    Training mean baseline: Average prediction over training samples.
    
    Baseline = E[f(x)] over training data
    
    More stable but requires training data access.
    """
    
    def __init__(self, training_logits_mean: np.ndarray):
        """
        training_logits_mean: [num_classes] mean logits over training set
        """
        self.training_logits_mean = training_logits_mean
    
    def get_baseline_input(self, max_length: int, pad_id: int) -> np.ndarray:
        # Not used directly, but return empty for compatibility
        return np.full((1, max_length), pad_id, dtype=np.int32)
    
    def get_baseline(self, model, max_length: int, pad_id: int,
                     target_class: int) -> float:
        return self.training_logits_mean[target_class]


class FrequentPrefixBaseline(BaselineStrategy):
    """
    Frequent prefix baseline: Most common starting pattern.
    
    Baseline = f(most_frequent_prefix)
    
    Provides "typical case" reference.
    """
    
    def __init__(self, frequent_prefix: np.ndarray):
        """
        frequent_prefix: [L] most frequent prefix tokens
        """
        self.frequent_prefix = frequent_prefix
    
    def get_baseline_input(self, max_length: int, pad_id: int) -> np.ndarray:
        result = np.full((1, max_length), pad_id, dtype=np.int32)
        L = len(self.frequent_prefix)
        result[0, -L:] = self.frequent_prefix
        return result
    
    def get_baseline(self, model, max_length: int, pad_id: int,
                     target_class: int) -> float:
        X = self.get_baseline_input(max_length, pad_id)
        logits = model.predict(X, verbose=0)[0]
        return logits[target_class]


# =============================================================================
# OUTPUT TRANSFORMATIONS
# =============================================================================

class OutputTransform(ABC):
    """Abstract base class for output transformation."""
    
    @abstractmethod
    def transform(self, logits: np.ndarray, target_class: int) -> float:
        """Transform model logits to scalar output."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass


class LogitOutput(OutputTransform):
    """
    Logit output: Raw logit for target class.
    
    output = logits[target_class]
    
    This is your current implementation. Best for additivity.
    """
    
    def transform(self, logits: np.ndarray, target_class: int) -> float:
        return logits[target_class]
    
    def name(self) -> str:
        return "logit"


class ProbabilityOutput(OutputTransform):
    """
    Probability output: Softmax probability for target class.
    
    output = softmax(logits)[target_class]
    
    More interpretable but non-additive.
    """
    
    def transform(self, logits: np.ndarray, target_class: int) -> float:
        # Stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return probs[target_class]
    
    def name(self) -> str:
        return "probability"


class LogOddsOutput(OutputTransform):
    """
    Log-odds output: Log ratio of target vs second-best class.
    
    output = log(p_target / p_second)
    
    Contrastive explanation: "Why A instead of B?"
    """
    
    def transform(self, logits: np.ndarray, target_class: int) -> float:
        # Stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        
        # Find second-best class
        sorted_indices = np.argsort(-probs)
        if sorted_indices[0] == target_class:
            second_class = sorted_indices[1]
        else:
            second_class = sorted_indices[0]
        
        # Log odds (with small epsilon for stability)
        eps = 1e-10
        return np.log(probs[target_class] + eps) - np.log(probs[second_class] + eps)
    
    def name(self) -> str:
        return "log_odds"


# =============================================================================
# SHAP EXPLAINERS
# =============================================================================

class SHAPExplainer(ABC):
    """Abstract base class for SHAP explainers."""
    
    @abstractmethod
    def explain(self, value_function: Callable, L: int, 
                n_permutations: int) -> np.ndarray:
        """Compute SHAP values."""
        pass


class PermutationExplainer(SHAPExplainer):
    """
    Permutation-based SHAP: Sample random permutations.
    
    For each permutation π:
        For each position i:
            φ[i] += f(S ∪ {i}) - f(S)
    
    This is your current implementation.
    """
    
    def explain(self, value_function: Callable, L: int,
                n_permutations: int) -> np.ndarray:
        phi = np.zeros(L)
        
        for _ in range(n_permutations):
            perm = np.random.permutation(L)
            mask = np.zeros(L, dtype=np.float32)
            prev_val = value_function(mask.reshape(1, -1))[0]
            
            for idx in perm:
                mask[idx] = 1.0
                curr_val = value_function(mask.reshape(1, -1))[0]
                phi[idx] += curr_val - prev_val
                prev_val = curr_val
        
        return phi / n_permutations


class ExactExplainer(SHAPExplainer):
    """
    Exact Shapley values: Enumerate all 2^L coalitions.
    
    Only feasible for L ≤ 12 or so.
    
    φ[i] = Σ_{S ⊆ N\{i}} [|S|!(L-|S|-1)!/L!] * [f(S∪{i}) - f(S)]
    """
    
    def explain(self, value_function: Callable, L: int,
                n_permutations: int = None) -> np.ndarray:
        from itertools import combinations
        from math import factorial
        
        if L > 15:
            raise ValueError(f"Exact Shapley infeasible for L={L} (max ~12-15)")
        
        phi = np.zeros(L)
        all_indices = set(range(L))
        
        # Precompute all coalition values
        coalition_values = {}
        for size in range(L + 1):
            for subset in combinations(range(L), size):
                mask = np.zeros(L, dtype=np.float32)
                mask[list(subset)] = 1.0
                coalition_values[subset] = value_function(mask.reshape(1, -1))[0]
        
        # Compute Shapley values
        for i in range(L):
            others = all_indices - {i}
            
            for size in range(L):
                for subset in combinations(others, size):
                    S = set(subset)
                    S_tuple = tuple(sorted(S))
                    S_with_i = tuple(sorted(S | {i}))
                    
                    # Shapley weight
                    weight = factorial(size) * factorial(L - size - 1) / factorial(L)
                    
                    # Marginal contribution
                    marginal = coalition_values[S_with_i] - coalition_values[S_tuple]
                    
                    phi[i] += weight * marginal
        
        return phi


class KernelExplainer(SHAPExplainer):
    """
    KernelSHAP: Weighted linear regression on coalition samples.
    
    Sample coalitions, weight by Shapley kernel, fit linear model.
    
    Note: Assumes feature independence (may be problematic for sequences).
    """
    
    def explain(self, value_function: Callable, L: int,
                n_permutations: int) -> np.ndarray:
        # Number of samples (more than permutation for stability)
        n_samples = n_permutations * 2
        
        # Sample coalitions
        masks = []
        weights = []
        values = []
        
        # Always include empty and full
        masks.append(np.zeros(L))
        masks.append(np.ones(L))
        
        # Sample random coalitions
        for _ in range(n_samples - 2):
            # Random coalition size
            size = np.random.randint(1, L)
            mask = np.zeros(L)
            mask[np.random.choice(L, size, replace=False)] = 1.0
            masks.append(mask)
        
        masks = np.array(masks)
        
        # Compute values
        values = value_function(masks)
        
        # Compute Shapley kernel weights
        for mask in masks:
            s = mask.sum()
            if s == 0 or s == L:
                w = 1e6  # High weight for empty/full
            else:
                # Shapley kernel: (L-1) / (C(L,s) * s * (L-s))
                from math import comb
                w = (L - 1) / (comb(L, int(s)) * s * (L - s))
            weights.append(w)
        
        weights = np.array(weights)
        
        # Weighted least squares
        # y = Xβ where β[i] = φ[i]
        # Solve: (X'WX)β = X'Wy
        
        X = masks
        y = values
        W = np.diag(weights)
        
        # Add regularization for stability
        XtWX = X.T @ W @ X + 1e-6 * np.eye(L)
        XtWy = X.T @ W @ y
        
        phi = np.linalg.solve(XtWX, XtWy)
        
        return phi


# =============================================================================
# ATTENTION AGGREGATION STRATEGIES
# =============================================================================

class AttentionAggregation(ABC):
    """Abstract base class for attention aggregation."""
    
    @abstractmethod
    def aggregate(self, attention: np.ndarray) -> np.ndarray:
        """
        Aggregate attention across heads.
        
        attention: [H, T, T] attention scores
        Returns: [T] aggregated attention for last query position
        """
        pass


class MeanAttention(AttentionAggregation):
    """
    Mean aggregation: Average across heads.
    
    This is your current implementation.
    """
    
    def aggregate(self, attention: np.ndarray, L: int) -> np.ndarray:
        H, T, _ = attention.shape
        start_pos = T - L
        last_pos = T - 1
        
        # Get last row for each head, then average
        attn_last = attention[:, last_pos, start_pos:].mean(axis=0)
        return attn_last


class MaxAttention(AttentionAggregation):
    """
    Max aggregation: Maximum across heads.
    
    Captures the "most attended" signal from any head.
    """
    
    def aggregate(self, attention: np.ndarray, L: int) -> np.ndarray:
        H, T, _ = attention.shape
        start_pos = T - L
        last_pos = T - 1
        
        attn_last = attention[:, last_pos, start_pos:].max(axis=0)
        return attn_last


class WeightedAttention(AttentionAggregation):
    """
    Weighted aggregation: Weight heads by their "confidence" (entropy).
    
    Heads with lower entropy (more focused) get higher weight.
    """
    
    def aggregate(self, attention: np.ndarray, L: int) -> np.ndarray:
        H, T, _ = attention.shape
        start_pos = T - L
        last_pos = T - 1
        
        head_attentions = attention[:, last_pos, start_pos:]  # [H, L]
        
        # Compute entropy for each head
        eps = 1e-10
        entropies = -np.sum(head_attentions * np.log(head_attentions + eps), axis=1)
        
        # Weight = inverse entropy (more focused = higher weight)
        weights = 1.0 / (entropies + eps)
        weights = weights / weights.sum()
        
        # Weighted average
        attn_last = np.sum(head_attentions * weights[:, np.newaxis], axis=0)
        return attn_last


class PerHeadAttention(AttentionAggregation):
    """
    Per-head analysis: Return attention for each head separately.
    
    Useful for understanding head specialization.
    """
    
    def aggregate(self, attention: np.ndarray, L: int) -> Dict[str, np.ndarray]:
        H, T, _ = attention.shape
        start_pos = T - L
        last_pos = T - 1
        
        result = {}
        for h in range(H):
            result[f"head_{h}"] = attention[h, last_pos, start_pos:]
        
        result["mean"] = attention[:, last_pos, start_pos:].mean(axis=0)
        return result


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

class SHAPExperiment:
    """Run SHAP experiments with different configurations."""
    
    def __init__(self, model, max_length: int, pad_id: int, 
                 vocab: dict, y_vocab: dict):
        self.model = model
        self.max_length = max_length
        self.pad_id = pad_id
        self.vocab = vocab
        self.y_vocab = y_vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.inv_y_vocab = {v: k for k, v in y_vocab.items()}
    
    def run_experiment(self, config: SHAPConfig, prefix_tokens: np.ndarray,
                       target_class: int = None,
                       attention: np.ndarray = None,
                       training_data: np.ndarray = None) -> Dict:
        """
        Run SHAP computation with given configuration.
        """
        L = len(prefix_tokens)
        
        # Setup masking strategy
        if config.masking_strategy == "deletion":
            masker = DeletionMasking()
        elif config.masking_strategy == "pad_replace":
            masker = PADReplaceMasking()
        elif config.masking_strategy == "mean_replace":
            # Find most frequent token
            if training_data is not None:
                flat = training_data[training_data != self.pad_id]
                mean_token = int(np.bincount(flat).argmax())
            else:
                mean_token = 1  # Default to first non-PAD token
            masker = MeanReplaceMasking(mean_token)
        else:
            raise ValueError(f"Unknown masking: {config.masking_strategy}")
        
        # Setup baseline
        if config.baseline_type == "empty":
            baseline = EmptyBaseline()
        elif config.baseline_type == "training_mean":
            if training_data is not None:
                # Compute mean logits over training
                logits = self.model.predict(training_data[:1000], verbose=0)
                mean_logits = logits.mean(axis=0)
            else:
                mean_logits = np.zeros(len(self.y_vocab))
            baseline = TrainingMeanBaseline(mean_logits)
        elif config.baseline_type == "frequent_prefix":
            # Use first token repeated as simple "frequent" prefix
            frequent = np.array([prefix_tokens[0]])
            baseline = FrequentPrefixBaseline(frequent)
        else:
            raise ValueError(f"Unknown baseline: {config.baseline_type}")
        
        # Setup output transform
        if config.output_type == "logit":
            output_transform = LogitOutput()
        elif config.output_type == "probability":
            output_transform = ProbabilityOutput()
        elif config.output_type == "log_odds":
            output_transform = LogOddsOutput()
        else:
            raise ValueError(f"Unknown output: {config.output_type}")
        
        # Get full prediction
        X_full = np.full(self.max_length, self.pad_id, dtype=np.int32)
        X_full[-L:] = prefix_tokens
        logits_full = self.model.predict(X_full.reshape(1, -1), verbose=0)[0]
        
        if target_class is None:
            target_class = int(np.argmax(logits_full))
        
        f_full = output_transform.transform(logits_full, target_class)
        
        # Get baseline value
        base_value = baseline.get_baseline(
            self.model, self.max_length, self.pad_id, target_class
        )
        # Transform baseline too if using probability/log_odds
        if config.output_type != "logit":
            X_base = baseline.get_baseline_input(self.max_length, self.pad_id)
            logits_base = self.model.predict(X_base, verbose=0)[0]
            base_value = output_transform.transform(logits_base, target_class)
        
        # Create value function
        def value_function(masks: np.ndarray) -> np.ndarray:
            N = masks.shape[0]
            X_batch = np.zeros((N, self.max_length), dtype=np.int32)
            
            for i in range(N):
                X_batch[i] = masker.apply_mask(
                    prefix_tokens, masks[i], self.max_length, self.pad_id
                )
            
            logits = self.model.predict(X_batch, verbose=0)
            outputs = np.array([
                output_transform.transform(logits[i], target_class) 
                for i in range(N)
            ])
            return outputs
        
        # Setup explainer
        if config.explainer_type == "permutation":
            explainer = PermutationExplainer()
        elif config.explainer_type == "exact":
            explainer = ExactExplainer()
        elif config.explainer_type == "kernel":
            explainer = KernelExplainer()
        else:
            raise ValueError(f"Unknown explainer: {config.explainer_type}")
        
        # Compute SHAP values
        np.random.seed(42)  # Reproducibility
        phi = explainer.explain(value_function, L, config.n_permutations)
        
        # Verify additivity
        reconstructed = base_value + phi.sum()
        additivity_error = abs(reconstructed - f_full)
        
        # Compute attention alignment if available
        alignment = {}
        if attention is not None:
            attn_agg = MeanAttention()
            attn = attn_agg.aggregate(attention, L)
            
            # Normalize attention
            attn = attn / (attn.sum() + 1e-10)
            
            # Correlations
            abs_phi = np.abs(phi)
            alignment["pearson"] = float(np.corrcoef(attn, abs_phi)[0, 1])
            alignment["spearman"] = float(scipy.stats.spearmanr(attn, abs_phi).correlation)
            
            # Top-k overlap
            k = min(5, L)
            top_attn = set(np.argsort(-attn)[:k])
            top_phi = set(np.argsort(-abs_phi)[:k])
            alignment["top_k_overlap"] = len(top_attn & top_phi) / k
        
        return {
            "config": config.name,
            "description": config.description,
            "phi": phi.tolist(),
            "base_value": float(base_value),
            "f_full": float(f_full),
            "reconstructed": float(reconstructed),
            "additivity_error": float(additivity_error),
            "target_class": target_class,
            "predicted_label": self.inv_y_vocab.get(target_class, str(target_class)),
            "feature_names": [
                f"E{i+1}:{self.inv_vocab.get(t, t)}" 
                for i, t in enumerate(prefix_tokens)
            ],
            "alignment": alignment,
            "settings": {
                "masking": config.masking_strategy,
                "baseline": config.baseline_type,
                "output": config.output_type,
                "explainer": config.explainer_type,
                "n_permutations": config.n_permutations
            }
        }


# =============================================================================
# PREDEFINED CONFIGURATIONS
# =============================================================================

CONFIGS = {
    # Your current implementation (baseline for comparison)
    "current": SHAPConfig(
        name="current",
        masking_strategy="deletion",
        baseline_type="empty",
        output_type="logit",
        explainer_type="permutation",
        n_permutations=1000,
        description="Current implementation: deletion + empty + logit + permutation"
    ),
    
    # Alternative masking strategies
    "pad_replace": SHAPConfig(
        name="pad_replace",
        masking_strategy="pad_replace",
        baseline_type="empty",
        output_type="logit",
        explainer_type="permutation",
        n_permutations=1000,
        description="PAD replacement (Attention Please style): preserves positions"
    ),
    
    "mean_replace": SHAPConfig(
        name="mean_replace",
        masking_strategy="mean_replace",
        baseline_type="empty",
        output_type="logit",
        explainer_type="permutation",
        n_permutations=1000,
        description="Mean token replacement: replace with most frequent activity"
    ),
    
    # Alternative baselines
    "training_mean": SHAPConfig(
        name="training_mean",
        masking_strategy="deletion",
        baseline_type="training_mean",
        output_type="logit",
        explainer_type="permutation",
        n_permutations=1000,
        description="Training mean baseline: average prediction over training set"
    ),
    
    # Alternative outputs
    "probability": SHAPConfig(
        name="probability",
        masking_strategy="deletion",
        baseline_type="empty",
        output_type="probability",
        explainer_type="permutation",
        n_permutations=1000,
        description="Probability output: explain softmax probability (non-additive)"
    ),
    
    "log_odds": SHAPConfig(
        name="log_odds",
        masking_strategy="deletion",
        baseline_type="empty",
        output_type="log_odds",
        explainer_type="permutation",
        n_permutations=1000,
        description="Log-odds output: contrastive explanation vs second-best class"
    ),
    
    # Alternative explainers
    "kernel": SHAPConfig(
        name="kernel",
        masking_strategy="deletion",
        baseline_type="empty",
        output_type="logit",
        explainer_type="kernel",
        n_permutations=1000,
        description="KernelSHAP: weighted linear regression approach"
    ),
    
    "exact": SHAPConfig(
        name="exact",
        masking_strategy="deletion",
        baseline_type="empty",
        output_type="logit",
        explainer_type="exact",
        n_permutations=0,  # Not used
        description="Exact Shapley: enumerate all 2^L coalitions (small L only)"
    ),
    
    # Combined alternatives
    "pad_probability": SHAPConfig(
        name="pad_probability",
        masking_strategy="pad_replace",
        baseline_type="empty",
        output_type="probability",
        explainer_type="permutation",
        n_permutations=1000,
        description="PAD replace + probability: position-preserving, probability output"
    ),
}


# =============================================================================
# MAIN
# =============================================================================

def run_all_experiments(
    model,
    prefix_tokens: np.ndarray,
    max_length: int,
    pad_id: int,
    vocab: dict,
    y_vocab: dict,
    attention: np.ndarray = None,
    training_data: np.ndarray = None,
    configs_to_run: List[str] = None
) -> Dict[str, Dict]:
    """
    Run all specified SHAP configurations and compare results.
    """
    if configs_to_run is None:
        configs_to_run = list(CONFIGS.keys())
    
    # Skip exact for long sequences
    L = len(prefix_tokens)
    if L > 12 and "exact" in configs_to_run:
        print(f"Skipping 'exact' config for L={L} (too expensive)")
        configs_to_run = [c for c in configs_to_run if c != "exact"]
    
    experiment = SHAPExperiment(model, max_length, pad_id, vocab, y_vocab)
    
    results = {}
    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        print(f"\nRunning config: {config_name}")
        print(f"  {config.description}")
        
        try:
            result = experiment.run_experiment(
                config=config,
                prefix_tokens=prefix_tokens,
                attention=attention,
                training_data=training_data
            )
            results[config_name] = result
            
            print(f"  Additivity error: {result['additivity_error']:.2e}")
            if result['alignment']:
                print(f"  Attention correlation: {result['alignment']['pearson']:.3f}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results[config_name] = {"error": str(e)}
    
    return results


def compare_results(results: Dict[str, Dict]) -> None:
    """Print comparison table of results."""
    print("\n" + "="*80)
    print("COMPARISON OF SHAP CONFIGURATIONS")
    print("="*80)
    
    # Header
    print(f"\n{'Config':<20} {'Additivity':<12} {'Pearson':<10} {'Spearman':<10} {'Top-5':<10}")
    print("-"*62)
    
    for name, result in results.items():
        if "error" in result:
            print(f"{name:<20} ERROR: {result['error'][:40]}")
            continue
        
        add_err = result['additivity_error']
        align = result.get('alignment', {})
        pearson = align.get('pearson', float('nan'))
        spearman = align.get('spearman', float('nan'))
        top_k = align.get('top_k_overlap', float('nan'))
        
        print(f"{name:<20} {add_err:<12.2e} {pearson:<10.3f} {spearman:<10.3f} {top_k:<10.1%}")
    
    # Feature importance comparison
    print("\n" + "="*80)
    print("TOP-3 FEATURES BY CONFIGURATION")
    print("="*80)
    
    for name, result in results.items():
        if "error" in result:
            continue
        
        phi = np.array(result['phi'])
        names = result['feature_names']
        top_3 = np.argsort(-np.abs(phi))[:3]
        
        print(f"\n{name}:")
        for i, idx in enumerate(top_3, 1):
            print(f"  #{i}: {names[idx]} (φ={phi[idx]:+.3f})")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Configuration Experiments")
    parser.add_argument("--dataset", default="BPIC2012-W", help="Dataset name")
    parser.add_argument("--prefix_index", type=int, default=12, help="Prefix index")
    parser.add_argument("--experiment", default="all", 
                        help="Config to run: all, current, pad_replace, etc.")
    parser.add_argument("--output", default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    # Import your data loading utilities
    from processtransformer import constants
    from processtransformer.data import loader
    from processtransformer.models import transformer
    
    print(f"\nLoading data for {args.dataset}...")
    
    # Load data
    dl = loader.LogsDataLoader(name=args.dataset)
    train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = \
        dl.load_data(constants.Task.NEXT_ACTIVITY)
    
    pad_token = getattr(constants, "PAD_TOKEN", "[PAD]")
    pad_id = int(x_word_dict.get(pad_token, x_word_dict.get("<pad>", 0)))
    
    # Load model
    model = transformer.get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size,
        output_dim=num_output
    )
    
    ckpt_path = f"./models/{args.dataset}/next_activity_ckpt"
    model.load_weights(ckpt_path).expect_partial()
    print(f"Loaded model from {ckpt_path}")
    
    # Load prefix and attention
    outputs_dir = Path(f"./outputs/{args.dataset}")
    
    with open(outputs_dir / "batch_predictions.json") as f:
        predictions = json.load(f)["predictions"]
    
    prefix_info = predictions[args.prefix_index]
    prefix_activities = prefix_info["prefix_activities"]
    prefix_tokens = np.array([x_word_dict[a] for a in prefix_activities], dtype=np.int32)
    
    print(f"Prefix ({len(prefix_activities)} events): {' → '.join(prefix_activities[:5])}...")
    
    # Load attention
    attention_path = outputs_dir / "block_mha_scores.npy"
    attention = np.load(attention_path)[args.prefix_index] if attention_path.exists() else None
    
    # Select configs to run
    if args.experiment == "all":
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [args.experiment]
    
    # Run experiments
    results = run_all_experiments(
        model=model,
        prefix_tokens=prefix_tokens,
        max_length=max_case_length,
        pad_id=pad_id,
        vocab=x_word_dict,
        y_vocab=y_word_dict,
        attention=attention,
        configs_to_run=configs_to_run
    )
    
    # Compare results
    compare_results(results)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = outputs_dir / f"shap_experiments_batch_{args.prefix_index}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
