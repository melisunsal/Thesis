# explainers/shap_kernel_ppm.py
import os, json, time
import numpy as np
import tensorflow as tf
from pathlib import Path
import shap

# =========================
# Utilities
# =========================

def get_prefix_lengths(X, pad_id):
    # right-aligned padding → length = count of non-PAD tokens
    return np.sum(X != pad_id, axis=1)

@tf.function
def _forward_tf(model, X_ids):
    return model(X_ids, training=False)

def model_laststep_logits(model, pad_id):
    """Return a function f(X)->(n,C) that extracts last-step class scores for any batch X."""
    def _fn(X):
        X = np.array(X, dtype=np.int32)
        logits = _forward_tf(model, tf.convert_to_tensor(X, dtype=tf.int32))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = np.array(logits)

        if logits.ndim == 3:
            # pick last valid timestep per row (right-aligned PAD)
            Ltmp = (X != pad_id).sum(axis=1)
            Ltmp = np.clip(Ltmp, 1, None)
            idx = np.arange(X.shape[0])
            return logits[idx, Ltmp - 1, :]   # (n, C)
        elif logits.ndim == 2:
            return logits                     # (n, C)
        else:
            raise RuntimeError(f"Unexpected model output shape: {logits.shape}")
    return _fn

def build_target_selector(model, pad_id, as_logit=True):
    @tf.function
    def _forward(X_ids):
        return model(X_ids, training=False)

    def scorer(X_ids, c_star):
        logits = _forward(tf.convert_to_tensor(X_ids, dtype=tf.int32))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = np.array(logits)

        if logits.ndim == 3:
            L = get_prefix_lengths(np.asarray(X_ids, dtype=np.int32), pad_id)
            Lc = np.clip(L, 1, None)
            idx_b = np.arange(logits.shape[0])
            last_logits = logits[idx_b, Lc - 1, :]  # (B, C)
        elif logits.ndim == 2:
            last_logits = logits                     # (B, C)
            idx_b = np.arange(last_logits.shape[0])
        else:
            raise RuntimeError(f"Unexpected model output shape: {logits.shape}")

        if as_logit:
            return last_logits[idx_b, c_star]
        # probs
        exp = np.exp(last_logits - np.max(last_logits, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[idx_b, c_star]
    return scorer

# =========================
# Maskers
# =========================

class SuffixKeepMasker:
    """Tutulan olay sayısına göre en sondaki r olayı korur, diğerlerini PAD yapar."""
    def __init__(self, pad_id, T):
        self.pad_id, self.T = int(pad_id), int(T)
    def __call__(self, mask, x):
        x = np.asarray(x, np.int32); T = self.T
        L = int((x != self.pad_id).sum()); a = x[T-L:]  # active tail
        M = np.asarray(mask);  M = M[None,:] if M.ndim==1 else M
        M = M.astype(bool)
        out = np.tile(x, (M.shape[0], 1))
        for i in range(M.shape[0]):
            r = int(M[i, -L:].sum())          # how many active events to keep
            kept = a[L-r:] if r>0 else np.array([], dtype=np.int32)
            out[i, :] = self.pad_id
            out[i, T-r:] = kept
        return out

class CausalPrefixMasker:
    """Aktif bölgedeki açık bit sayısı kadar baştan r olayı tutar, geri kalanı PAD yapar."""
    def __init__(self, pad_id, T, min_tokens=0):
        self.pad_id = int(pad_id)
        self.T = int(T)
        self.min_tokens = int(min_tokens)  # 0 ya da 1 önerilir

    def __call__(self, mask, x):
        x = np.asarray(x, dtype=np.int32)
        if x.ndim != 1 or x.shape[0] != self.T:
            raise ValueError(f"x shape {x.shape} != (T,) with T={self.T}")

        # aktif bölge: [k0..T-1]
        L  = int(np.sum(x != self.pad_id))
        k0 = self.T - L

        M = np.asarray(mask)
        if M.ndim == 1:
            M = M[None, :]
        if M.shape[1] != self.T:
            raise ValueError(f"mask shape {M.shape} != (?, T) with T={self.T}")

        M = M.astype(bool)
        out = np.tile(x, (M.shape[0], 1))

        for i, mi in enumerate(M):
            # aktif bölgede açık olan özellik sayısı = r
            r = int(mi[k0:].sum())
            # tamamen boş koalisyonu engelle
            if self.min_tokens and L > 0:
                r = max(r, self.min_tokens)
            r = min(r, L)  # üst sınır

            cut = k0 + r             # [k0..cut-1] tutulur, [cut..] PAD
            out[i, cut:] = self.pad_id

        return out

class SubsequenceMasker:
    """
    Aktif bölgeden True olan konumları sırayı koruyarak seçer ve sağa hizalar.
    Böylece 'tek tek olay' koalisyonlarını gerçekten uygular.
    """
    def __init__(self, pad_id, T):
        self.pad_id, self.T = int(pad_id), int(T)
    def __call__(self, mask, x):
        x = np.asarray(x, np.int32); T = self.T
        if x.ndim != 1 or x.shape[0] != T:
            raise ValueError(f"x shape {x.shape} != (T,) with T={T}")
        L  = int((x != self.pad_id).sum())
        k0 = T - L                      # aktif bölgenin başı
        M  = np.atleast_2d(mask).astype(bool)
        out = np.full((M.shape[0], T), self.pad_id, np.int32)
        active = x[k0:]
        for i, mi in enumerate(M):
            sel = active[mi[k0:]]       # sırayı koru
            t = sel.shape[0]
            out[i, T - t:] = sel        # sağa hizala
        return out

def choose_masker(mode, pad_id, T, **kwargs):
    mode = (mode or "causal_prefix").lower()
    if mode in ["causal", "causal_prefix", "prefix", "prefix_add"]:
        return CausalPrefixMasker(pad_id=pad_id, T=T, min_tokens=kwargs.get("min_tokens", 0))
    if mode in ["subsequence", "subseq", "event"]:
        return SubsequenceMasker(pad_id=pad_id, T=T)
    if mode in ["suffix", "suffix_keep"]:
        return SuffixKeepMasker(pad_id=pad_id, T=T)
    raise ValueError(f"Unknown masker_mode: {mode}")

# =========================
# Background sampling
# =========================

def build_background_prefixes(X_all, lengths, y_pred=None, k_per_decile=3, max_bg=40, rng_seed=123):
    """
    X_all: np.ndarray [N, T] of token ids from the test set (or train+valid)
    lengths: np.ndarray [N,] valid prefix lengths for each row
    y_pred: optional np.ndarray [N,] predicted class per row, used to avoid one-class dominance
    """
    rng = np.random.default_rng(rng_seed)

    # make deciles over lengths
    qs = np.quantile(lengths, np.linspace(0, 1, 11), method="nearest")
    idxs = []
    for a, b in zip(qs[:-1], qs[1:]):
        mask = (lengths >= a) & (lengths <= b)
        cand = np.where(mask)[0]
        if cand.size == 0:
            continue
        pick = rng.choice(cand, size=min(k_per_decile, cand.size), replace=False)
        idxs.extend(pick.tolist())

    idxs = np.array(list(dict.fromkeys(idxs)))  # unique, keep order

    # optional light balancing by predicted class
    if y_pred is not None and idxs.size > 0:
        taken = []
        for c in np.unique(y_pred[idxs]):
            cand = idxs[y_pred[idxs] == c]
            take = cand[: max(1, len(cand)//4)]
            taken.extend(take.tolist())
        idxs = np.array(list(dict.fromkeys(taken))) or idxs

    # cap size
    if idxs.size > max_bg:
        idxs = idxs[:max_bg]

    return X_all[idxs]

# =========================
# Verification helpers
# =========================

def compute_additivity_metrics(explanation, c_star, selected_logits):
    """
    explanation.values: (B, T, C)
    explanation.base_values: (B, C)
    """
    vals = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    B, T, C = vals.shape
    idx = np.arange(B)
    shap_sel = vals[idx, :, c_star]             # (B, T)
    base_sel = base[idx, c_star]                # (B,)
    recon = base_sel + shap_sel.sum(axis=1)     # (B,)
    resid = selected_logits - recon             # (B,)
    out = {
        "additivity_mae": float(np.mean(np.abs(resid))),
        "additivity_max_abs": float(np.max(np.abs(resid))),
    }
    return out, shap_sel, base_sel, resid

def pad_value_mean(shap_vals, X_batch, pad_id):
    pad_mask = (X_batch == pad_id)
    if not np.any(pad_mask):
        return 0.0
    return float(np.mean(np.abs(shap_vals[pad_mask])))

def truncate_suffix_by_k(X, pad_id, k):
    """Son k olayı kesip PAD yap."""
    X = np.asarray(X, np.int32).copy()
    L = (X != pad_id).sum(axis=1)
    B, T = X.shape
    for i in range(B):
        Li = int(L[i])
        if Li <= 1:
            continue
        k_use = min(k, max(0, Li - 1))
        k0 = T - Li
        X[i, k0 + (Li - k_use):] = pad_id
    return X

def prefix_truncation_eval(model, X_batch, pad_id, k_list=(1,2,3)):
    f = model_laststep_logits(model, pad_id)
    full = f(X_batch)
    # target class per row from full
    c_star = full.argmax(axis=1)
    full_sel = full[np.arange(full.shape[0]), c_star]
    deltas = {}
    for k in k_list:
        X_tr = truncate_suffix_by_k(X_batch, pad_id, k)
        sc = f(X_tr)
        sc_sel = sc[np.arange(sc.shape[0]), c_star]
        deltas[f"k{k}_mean_drop"] = float(np.mean(full_sel - sc_sel))
    return deltas

def _subsequence_remove_topk_single_row(x_row, shap_row, pad_id, k):
    """Aktif bölgeden SHAP'e göre en yüksek k olayı kaldırıp sağa hizalar."""
    x_row = np.asarray(x_row, np.int32)
    T = x_row.shape[0]
    L = int((x_row != pad_id).sum())
    if L <= 1 or k <= 0:
        return x_row.copy()
    k = min(k, L - 1)
    k0 = T - L
    phi_act = shap_row[k0:]
    # en yüksek katkılı k pozisyonu kaldır
    idxs = np.argsort(-phi_act)[:k]  # descending
    keep_mask = np.ones(L, dtype=bool)
    keep_mask[idxs] = False
    active = x_row[k0:]
    kept = active[keep_mask]
    out = np.full(T, pad_id, np.int32)
    out[T - kept.shape[0]:] = kept
    return out

def _subsequence_remove_random_single_row(x_row, pad_id, k, rng):
    x_row = np.asarray(x_row, np.int32)
    T = x_row.shape[0]
    L = int((x_row != pad_id).sum())
    if L <= 1 or k <= 0:
        return x_row.copy()
    k = min(k, L - 1)
    k0 = T - L
    choices = rng.choice(np.arange(L), size=k, replace=False)
    keep_mask = np.ones(L, dtype=bool)
    keep_mask[choices] = False
    active = x_row[k0:]
    kept = active[keep_mask]
    out = np.full(T, pad_id, np.int32)
    out[T - kept.shape[0]:] = kept
    return out

def subsequence_deletion_eval(model, X_batch, pad_id, shap_vals, k_list=(1,2,3), random_trials=20, seed=42):
    """
    SHAP'e göre top-k olayı kaldırınca hedef logit düşüşü,
    aynı k için rastgele kaldırmadan daha fazla olmalı.
    """
    f = model_laststep_logits(model, pad_id)
    full_scores = f(X_batch)
    c_star = full_scores.argmax(axis=1)
    full_sel = full_scores[np.arange(full_scores.shape[0]), c_star]
    results = {}
    rng = np.random.default_rng(seed)

    for k in k_list:
        # top-k kaldır
        X_top = np.stack([_subsequence_remove_topk_single_row(x, s, pad_id, k)
                          for x, s in zip(X_batch, shap_vals)], axis=0)
        sc_top = f(X_top)
        sc_top_sel = sc_top[np.arange(sc_top.shape[0]), c_star]
        drop_top = full_sel - sc_top_sel
        mean_top = float(np.mean(drop_top))

        # random kaldır ortalaması
        rand_drops = []
        for _ in range(random_trials):
            X_r = np.stack([_subsequence_remove_random_single_row(x, pad_id, k, rng)
                            for x in X_batch], axis=0)
            sc_r = f(X_r)
            sc_r_sel = sc_r[np.arange(sc_r.shape[0]), c_star]
            rand_drops.append(full_sel - sc_r_sel)
        mean_rand = float(np.mean(np.stack(rand_drops, axis=0)))
        results[f"k{k}_topk_mean_drop"] = mean_top
        results[f"k{k}_random_mean_drop"] = mean_rand
        results[f"k{k}_diff_topk_minus_random"] = float(mean_top - mean_rand)
    return results

# =========================
# Main runner
# =========================

def run_shap(
    model,
    X_batch,                # (B, T) int32 token ids (right-aligned PAD)
    xdict,                  # dict containing PAD id etc.
    run_dir,                # where to save outputs
    background_X=None,      # (Bbg, T) int32; if None -> take first min(B, 20)
    nsamples="auto",        # SHAP sampling
    link_logit=True,        # explain logits (stable)
    class_mode="predicted", # "predicted" or int (class id)
    masker_mode="causal_prefix",  # "causal_prefix" or "subsequence" or "suffix_keep"
    verify=True,
    trunc_k_list=(1,2,3),
    del_k_list=(1,2,3),
    random_trials=20
):
    t0 = time.time()
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=True)
    pad_id = xdict.get("[PAD]") or xdict.get("<pad>") or xdict.get("PAD") or 0

    X_batch = np.asarray(X_batch, dtype=np.int32)
    B, T = X_batch.shape
    L = get_prefix_lengths(X_batch, pad_id)

    # choose background
    if background_X is None:
        Bbg = min(B, 20)
        background_X = X_batch[:Bbg]
    else:
        background_X = np.asarray(background_X, dtype=np.int32)

    # logits for class selection
    logits_full = model(X_batch, training=False)
    if isinstance(logits_full, (tuple, list)):
        logits_full = logits_full[0]
    logits_full = np.array(logits_full)

    if logits_full.ndim == 3:
        Lc = np.clip(L, 1, None)
        idx_b = np.arange(B)
        last_logits = logits_full[idx_b, Lc - 1, :]  # (B, C)
    elif logits_full.ndim == 2:
        last_logits = logits_full  # (B, C)
        idx_b = np.arange(B)
    else:
        raise RuntimeError(f"Unexpected model output shape: {logits_full.shape}")

    if class_mode == "predicted":
        c_star = last_logits.argmax(axis=1).astype(np.int32)
    elif isinstance(class_mode, int):
        c_star = np.full((last_logits.shape[0],), int(class_mode), dtype=np.int32)
    else:
        raise ValueError("class_mode must be 'predicted' or an int")

    # explainer
    fullvec_model = model_laststep_logits(model, pad_id)
    masker = choose_masker(masker_mode, pad_id=pad_id, T=T)

    explainer = shap.Explainer(
        model=fullvec_model,                   # returns (n, C)
        masker=masker,
        data=background_X,                     # small set of valid prefixes
        algorithm="permutation",               # model-agnostic Shapley
        link=(shap.links.identity if link_logit else shap.links.logit),
    )

    # run SHAP
    if isinstance(nsamples, int):
        explanation = explainer(X_batch, max_evals=nsamples)
    else:
        explanation = explainer(X_batch)

    vals = np.asarray(explanation.values)      # (B, T, C)
    base = np.asarray(explanation.base_values) # (B, C)
    shap_vals = vals[idx_b, :, c_star]         # (B, T)
    base_vals_sel = base[idx_b, c_star]        # (B,)
    selected_logits = last_logits[idx_b, c_star]

    # save core outputs
    out_dir = run
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "shap_values_batch.npy", shap_vals)
    np.save(out_dir / "base_values.npy", base_vals_sel)
    np.save(out_dir / "x_ids_batch.npy", X_batch)
    np.save(out_dir / "c_star_batch.npy", c_star)
    np.save(out_dir / "prefix_lengths.npy", L)
    np.save(out_dir / "selected_logits.npy", selected_logits)

    report = {
        "masker_mode": masker_mode,
        "pad_id": int(pad_id),
        "B": int(B),
        "T": int(T),
        "link_logit": bool(link_logit),
        "class_mode": class_mode,
        "nsamples": nsamples if isinstance(nsamples, int) else str(nsamples)
    }

    # verification
    if verify:
        add_metrics, shap_sel, base_sel, resid = compute_additivity_metrics(
            explanation, c_star, selected_logits
        )
        report.update(add_metrics)
        report["pad_absphi_mean"] = pad_value_mean(shap_vals, X_batch, pad_id)

        if masker_mode.lower().startswith("causal"):
            report["prefix_truncation"] = prefix_truncation_eval(
                model, X_batch, pad_id, k_list=trunc_k_list
            )
        if masker_mode.lower().startswith("subsequence"):
            report["subsequence_deletion"] = subsequence_deletion_eval(
                model, X_batch, pad_id, shap_vals,
                k_list=del_k_list, random_trials=random_trials
            )

        # save residuals for debug
        np.save(out_dir / "additivity_residuals.npy", resid)

    # write verification summary
    with open(out_dir / "verification.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ SHAP saved to: {out_dir}")
    print(f"ℹ️  Verification: {json.dumps(report, ensure_ascii=False)}")
    print(f"⏱️  Elapsed: {time.time() - t0:.2f}s")
    return str(out_dir)
