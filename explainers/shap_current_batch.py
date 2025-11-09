# explainers/shap_kernel_ppm.py
import os, json, numpy as np, tensorflow as tf
from pathlib import Path

import shap


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


class SuffixKeepMasker:
    def __init__(self, pad_id, T):
        self.pad_id, self.T = int(pad_id), int(T)
    def __call__(self, mask, x):
        import numpy as np
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
    def __init__(self, pad_id, T, min_tokens=0):
        self.pad_id = int(pad_id)
        self.T = int(T)
        self.min_tokens = int(min_tokens)  # 0 ya da 1 önerilir

    def __call__(self, mask, x):
        import numpy as np
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
            # istersen tamamen boş koalisyonu engelle
            if self.min_tokens and L > 0:
                r = max(r, self.min_tokens)
            r = min(r, L)  # üst sınır

            cut = k0 + r             # [k0..cut-1] tutulur, [cut..] PAD
            out[i, cut:] = self.pad_id

        return out


def run_kernel_shap_for_batch(
    model,
    X_batch,                # (B, T) int32 token ids (right-aligned PAD)
    xdict,             # path to xdict.json for PAD id
    run_dir,                # where to save outputs
    background_X=None,      # (Bbg, T) int32; if None -> take first min(B, 20)
    nsamples="auto",        # SHAP sampling
    link_logit=True,        # explain logits (stable)
    class_mode="predicted"  # "predicted" or int (class id) or "label" with y_true array
):
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=True)
    pad_id = xdict.get("[PAD]") or xdict.get("<pad>") or xdict.get("PAD") or 0

    X_batch = np.asarray(X_batch, dtype=np.int32)
    L = get_prefix_lengths(X_batch, pad_id)  # always available for saving later

    B, T = X_batch.shape

    # choose background
    if background_X is None:
        Bbg = min(B, 20)
        background_X = X_batch[:Bbg]
    else:
        background_X = np.asarray(background_X, dtype=np.int32)

    # choose target class per sample
    logits_full = model(X_batch, training=False)
    if isinstance(logits_full, (tuple, list)):
        logits_full = logits_full[0]
    logits_full = np.array(logits_full)

    if logits_full.ndim == 3:
        Lc = np.clip(L, 1, None)
        idx_b = np.arange(X_batch.shape[0])
        last_logits = logits_full[idx_b, Lc - 1, :]  # (B, C)
    elif logits_full.ndim == 2:
        last_logits = logits_full  # (B, C)
    else:
        raise RuntimeError(f"Unexpected model output shape: {logits_full.shape}")

    if class_mode == "predicted":
        c_star = last_logits.argmax(axis=1).astype(np.int32)
    elif isinstance(class_mode, int):
        c_star = np.full((last_logits.shape[0],), int(class_mode), dtype=np.int32)
    else:
        raise ValueError("class_mode must be 'predicted' or an int")

    scorer = build_target_selector(model, pad_id, as_logit=link_logit)

    fullvec_model = model_laststep_logits(model, pad_id)

    #masker = SuffixKeepMasker(pad_id=pad_id, T=T)
    masker = CausalPrefixMasker(pad_id=pad_id, T=T)

    explainer = shap.Explainer(
        #model=lambda X: scorer(np.array(X, dtype=np.int32), c_star[:len(X)]),
        model = fullvec_model,
        masker=masker,
        data=background_X,  # <- small valid prefixes
        algorithm="auto",  # <- SHAP decides (PermutationExplainer under the hood)
        link=shap.links.identity if link_logit else shap.links.logit,
    )

    if isinstance(nsamples, int):
        explanation = explainer(X_batch, max_evals=nsamples)
    else:
        explanation = explainer(X_batch)

    shap_vals = explanation.values

    vals = explanation.values  # (B, T, C)
    idx_b = np.arange(B)
    shap_vals = vals[idx_b, :, c_star]  # (B, T)

    # (optional, for additivity checks)
    base_vals = np.array(explanation.base_values)  # (B, C)
    base_vals_sel = base_vals[idx_b, c_star]  # (B,)

    # save
    out_dir = run
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "shap_values_batch.npy", shap_vals)
    np.save(out_dir / "base_values.npy", base_vals_sel)
    np.save(out_dir / "x_ids_batch.npy", X_batch)
    np.save(out_dir / "c_star_batch.npy", c_star)
    np.save(out_dir / "prefix_lengths.npy", L)

    # (nice to have) save selected logits for sanity:
    selected_logits = last_logits[idx_b, c_star]
    np.save(out_dir / "selected_logits.npy", selected_logits)

    print(f"✅ SHAP saved to: {out_dir}")
    return str(out_dir)
