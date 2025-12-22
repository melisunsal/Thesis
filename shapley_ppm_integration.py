import os
import re
import json
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# -------------------------
# EDIT ONLY THESE
# -------------------------
REPO_ROOT = "/Users/Q671967/PycharmProjects/Thesis"
DATASET = "BPIC2012-O"
BATCH_INDEX = 24

# Where batch_predictions.json is saved for that dataset
BATCH_PRED_PATH = os.path.join(REPO_ROOT, "outputs", DATASET, "batch_predictions.json")

# Output folder
OUT_DIR = os.path.join(REPO_ROOT, "outputs", DATASET, f"shap_batch_index_{BATCH_INDEX}")

# SHAP settings (slow is ok)
EXPLAIN_LOGIT = True          # strongly recommended for sanity checks
NSAMPLES = 8192
BACKGROUND_SIZE = 50
LENGTH_TOL_START = 2

# Verification settings
TOP_K_MAX = 10                # for top-k removal checks
DO_LOO_CHECK = True           # leave-one-out ablation check (O(L) model calls)
DO_TOPK_CHECK = True          # top-k removal faithfulness check

# model weights root
MODELS_ROOT = os.path.join(REPO_ROOT, "models")


def resolve_ckpt_prefix(dataset: str) -> str:
    ds_dir = os.path.join(MODELS_ROOT, dataset)
    ckpt_file = os.path.join(ds_dir, "checkpoint")

    if os.path.exists(ckpt_file):
        with open(ckpt_file, "r", encoding="utf-8") as f:
            txt = f.read()
        m = re.search(r'model_checkpoint_path:\s*"([^"]+)"', txt)
        if m:
            rel = m.group(1)
            return rel if os.path.isabs(rel) else os.path.join(ds_dir, rel)

    return os.path.join(ds_dir, "next_activity_ckpt")


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def left_pad_from_unpadded(tokens_1d: np.ndarray, max_len: int, pad_id: int) -> np.ndarray:
    tokens_1d = np.asarray(tokens_1d, dtype=np.int32)
    compact = tokens_1d[tokens_1d != pad_id]
    out = np.full((max_len,), pad_id, dtype=np.int32)
    if compact.size == 0:
        return out
    if compact.size > max_len:
        compact = compact[-max_len:]
    out[-compact.size:] = compact
    return out


def make_background_from_train(train_token_x: np.ndarray, pad_id: int, target_L: int, k: int, tol_start: int = 2, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = train_token_x.astype(np.int32)
    lengths = (X != pad_id).sum(axis=1).astype(int)

    valid = np.where(lengths > 0)[0]
    tol = int(tol_start)

    while True:
        candidates = np.where((lengths >= 1) & (np.abs(lengths - target_L) <= tol))[0]
        if candidates.size >= k:
            chosen = rng.choice(candidates, size=k, replace=False)
            break
        if tol > max(10, target_L):
            sorted_idx = np.argsort(np.abs(lengths - target_L))
            sorted_idx = [i for i in sorted_idx if lengths[i] > 0]
            chosen = np.array(sorted_idx[:k], dtype=int)
            break
        tol += 1

    bg = np.full((k, target_L), pad_id, dtype=np.int32)
    for j, idx in enumerate(chosen):
        row = X[idx]
        tokens = row[row != pad_id]
        if tokens.size == 0:
            continue
        if tokens.size > target_L:
            tokens = tokens[-target_L:]
        bg[j, -tokens.size:] = tokens

    # ensure no all-PAD rows
    for j in range(k):
        if np.all(bg[j] == pad_id):
            ridx = int(rng.choice(valid))
            tok = X[ridx][X[ridx] != pad_id]
            if tok.size > 0:
                bg[j, -1] = int(tok[-1])

    return bg


def load_prefix_from_batch_predictions(path: str, batch_index: int):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    preds = obj["predictions"]
    for item in preds:
        if int(item.get("batch_index", -1)) == int(batch_index):
            return item["case_id"], item["prefix_activities"], item.get("predicted_label"), item.get("predicted_prob")

    raise ValueError(f"batch_index={batch_index} not found in {path}")


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    # simple rank correlation
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def rank(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        return r

    ra = rank(a)
    rb = rank(b)
    return pearson_corr(ra, rb)


def main():
    # imports from your repo
    from processtransformer.data import loader
    from processtransformer.models import transformer
    from processtransformer import constants

    os.makedirs(OUT_DIR, exist_ok=True)

    case_id, prefix_activities, stored_pred_label, stored_pred_prob = load_prefix_from_batch_predictions(
        BATCH_PRED_PATH, BATCH_INDEX
    )

    print(f"Using batch_index={BATCH_INDEX}, case_id={case_id}, L={len(prefix_activities)}")

    # load dicts and train data for background
    dl = loader.LogsDataLoader(name=DATASET)
    train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = dl.load_data(constants.Task.NEXT_ACTIVITY)

    pad_token = getattr(constants, "PAD_TOKEN", "[PAD]")
    pad_id = int(x_word_dict.get(pad_token, 0))
    inv_x = {v: k for k, v in x_word_dict.items()}
    inv_y = {v: k for k, v in y_word_dict.items()}

    # tokenize prefix
    x_unpadded = np.array([x_word_dict[a] for a in prefix_activities], dtype=np.int32)
    L = int(len(x_unpadded))
    feature_names = [f"E{i+1}:{inv_x.get(int(t), str(int(t)))}" for i, t in enumerate(x_unpadded)]

    # build model + load weights
    model = transformer.get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab_size,
        output_dim=num_output
    )
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    ckpt = resolve_ckpt_prefix(DATASET)
    model.load_weights(ckpt).expect_partial()

    # prediction on this prefix (chooses target class)
    x_padded = left_pad_from_unpadded(x_unpadded, max_case_length, pad_id).reshape(1, -1)
    logits_full = model.predict(x_padded, verbose=0)  # (1,C)
    probs_full = softmax(logits_full, axis=-1)
    class_idx = int(np.argmax(probs_full[0]))
    pred_label = inv_y.get(class_idx, str(class_idx))
    pred_prob = float(probs_full[0, class_idx])

    # f(z) : z is (n, L)
    def f(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        X = np.stack([left_pad_from_unpadded(row, max_case_length, pad_id) for row in z], axis=0)
        logits_ = model.predict(X, verbose=0)
        if EXPLAIN_LOGIT:
            return logits_[:, class_idx].astype(np.float64)
        probs_ = softmax(logits_, axis=-1)
        return probs_[:, class_idx].astype(np.float64)

    # background from train (length-matched)
    train_token_x, _ = dl.prepare_data_next_activity(train_df, x_word_dict, y_word_dict, max_case_length)
    background = make_background_from_train(train_token_x, pad_id, L, BACKGROUND_SIZE, LENGTH_TOL_START, seed=123)
    bg_lengths = (background != pad_id).sum(axis=1).astype(int)
    bg_scores = f(background)

    print(f"Model prediction: {pred_label} (p={pred_prob:.4f}), EXPLAIN_LOGIT={EXPLAIN_LOGIT}")
    if stored_pred_label is not None:
        print(f"Stored prediction in batch file: {stored_pred_label}, prob={stored_pred_prob}")

    print(f"Background length min/med/max: {int(bg_lengths.min())}/{int(np.median(bg_lengths))}/{int(bg_lengths.max())}")
    print(f"Background f() stats: mean={float(np.mean(bg_scores)):.4f}, std={float(np.std(bg_scores)):.4f}, "
          f"min={float(np.min(bg_scores)):.4f}, max={float(np.max(bg_scores)):.4f}")

    # SHAP
    explainer = shap.KernelExplainer(f, background)
    x0 = x_unpadded.reshape(1, L)
    shap_vals_raw = explainer.shap_values(x0, nsamples=NSAMPLES)

    if isinstance(shap_vals_raw, list):
        shap_vals_1 = np.asarray(shap_vals_raw[0], dtype=np.float64)
    else:
        shap_vals_1 = np.asarray(shap_vals_raw, dtype=np.float64)
    shap_vals = shap_vals_1[0]
    base_val = float(np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)[0])

    # Verification 1: additivity (local accuracy)
    fx_direct = float(f(x0)[0])
    fx_additive = base_val + float(np.sum(shap_vals))
    add_err = float(fx_additive - fx_direct)

    # Verification 2: direct model target score vs f(x0)
    if EXPLAIN_LOGIT:
        direct_target_score = float(logits_full[0, class_idx])
    else:
        direct_target_score = float(probs_full[0, class_idx])
    fx_vs_model_err = float(fx_direct - direct_target_score)

    verif = {
        "additivity_error": add_err,
        "f_x_direct": fx_direct,
        "f_x_base_plus_sum": fx_additive,
        "direct_model_target_score": direct_target_score,
        "f_vs_model_error": fx_vs_model_err,
    }

    # Verification 3: leave-one-out ablation correlation
    loo = {}
    if DO_LOO_CHECK:
        deltas = np.zeros((L,), dtype=np.float64)
        for i in range(L):
            z = x_unpadded.copy()
            z[i] = pad_id
            deltas[i] = fx_direct - float(f(z.reshape(1, L))[0])  # drop when removing i

        loo = {
            "loo_delta_mean": float(np.mean(deltas)),
            "loo_delta_max": float(np.max(deltas)),
            "pearson_shap_vs_loo": pearson_corr(shap_vals, deltas),
            "spearman_shap_vs_loo": spearman_corr(shap_vals, deltas),
        }

    # Verification 4: top-k removal drops
    topk = {}
    if DO_TOPK_CHECK:
        kmax = int(min(TOP_K_MAX, L))

        order_pos = np.argsort(-shap_vals)  # most positive first
        order_abs = np.argsort(-np.abs(shap_vals))

        drops_pos = []
        drops_abs = []

        for k in range(1, kmax + 1):
            z_pos = x_unpadded.copy()
            z_pos[order_pos[:k]] = pad_id
            drops_pos.append(fx_direct - float(f(z_pos.reshape(1, L))[0]))

            z_abs = x_unpadded.copy()
            z_abs[order_abs[:k]] = pad_id
            drops_abs.append(fx_direct - float(f(z_abs.reshape(1, L))[0]))

        topk = {
            "kmax": kmax,
            "drops_topk_positive": [float(x) for x in drops_pos],
            "drops_topk_abs": [float(x) for x in drops_abs],
            "top1_positive_feature": feature_names[int(order_pos[0])],
            "top1_abs_feature": feature_names[int(order_abs[0])],
        }

    # Save JSON
    out_json = {
        "dataset": DATASET,
        "batch_index": BATCH_INDEX,
        "case_id": case_id,
        "prefix_activities": prefix_activities,
        "stored_predicted_label": stored_pred_label,
        "stored_predicted_prob": stored_pred_prob,
        "model_predicted_label": pred_label,
        "model_predicted_prob": pred_prob,
        "predicted_class_index": class_idx,
        "explain_logit": EXPLAIN_LOGIT,
        "L": L,
        "feature_names": feature_names,
        "base_value": base_val,
        "shap_values": shap_vals.tolist(),
        "settings": {
            "nsamples": NSAMPLES,
            "background_size": BACKGROUND_SIZE,
            "length_tol_start": LENGTH_TOL_START,
        },
        "background_stats": {
            "length_min": int(bg_lengths.min()),
            "length_median": int(np.median(bg_lengths)),
            "length_max": int(bg_lengths.max()),
            "f_mean": float(np.mean(bg_scores)),
            "f_std": float(np.std(bg_scores)),
            "f_min": float(np.min(bg_scores)),
            "f_max": float(np.max(bg_scores)),
        },
        "verification": verif,
        "verification_loo": loo,
        "verification_topk": topk,
    }

    with open(os.path.join(OUT_DIR, "shap_explanation.json"), "w", encoding="utf-8") as f_out:
        json.dump(out_json, f_out, indent=2)

    # Plots
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=np.array(prefix_activities, dtype=object),
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Waterfall, pred={pred_label} (batch_index={BATCH_INDEX})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_waterfall.png"), dpi=200)
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=min(20, L))
    plt.title(f"SHAP Bar, pred={pred_label} (batch_index={BATCH_INDEX})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar.png"), dpi=200)
    plt.close()

    # Print verification summary
    print("\n" + "=" * 80)
    print(f"Saved SHAP outputs to: {OUT_DIR}")
    print(f"Additivity: f(x)={fx_direct:.6f}, base+sum={fx_additive:.6f}, error={add_err:.3e}")
    print(f"f(x) vs direct model target score error: {fx_vs_model_err:.3e}")
    if DO_LOO_CHECK and loo:
        print(f"LOO correlation: pearson={loo['pearson_shap_vs_loo']:.3f}, spearman={loo['spearman_shap_vs_loo']:.3f}")
    if DO_TOPK_CHECK and topk:
        print(f"Top-1 positive: {topk['top1_positive_feature']}")
        print(f"Top-1 abs:      {topk['top1_abs_feature']}")
        print(f"Drops top-k positive (k=1..{topk['kmax']}): {topk['drops_topk_positive']}")
        print(f"Drops top-k abs      (k=1..{topk['kmax']}): {topk['drops_topk_abs']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
