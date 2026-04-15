#!/usr/bin/env python3
"""
anonymous_institutionAgentMonty - Execution Node: bebcec9b
Type: original
Generated: 2026-01-13T22:27:27.942999
Mutation: None
Parent: None

DO NOT DELETE - This file is preserved for reproducibility.
"""
# Suppress warnings to prevent false failures
import warnings
warnings.filterwarnings("ignore")

#!/usr/bin/env python3
"""
Improved Method: LightGBM + Temporal Wide/Stats Features + OOF Target Encoding
+ Hurdle-style auxiliaries (p(neg), p(zero), pos/neg magnitude experts)
+ Conservative post-processing with SMALL grid-search on VAL only.

Goals vs baseline:
- Better user-level feature engineering that preserves 7-day temporal structure
- Strong GBDT models (often superior on tabular LTV)
- Reduce Error Rate via affine calibration (scale+shift) tuned on VAL only
- Conservative gating near zero using p(neg), p(zero), and quantile (tau=0.20)

CRITICAL: scoring/metric functions are preserved EXACTLY as provided.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import KFold
import lightgbm as lgb

# ---------------------------
# Environment / Paths
# ---------------------------
device = "cpu"  # (not used; kept for compatibility with prior templates)

TRAIN_PATH = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"
DAY_COL = "TDATE_RN"
N_DAYS = 7

# ---------------------------
# DO NOT MODIFY: scoring/metrics (preserved exactly)
# ---------------------------
def calc_gini(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    if len(y_true) == 0 or np.sum(y_true) <= 0: return 0.0
    order = np.argsort(y_pred)[::-1]
    cumsum = np.cumsum(y_true[order])
    total = cumsum[-1]
    if total == 0: return 0.0
    gini_actual = 2 * np.sum(cumsum / total) / len(y_true) - 1
    gini_perfect = 2 * np.sum(np.cumsum(np.sort(y_true)[::-1]) / total) / len(y_true) - 1
    return gini_actual / gini_perfect if gini_perfect != 0 else 1.0

def compute_score(y_true, y_pred):
    gini = calc_gini(y_true, y_pred)
    error_rate = abs(np.sum(y_true) - np.sum(y_pred)) / abs(np.sum(y_true))
    spearman = spearmanr(y_true, y_pred)[0]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    gs = np.clip((gini - 0.70) / 0.20, 0, 1)
    es = np.clip((0.35 - error_rate) / 0.34, 0, 1)
    ss = np.clip((spearman - 0.50) / 0.30, 0, 1)
    rs = np.clip((260 - rmse) / 60, 0, 1)
    
    base = 0.35*gs + 0.25*ss + 0.20*rs + 0.20*es
    pareto = sum([0.035*(gs>0.8), 0.025*(ss>0.8), 0.02*(rs>0.8), 0.02*(es>0.8)])
    exc = sum([gs>0.8, ss>0.8, rs>0.8, es>0.8])
    if exc >= 2: pareto += 0.02
    if exc >= 3: pareto += 0.03
    if exc == 4: pareto += 0.05
    
    final = base + pareto
    print(f"📊 Gini={gini:.4f}, Err={error_rate:.4f}, Spear={spearman:.4f}, RMSE={rmse:.2f}, Score={final:.4f}")
    return final, {'gini': gini, 'error_rate': error_rate, 'spearman': spearman, 'rmse': rmse}

# Optional compatibility shim (does NOT modify scoring)
def compute_pareto_multi_objective(y_true, y_pred):
    s, _ = compute_score(y_true, y_pred)
    return s


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def sign_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

def ensure_day_col(df: pd.DataFrame) -> pd.DataFrame:
    if DAY_COL not in df.columns:
        df = df.sort_values([ID_COL]).copy()
        df[DAY_COL] = df.groupby(ID_COL).cumcount() + 1
    return df

def infer_columns(train_df: pd.DataFrame):
    cat_cols = [
        c for c in train_df.columns
        if c not in (ID_COL, TARGET_COL, DAY_COL) and str(train_df[c].dtype) in ("object", "category")
    ]
    num_cols = [
        c for c in train_df.columns
        if c not in (ID_COL, TARGET_COL, DAY_COL) and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    return num_cols, cat_cols

def build_cat_maps(train_df: pd.DataFrame, cat_cols):
    maps = {}
    for c in cat_cols:
        vals = train_df[c].fillna("UNK").astype(str).values
        uniq = pd.unique(vals)
        mapping = {"UNK": 0}
        k = 1
        for v in uniq:
            if v == "UNK":
                continue
            mapping[v] = k
            k += 1
        maps[c] = mapping
    return maps

def encode_user_cats(df: pd.DataFrame, user_ids: np.ndarray, cat_cols, cat_maps):
    if not cat_cols:
        return np.zeros((len(user_ids), 0), dtype=np.int32), []
    df = ensure_day_col(df).sort_values([ID_COL, DAY_COL]).drop_duplicates([ID_COL, DAY_COL], keep="last")

    # take first observed per user (most cats are static anyway)
    first = df.groupby(ID_COL)[cat_cols].first().reindex(user_ids)
    Xc = np.zeros((len(user_ids), len(cat_cols)), dtype=np.int32)
    for j, c in enumerate(cat_cols):
        mp = cat_maps[c]
        vals = first[c].fillna("UNK").astype(str).values
        Xc[:, j] = np.array([mp.get(v, 0) for v in vals], dtype=np.int32)
    return Xc, cat_cols

def build_user_numeric_tensor(df: pd.DataFrame, user_ids: np.ndarray, num_cols, n_days: int = 7):
    df = ensure_day_col(df)
    df = df.sort_values([ID_COL, DAY_COL]).copy()
    df = df.drop_duplicates([ID_COL, DAY_COL], keep="last")

    u2i = pd.Series(np.arange(len(user_ids)), index=user_ids)
    idx = u2i.loc[df[ID_COL].values].values
    t_raw = df[DAY_COL].astype(np.int32).values - 1
    t = np.clip(t_raw, 0, n_days - 1)

    if num_cols:
        X_rows = df[num_cols].fillna(0).astype(np.float32).values
        X_rows = sign_log1p(X_rows).astype(np.float32)
    else:
        X_rows = np.zeros((len(df), 0), dtype=np.float32)

    U = len(user_ids)
    D = X_rows.shape[1]
    X = np.zeros((U, n_days, D), dtype=np.float32)
    mask = np.zeros((U, n_days, 1), dtype=np.float32)

    X[idx, t, :] = X_rows
    mask[idx, t, 0] = 1.0
    return X, mask

def numeric_time_features(X: np.ndarray, mask: np.ndarray, num_cols, prefix_days=True):
    """
    X: (U,T,D) float32 (already sign_log1p)
    mask: (U,T,1) float32 in {0,1}
    Returns: X_num (U,F) float32, feature_names
    """
    U, T, D = X.shape
    m = mask  # (U,T,1)
    cnt = np.clip(m.sum(axis=1), 1.0, None)  # (U,1)
    sum_x = (X * m).sum(axis=1)  # (U,D)
    mean = sum_x / cnt  # (U,D)

    # std
    xc = X - mean[:, None, :]
    var = ((xc * xc) * m).sum(axis=1) / cnt
    std = np.sqrt(np.maximum(var, 0)).astype(np.float32)

    # min/max with mask
    big = np.float32(1e9)
    X_min = np.where(m.astype(bool), X, big).min(axis=1).astype(np.float32)
    X_max = np.where(m.astype(bool), X, -big).max(axis=1).astype(np.float32)

    first = X[:, 0, :].astype(np.float32)
    last = X[:, -1, :].astype(np.float32)
    delta = (last - first).astype(np.float32)

    # slope (per-feature)
    tvec = np.arange(T, dtype=np.float32)[None, :, None]  # (1,T,1)
    tmean = (tvec * m).sum(axis=1) / cnt  # (U,1)
    tc = (tvec - tmean[:, None, :])  # (U,T,1) via broadcast
    cov = ((tc * xc) * m).sum(axis=1)  # (U,D)
    var_t = ((tc * tc) * m).sum(axis=1) + 1e-6  # (U,1)
    slope = (cov / var_t).astype(np.float32)

    # daywise flatten
    feats = []
    names = []

    if prefix_days:
        Xflat = X.reshape(U, T * D).astype(np.float32)
        feats.append(Xflat)
        for d in range(T):
            for c in num_cols:
                names.append(f"{c}_d{d+1}")

    # stats
    feats.extend([mean, std, X_min, X_max, delta, slope])
    for c in num_cols: names.append(f"{c}__mean")
    for c in num_cols: names.append(f"{c}__std")
    for c in num_cols: names.append(f"{c}__min")
    for c in num_cols: names.append(f"{c}__max")
    for c in num_cols: names.append(f"{c}__delta")
    for c in num_cols: names.append(f"{c}__slope")

    X_num = np.concatenate(feats, axis=1).astype(np.float32)
    return X_num, names

def drop_constant_columns(Xtr, Xva, Xte, names):
    # drop features with 0 variance on train (after float32)
    v = Xtr.var(axis=0)
    keep = v > 1e-12
    Xtr2 = Xtr[:, keep]
    Xva2 = Xva[:, keep]
    Xte2 = Xte[:, keep]
    names2 = [n for n, k in zip(names, keep) if k]
    return Xtr2, Xva2, Xte2, names2

def oof_target_encode_single(cat_train: np.ndarray, y_train: np.ndarray, cat_other: np.ndarray,
                             n_splits=5, alpha=30.0, seed=42):
    """
    Smoothed target mean encoding (OOF for train, mapping for other).
    cat_* are integer codes.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = float(np.mean(y_train))
    oof = np.zeros(len(cat_train), dtype=np.float32)

    s_cat = pd.Series(cat_train)
    s_y = pd.Series(y_train.astype(np.float32))

    for tr_idx, va_idx in kf.split(cat_train):
        c_tr = s_cat.iloc[tr_idx]
        y_tr = s_y.iloc[tr_idx]
        stats = pd.DataFrame({"c": c_tr.values, "y": y_tr.values}).groupby("c")["y"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
        mapping = smooth.to_dict()
        oof[va_idx] = pd.Series(cat_train[va_idx]).map(mapping).fillna(global_mean).values.astype(np.float32)

    # fit full mapping
    stats_full = pd.DataFrame({"c": cat_train, "y": y_train.astype(np.float32)}).groupby("c")["y"].agg(["mean", "count"])
    smooth_full = (stats_full["mean"] * stats_full["count"] + global_mean * alpha) / (stats_full["count"] + alpha)
    mapping_full = smooth_full.to_dict()

    other = pd.Series(cat_other).map(mapping_full).fillna(global_mean).values.astype(np.float32)
    return oof, other

def build_target_encoding_features(Xc_tr, ytr, Xc_va, Xc_te, cat_names, seed=42):
    if Xc_tr.shape[1] == 0:
        return (np.zeros((len(ytr), 0), np.float32),
                np.zeros((len(Xc_va), 0), np.float32),
                np.zeros((len(Xc_te), 0), np.float32),
                [])

    te_tr_list, te_va_list, te_te_list, te_names = [], [], [], []

    y = ytr.astype(np.float32)
    y_neg = (ytr < 0).astype(np.float32)
    y_zero = (ytr == 0).astype(np.float32)

    for j, cname in enumerate(cat_names):
        c_tr = Xc_tr[:, j]
        c_va = Xc_va[:, j]
        c_te = Xc_te[:, j]

        oof_mean, va_mean = oof_target_encode_single(c_tr, y, c_va, n_splits=5, alpha=30.0, seed=seed)
        _, te_mean = oof_target_encode_single(c_tr, y, c_te, n_splits=5, alpha=30.0, seed=seed)

        oof_neg, va_neg = oof_target_encode_single(c_tr, y_neg, c_va, n_splits=5, alpha=50.0, seed=seed)
        _, te_neg = oof_target_encode_single(c_tr, y_neg, c_te, n_splits=5, alpha=50.0, seed=seed)

        oof_zero, va_zero = oof_target_encode_single(c_tr, y_zero, c_va, n_splits=5, alpha=50.0, seed=seed)
        _, te_zero = oof_target_encode_single(c_tr, y_zero, c_te, n_splits=5, alpha=50.0, seed=seed)

        te_tr_list += [oof_mean[:, None], oof_neg[:, None], oof_zero[:, None]]
        te_va_list += [va_mean[:, None], va_neg[:, None], va_zero[:, None]]
        te_te_list += [te_mean[:, None], te_neg[:, None], te_zero[:, None]]
        te_names += [f"{cname}__te_ymean", f"{cname}__te_neg", f"{cname}__te_zero"]

    te_tr = np.concatenate(te_tr_list, axis=1).astype(np.float32)
    te_va = np.concatenate(te_va_list, axis=1).astype(np.float32)
    te_te = np.concatenate(te_te_list, axis=1).astype(np.float32)
    return te_tr, te_va, te_te, te_names


# ---------------------------
# LightGBM models
# ---------------------------
def lgb_aw_obj(preds, train_data):
    """
    Custom AWMSE gradients/hessians.
    False positives (pred>0, y<0) penalized more.
    """
    y = train_data.get_label()
    fp = (preds > 0) & (y < 0)
    fn = (preds < 0) & (y > 0)

    w = np.ones_like(y, dtype=np.float32)
    if np.any(fp):
        w[fp] = (2.5 + 0.02 * np.abs(y[fp])).astype(np.float32)
    if np.any(fn):
        w[fn] = (1.5 + 0.01 * np.clip(y[fn], 0, None)).astype(np.float32)

    grad = 2.0 * w * (preds - y)
    hess = 2.0 * w
    return grad, hess

def train_lgb_reg_aw(Xtr, ytr, Xva, yva, cat_idx, seed=42):
    params = dict(
        boosting_type="gbdt",
        objective="none",
        metric=["rmse"],
        learning_rate=0.03,
        num_leaves=160,
        min_data_in_leaf=60,
        feature_fraction=0.65,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=2.0,
        max_depth=-1,
        verbosity=-1,
        seed=seed,
        num_threads=max(1, os.cpu_count() or 1),
    )
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_idx, free_raw_data=False)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=8000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
        fobj=lgb_aw_obj,
    )
    return model

def train_lgb_quantile(Xtr, ytr, Xva, yva, cat_idx, alpha=0.20, seed=42):
    params = dict(
        boosting_type="gbdt",
        objective="quantile",
        alpha=alpha,
        metric=["quantile"],
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=80,
        feature_fraction=0.70,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=2.0,
        max_depth=-1,
        verbosity=-1,
        seed=seed,
        num_threads=max(1, os.cpu_count() or 1),
    )
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_idx, free_raw_data=False)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=6000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

def train_lgb_binary(Xtr, ytr_bin, Xva, yva_bin, cat_idx, seed=42):
    pos = float(np.sum(ytr_bin == 1))
    neg = float(np.sum(ytr_bin == 0))
    scale_pos_weight = (neg / max(pos, 1.0))

    params = dict(
        boosting_type="gbdt",
        objective="binary",
        metric=["auc"],
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=120,
        feature_fraction=0.75,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=2.0,
        max_depth=-1,
        verbosity=-1,
        seed=seed,
        scale_pos_weight=scale_pos_weight,
        num_threads=max(1, os.cpu_count() or 1),
    )
    dtr = lgb.Dataset(Xtr, label=ytr_bin, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva_bin, categorical_feature=cat_idx, free_raw_data=False)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=5000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

def train_lgb_mag_reg(Xtr, ytr_mag, Xva, yva_mag, cat_idx, seed=42):
    # magnitude regression in log1p space (targets are >= 0)
    params = dict(
        boosting_type="gbdt",
        objective="regression",
        metric=["rmse"],
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=80,
        feature_fraction=0.70,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=2.0,
        max_depth=-1,
        verbosity=-1,
        seed=seed,
        num_threads=max(1, os.cpu_count() or 1),
    )
    dtr = lgb.Dataset(Xtr, label=ytr_mag, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva_mag, categorical_feature=cat_idx, free_raw_data=False)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=6000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

def predict_lgb(model, X):
    return model.predict(X, num_iteration=model.best_iteration)


# ---------------------------
# Post-processing / calibration (VAL only tuning)
# ---------------------------
def affine_fit(y_true, y_pred):
    """
    Fit y_true ≈ a*y_pred + b (least squares). Constrain a > 0 to preserve ranking.
    """
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    yp_c = yp - yp.mean()
    yt_c = yt - yt.mean()
    var = np.mean(yp_c * yp_c)
    if var < 1e-12:
        a = 1.0
    else:
        a = float(np.mean(yp_c * yt_c) / var)
        if a <= 0:
            a = 1.0
    b = float(yt.mean() - a * yp.mean())
    return a, b

def apply_policy(pred_mean, pred_q20, pred_moe, pneg, pzero,
                 w_aw, w_moe, w_q,
                 thr_gate, thr_zero, use_qmin):
    y = (w_aw * pred_mean + w_moe * pred_moe + w_q * pred_q20).astype(np.float32)

    if use_qmin:
        y = np.minimum(y, pred_q20.astype(np.float32))

    # Negative risk gate
    if thr_gate is not None:
        y = np.where(pneg > thr_gate, np.minimum(y, 0.0), y)

    # Zero inflation clamp (only clamp positives to 0)
    if thr_zero is not None:
        y = np.where((pzero > thr_zero) & (y > 0), 0.0, y)

    return y

def tune_on_val(yva, pred_mean_va, pred_q20_va, pred_moe_va, pneg_va, pzero_va):
    # Small grid to avoid overfitting/noisy logs.
    w_q_grid = [0.00, 0.05, 0.10, 0.15]
    w_aw_grid = [0.35, 0.50, 0.65]
    thr_gate_grid = [0.50, 0.60, 0.70, 0.80]
    thr_zero_grid = [None, 0.70, 0.80, 0.90]
    use_qmin_grid = [True, False]

    best = (-1e9, None)

    import contextlib
    import io

    for use_qmin in use_qmin_grid:
        for thr_gate in thr_gate_grid:
            for thr_zero in thr_zero_grid:
                for w_q in w_q_grid:
                    for w_aw in w_aw_grid:
                        w_moe = 1.0 - w_aw - w_q
                        if w_moe < 0:
                            continue

                        y_raw = apply_policy(
                            pred_mean_va, pred_q20_va, pred_moe_va, pneg_va, pzero_va,
                            w_aw=w_aw, w_moe=w_moe, w_q=w_q,
                            thr_gate=thr_gate, thr_zero=thr_zero, use_qmin=use_qmin
                        )

                        a, b = affine_fit(yva, y_raw)
                        y_cal = a * y_raw + b

                        # suppress repeated prints from compute_score during tuning
                        with contextlib.redirect_stdout(io.StringIO()):
                            s = compute_pareto_multi_objective(yva, y_cal)

                        if s > best[0]:
                            best = (s, dict(
                                use_qmin=use_qmin,
                                thr_gate=float(thr_gate),
                                thr_zero=None if thr_zero is None else float(thr_zero),
                                w_aw=float(w_aw),
                                w_q=float(w_q),
                                w_moe=float(w_moe),
                                a=float(a),
                                b=float(b),
                            ))

    return best


# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(42)

    print("=" * 70)
    print("Improved Method: LightGBM temporal wide+stats + TE + hurdle auxiliaries")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    train_df = ensure_day_col(train_df)
    val_df   = ensure_day_col(val_df)
    test_df  = ensure_day_col(test_df)

    num_cols, cat_cols = infer_columns(train_df)
    print(f"Detected num_cols={len(num_cols)}, cat_cols={len(cat_cols)}")

    # user ids
    tr_users = train_df[ID_COL].drop_duplicates().values
    va_users = val_df[ID_COL].drop_duplicates().values
    te_users = test_df[ID_COL].drop_duplicates().values

    # targets
    ytr = train_df.groupby(ID_COL)[TARGET_COL].first().reindex(tr_users).values.astype(np.float32)
    yva = val_df.groupby(ID_COL)[TARGET_COL].first().reindex(va_users).values.astype(np.float32)
    yte = test_df.groupby(ID_COL)[TARGET_COL].first().reindex(te_users).values.astype(np.float32)

    # categorical mapping from TRAIN only
    cat_maps = build_cat_maps(train_df, cat_cols)
    Xc_tr, cat_names = encode_user_cats(train_df, tr_users, cat_cols, cat_maps)
    Xc_va, _ = encode_user_cats(val_df,   va_users, cat_cols, cat_maps)
    Xc_te, _ = encode_user_cats(test_df,  te_users, cat_cols, cat_maps)

    # numeric tensors (sign_log1p)
    Xn_tr_3d, m_tr = build_user_numeric_tensor(train_df, tr_users, num_cols, n_days=N_DAYS)
    Xn_va_3d, m_va = build_user_numeric_tensor(val_df,   va_users, num_cols, n_days=N_DAYS)
    Xn_te_3d, m_te = build_user_numeric_tensor(test_df,  te_users, num_cols, n_days=N_DAYS)

    Xn_tr, n_names = numeric_time_features(Xn_tr_3d, m_tr, num_cols, prefix_days=True)
    Xn_va, _ = numeric_time_features(Xn_va_3d, m_va, num_cols, prefix_days=True)
    Xn_te, _ = numeric_time_features(Xn_te_3d, m_te, num_cols, prefix_days=True)

    # add day_count (missing days signal)
    daycnt_tr = m_tr.sum(axis=1).reshape(-1, 1).astype(np.float32)
    daycnt_va = m_va.sum(axis=1).reshape(-1, 1).astype(np.float32)
    daycnt_te = m_te.sum(axis=1).reshape(-1, 1).astype(np.float32)

    # target encoding on TRAIN only (OOF for train)
    te_tr, te_va, te_te, te_names = build_target_encoding_features(
        Xc_tr, ytr, Xc_va, Xc_te, cat_names, seed=42
    )

    # Assemble final matrices:
    # numeric + daycount + target enc + cat codes
    Xtr = np.concatenate([Xn_tr, daycnt_tr, te_tr, Xc_tr.astype(np.float32)], axis=1).astype(np.float32)
    Xva = np.concatenate([Xn_va, daycnt_va, te_va, Xc_va.astype(np.float32)], axis=1).astype(np.float32)
    XteX = np.concatenate([Xn_te, daycnt_te, te_te, Xc_te.astype(np.float32)], axis=1).astype(np.float32)

    feature_names = n_names + ["day_count"] + te_names + [f"{c}__catcode" for c in cat_names]

    # categorical indices correspond to the LAST block (cat codes)
    cat_start = Xtr.shape[1] - Xc_tr.shape[1]
    cat_idx = list(range(cat_start, Xtr.shape[1])) if Xc_tr.shape[1] > 0 else []

    # drop constants (helps stability)
    Xtr, Xva, XteX, feature_names = drop_constant_columns(Xtr, Xva, XteX, feature_names)

    # adjust cat_idx after dropping columns (recompute by name suffix)
    cat_idx = [i for i, n in enumerate(feature_names) if n.endswith("__catcode")]

    print(f"Final feature count = {Xtr.shape[1]} (categorical={len(cat_idx)})")
    print(f"Train/Val/Test users = {len(ytr)}/{len(yva)}/{len(yte)}")

    # Train base models (small ensemble by varying seed)
    print("\n[Training] AWMSE regressors...")
    reg_aw_1 = train_lgb_reg_aw(Xtr, ytr, Xva, yva, cat_idx, seed=42)
    reg_aw_2 = train_lgb_reg_aw(Xtr, ytr, Xva, yva, cat_idx, seed=202)

    print("[Training] Quantile (tau=0.20) regressors...")
    reg_q_1 = train_lgb_quantile(Xtr, ytr, Xva, yva, cat_idx, alpha=0.20, seed=42)
    reg_q_2 = train_lgb_quantile(Xtr, ytr, Xva, yva, cat_idx, alpha=0.20, seed=202)

    print("[Training] Negative classifier p(y<0)...")
    ytr_neg = (ytr < 0).astype(np.int32)
    yva_neg = (yva < 0).astype(np.int32)
    clf_neg = train_lgb_binary(Xtr, ytr_neg, Xva, yva_neg, cat_idx, seed=42)

    print("[Training] Zero classifier p(y==0 | y>=0)...")
    tr_mask_nneg = (ytr >= 0)
    va_mask_nneg = (yva >= 0)
    # guard if rare
    if np.sum(tr_mask_nneg) > 100:
        ytr_zero = (ytr[tr_mask_nneg] == 0).astype(np.int32)
        yva_zero = (yva[va_mask_nneg] == 0).astype(np.int32) if np.sum(va_mask_nneg) > 0 else np.zeros(0, np.int32)
        clf_zero = train_lgb_binary(Xtr[tr_mask_nneg], ytr_zero, Xva[va_mask_nneg], yva_zero, cat_idx, seed=99)
    else:
        clf_zero = None

    print("[Training] Pos/Neg magnitude experts...")
    tr_pos = ytr > 0
    tr_neg = ytr < 0
    va_pos = yva > 0
    va_negm = yva < 0

    # If a side is too small, fallback to global model for that expert
    reg_pos = None
    reg_negm = None
    if np.sum(tr_pos) > 200:
        ytr_pos_mag = np.log1p(ytr[tr_pos].astype(np.float32))
        yva_pos_mag = np.log1p(np.clip(yva[va_pos], 0, None).astype(np.float32)) if np.sum(va_pos) > 0 else np.zeros(0, np.float32)
        reg_pos = train_lgb_mag_reg(Xtr[tr_pos], ytr_pos_mag, Xva[va_pos], yva_pos_mag, cat_idx, seed=7)
    if np.sum(tr_neg) > 200:
        ytr_neg_mag = np.log1p((-ytr[tr_neg]).astype(np.float32))
        yva_neg_mag = np.log1p((-yva[va_negm]).astype(np.float32)) if np.sum(va_negm) > 0 else np.zeros(0, np.float32)
        reg_negm = train_lgb_mag_reg(Xtr[tr_neg], ytr_neg_mag, Xva[va_negm], yva_neg_mag, cat_idx, seed=8)

    # Predictions on VAL
    pred_aw_va = 0.5 * (predict_lgb(reg_aw_1, Xva) + predict_lgb(reg_aw_2, Xva))
    pred_q_va = 0.5 * (predict_lgb(reg_q_1, Xva) + predict_lgb(reg_q_2, Xva))
    pneg_va = np.clip(predict_lgb(clf_neg, Xva), 0, 1).astype(np.float32)

    if clf_zero is not None and np.sum(va_mask_nneg) > 0:
        pzero_va = np.zeros(len(yva), dtype=np.float32)
        pzero_va[va_mask_nneg] = np.clip(predict_lgb(clf_zero, Xva[va_mask_nneg]), 0, 1).astype(np.float32)
    else:
        pzero_va = np.zeros(len(yva), dtype=np.float32)

    # MOE prediction on VAL
    if reg_pos is not None:
        pos_mag_hat = np.expm1(np.maximum(predict_lgb(reg_pos, Xva), 0.0)).astype(np.float32)
    else:
        pos_mag_hat = np.maximum(pred_aw_va, 0.0).astype(np.float32)

    if reg_negm is not None:
        neg_mag_hat = -np.expm1(np.maximum(predict_lgb(reg_negm, Xva), 0.0)).astype(np.float32)
    else:
        neg_mag_hat = np.minimum(pred_aw_va, 0.0).astype(np.float32)

    pred_moe_va = ((1.0 - pneg_va) * pos_mag_hat + pneg_va * neg_mag_hat).astype(np.float32)

    # Clip to reduce catastrophic outliers (helps RMSE, keeps ranking mostly intact)
    lo, hi = np.quantile(ytr, [0.002, 0.998])
    pred_aw_va = np.clip(pred_aw_va, lo, hi).astype(np.float32)
    pred_q_va = np.clip(pred_q_va, lo, hi).astype(np.float32)
    pred_moe_va = np.clip(pred_moe_va, lo, hi).astype(np.float32)

    # Tune policy on VAL only (small grid)
    print("\n[Tuning] Searching conservative policy on VAL (small grid)...")
    best_val_score, best_params = tune_on_val(yva, pred_aw_va, pred_q_va, pred_moe_va, pneg_va, pzero_va)
    print(f"Best VAL policy score = {best_val_score:.6f}")
    print("Best params:", best_params)

    # Predictions on TEST
    pred_aw_te = 0.5 * (predict_lgb(reg_aw_1, XteX) + predict_lgb(reg_aw_2, XteX))
    pred_q_te = 0.5 * (predict_lgb(reg_q_1, XteX) + predict_lgb(reg_q_2, XteX))
    pneg_te = np.clip(predict_lgb(clf_neg, XteX), 0, 1).astype(np.float32)

    if clf_zero is not None:
        pzero_te = np.zeros(len(te_users), dtype=np.float32)
        te_mask_nneg = np.ones(len(te_users), dtype=bool)  # unknown in prod; but here test has labels; don't use them
        # We intentionally do NOT use yte to decide mask; just compute everywhere
        pzero_te = np.clip(predict_lgb(clf_zero, XteX), 0, 1).astype(np.float32)
    else:
        pzero_te = np.zeros(len(te_users), dtype=np.float32)

    # MOE on TEST
    if reg_pos is not None:
        pos_mag_hat_te = np.expm1(np.maximum(predict_lgb(reg_pos, XteX), 0.0)).astype(np.float32)
    else:
        pos_mag_hat_te = np.maximum(pred_aw_te, 0.0).astype(np.float32)

    if reg_negm is not None:
        neg_mag_hat_te = -np.expm1(np.maximum(predict_lgb(reg_negm, XteX), 0.0)).astype(np.float32)
    else:
        neg_mag_hat_te = np.minimum(pred_aw_te, 0.0).astype(np.float32)

    pred_moe_te = ((1.0 - pneg_te) * pos_mag_hat_te + pneg_te * neg_mag_hat_te).astype(np.float32)

    pred_aw_te = np.clip(pred_aw_te, lo, hi).astype(np.float32)
    pred_q_te = np.clip(pred_q_te, lo, hi).astype(np.float32)
    pred_moe_te = np.clip(pred_moe_te, lo, hi).astype(np.float32)

    # Apply tuned policy + affine calibration (a,b from VAL)
    y_raw_te = apply_policy(
        pred_aw_te, pred_q_te, pred_moe_te, pneg_te, pzero_te,
        w_aw=best_params["w_aw"],
        w_moe=best_params["w_moe"],
        w_q=best_params["w_q"],
        thr_gate=best_params["thr_gate"],
        thr_zero=best_params["thr_zero"],
        use_qmin=best_params["use_qmin"],
    )
    final_pred = best_params["a"] * y_raw_te + best_params["b"]

    # Align with test ground truth order
    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(final_pred, index=te_users).reindex(y_test.index).values

    print("\n" + "=" * 70)
    score_val, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 70)
    print(f"\n🎯 Final Score: {score_val:.6f}")

    # Optional save
    out_path = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/improved_lgbm_temporal_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "method": "Improved LightGBM temporal+TE+hurdle",
                "score": float(score_val),
                "metrics": metrics,
                "best_val_policy_score": float(best_val_score),
                "policy": best_params,
                "num_cols": len(num_cols),
                "cat_cols": len(cat_cols),
                "features": int(Xtr.shape[1]),
            },
            f,
            indent=2,
        )

    # MANDATORY: assign to variable named exactly `score` and print
    global score, test_predictions
    test_predictions = pred_aligned
    score = float(score_val)
    print(f"score = {score}")


if __name__ == "__main__":
    main()

# =============================================================================
# SCORE CALCULATION - MANDATORY (for EVALUATION, not training)
# =============================================================================
print(f"score = {score}")  # This will be parsed by the system

# =============================================================================
# SCORE CALCULATION - MANDATORY (for EVALUATION, not training)
# =============================================================================
# NOTE: This is the EVALUATION metric. You can use ANY training loss you prefer.
# But the final score MUST be calculated using this function.
# The variable MUST be named exactly 'score' for the system to read it.
# Using the pareto_multi_objective metric (higher is better)

score = compute_pareto_multi_objective(y_test, test_predictions)
print(f"score = {score}")  # This will be parsed by the system

