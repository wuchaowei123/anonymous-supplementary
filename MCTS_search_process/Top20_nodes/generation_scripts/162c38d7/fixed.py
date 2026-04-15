#!/usr/bin/env python3
"""
AviaAgentMonty - Execution Node: 162c38d7
Type: fixed
Generated: 2026-01-14T00:19:57.123769
Fix Attempts: 1

DO NOT DELETE - This file is preserved for reproducibility.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import contextlib
import io

from sklearn.model_selection import KFold
import lightgbm as lgb

device = "cpu"

TRAIN_PATH = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"
DAY_COL = "TDATE_RN"
N_DAYS = 7

# ============================================================
# !!! CRITICAL: PRESERVE scoring/metric functions EXACTLY !!!
# ============================================================
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
    # Per-sample absolute error rate (MAE normalized by sum of true values)
    error_rate = np.sum(np.abs(y_pred - y_true)) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
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

def compute_pareto_multi_objective(y_true, y_pred):
    s, _ = compute_score(y_true, y_pred)
    return s


# ============================================================
# Utilities / FE
# ============================================================
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

def numeric_time_features_v2(X: np.ndarray, mask: np.ndarray, num_cols, prefix_days=True, add_diffs=True, add_windows=True):
    """
    Extends original numeric_time_features with:
      - consecutive day diffs (flatten + summary stats)
      - window means (first3/last3/last2) and their deltas
    """
    U, T, D = X.shape
    m = mask
    cnt = np.clip(m.sum(axis=1), 1.0, None)
    sum_x = (X * m).sum(axis=1)
    mean = sum_x / cnt

    xc = X - mean[:, None, :]
    var = ((xc * xc) * m).sum(axis=1) / cnt
    std = np.sqrt(np.maximum(var, 0)).astype(np.float32)

    big = np.float32(1e9)
    X_min = np.where(m.astype(bool), X, big).min(axis=1).astype(np.float32)
    X_max = np.where(m.astype(bool), X, -big).max(axis=1).astype(np.float32)

    first = X[:, 0, :].astype(np.float32)
    last = X[:, -1, :].astype(np.float32)
    delta = (last - first).astype(np.float32)

    tvec = np.arange(T, dtype=np.float32)[None, :, None]
    tmean = (tvec * m).sum(axis=1) / cnt
    tc = (tvec - tmean[:, None, :])
    cov = ((tc * xc) * m).sum(axis=1)
    var_t = ((tc * tc) * m).sum(axis=1) + 1e-6
    slope = (cov / var_t).astype(np.float32)

    feats = []
    names = []

    if prefix_days:
        Xflat = X.reshape(U, T * D).astype(np.float32)
        feats.append(Xflat)
        for d in range(T):
            for c in num_cols:
                names.append(f"{c}_d{d+1}")

    feats.extend([mean, std, X_min, X_max, delta, slope])
    for c in num_cols: names.append(f"{c}__mean")
    for c in num_cols: names.append(f"{c}__std")
    for c in num_cols: names.append(f"{c}__min")
    for c in num_cols: names.append(f"{c}__max")
    for c in num_cols: names.append(f"{c}__delta")
    for c in num_cols: names.append(f"{c}__slope")

    if add_diffs and T >= 2:
        dX = X[:, 1:, :] - X[:, :-1, :]
        md = (m[:, 1:, :] * m[:, :-1, :]).astype(np.float32)
        dcnt = np.clip(md.sum(axis=1), 1.0, None)
        dmean = (dX * md).sum(axis=1) / dcnt
        dvar = (((dX - dmean[:, None, :]) ** 2) * md).sum(axis=1) / dcnt
        dstd = np.sqrt(np.maximum(dvar, 0)).astype(np.float32)
        dmax = np.where(md.astype(bool), dX, -big).max(axis=1).astype(np.float32)
        dmin = np.where(md.astype(bool), dX, big).min(axis=1).astype(np.float32)

        feats.extend([dmean, dstd, dmin, dmax])
        for c in num_cols: names.append(f"{c}__diff_mean")
        for c in num_cols: names.append(f"{c}__diff_std")
        for c in num_cols: names.append(f"{c}__diff_min")
        for c in num_cols: names.append(f"{c}__diff_max")

        dflat = dX.reshape(U, (T - 1) * D).astype(np.float32)
        feats.append(dflat)
        for d in range(T - 1):
            for c in num_cols:
                names.append(f"{c}__diff_d{d+2}_minus_d{d+1}")

    if add_windows and T >= 3:
        def window_mean(Xw, mw):
            c = np.clip(mw.sum(axis=1), 1.0, None)
            return (Xw * mw).sum(axis=1) / c

        first3 = window_mean(X[:, :3, :], m[:, :3, :])
        last3  = window_mean(X[:, -3:, :], m[:, -3:, :])
        last2  = window_mean(X[:, -2:, :], m[:, -2:, :])
        feats.extend([first3, last3, last2, (last3 - first3).astype(np.float32), (last2 - first3).astype(np.float32)])
        for c in num_cols: names.append(f"{c}__first3_mean")
        for c in num_cols: names.append(f"{c}__last3_mean")
        for c in num_cols: names.append(f"{c}__last2_mean")
        for c in num_cols: names.append(f"{c}__last3_minus_first3")
        for c in num_cols: names.append(f"{c}__last2_minus_first3")

    X_num = np.concatenate(feats, axis=1).astype(np.float32)
    return X_num, names

def drop_constant_columns(Xtr, Xva, Xte, names):
    v = Xtr.var(axis=0)
    keep = v > 1e-12
    Xtr2 = Xtr[:, keep]
    Xva2 = Xva[:, keep]
    Xte2 = Xte[:, keep]
    names2 = [n for n, k in zip(names, keep) if k]
    return Xtr2, Xva2, Xte2, names2

def oof_target_encode_single(cat_train: np.ndarray, y_train: np.ndarray, cat_other: np.ndarray,
                             n_splits=5, alpha=30.0, seed=42):
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

    stats_full = pd.DataFrame({"c": cat_train, "y": y_train.astype(np.float32)}).groupby("c")["y"].agg(["mean", "count"])
    smooth_full = (stats_full["mean"] * stats_full["count"] + global_mean * alpha) / (stats_full["count"] + alpha)
    mapping_full = smooth_full.to_dict()

    other = pd.Series(cat_other).map(mapping_full).fillna(global_mean).values.astype(np.float32)
    return oof, other

def build_target_encoding_features_v2(Xc_tr, ytr, Xc_va, Xc_te, cat_names, seed=42):
    """
    Adds richer TE signals for signed/zero-inflated target:
      - y mean
      - P(neg), P(zero), P(pos)
      - expected positive value (y_pos)
      - expected negative value (y_neg) (negative, <=0)
      - log1p(abs(y)) mean (magnitude)
    """
    if Xc_tr.shape[1] == 0:
        ztr = np.zeros((len(ytr), 0), np.float32)
        zva = np.zeros((len(Xc_va), 0), np.float32)
        zte = np.zeros((len(Xc_te), 0), np.float32)
        return ztr, zva, zte, []

    te_tr_list, te_va_list, te_te_list, te_names = [], [], [], []

    y = ytr.astype(np.float32)
    y_neg_ind = (ytr < 0).astype(np.float32)
    y_zero_ind = (ytr == 0).astype(np.float32)
    y_pos_ind = (ytr > 0).astype(np.float32)
    y_pos_val = np.clip(ytr, 0, None).astype(np.float32)
    y_neg_val = np.clip(ytr, None, 0).astype(np.float32)
    y_logabs  = np.log1p(np.abs(ytr)).astype(np.float32)

    for j, cname in enumerate(cat_names):
        c_tr = Xc_tr[:, j]
        c_va = Xc_va[:, j]
        c_te = Xc_te[:, j]

        oof_mean, va_mean = oof_target_encode_single(c_tr, y, c_va, n_splits=5, alpha=25.0, seed=seed)
        _, te_mean = oof_target_encode_single(c_tr, y, c_te, n_splits=5, alpha=25.0, seed=seed)

        oof_neg, va_neg = oof_target_encode_single(c_tr, y_neg_ind, c_va, n_splits=5, alpha=60.0, seed=seed)
        _, te_neg = oof_target_encode_single(c_tr, y_neg_ind, c_te, n_splits=5, alpha=60.0, seed=seed)

        oof_zero, va_zero = oof_target_encode_single(c_tr, y_zero_ind, c_va, n_splits=5, alpha=60.0, seed=seed)
        _, te_zero = oof_target_encode_single(c_tr, y_zero_ind, c_te, n_splits=5, alpha=60.0, seed=seed)

        oof_pos, va_pos = oof_target_encode_single(c_tr, y_pos_ind, c_va, n_splits=5, alpha=60.0, seed=seed)
        _, te_pos = oof_target_encode_single(c_tr, y_pos_ind, c_te, n_splits=5, alpha=60.0, seed=seed)

        oof_ypos, va_ypos = oof_target_encode_single(c_tr, y_pos_val, c_va, n_splits=5, alpha=40.0, seed=seed)
        _, te_ypos = oof_target_encode_single(c_tr, y_pos_val, c_te, n_splits=5, alpha=40.0, seed=seed)

        oof_yneg, va_yneg = oof_target_encode_single(c_tr, y_neg_val, c_va, n_splits=5, alpha=40.0, seed=seed)
        _, te_yneg = oof_target_encode_single(c_tr, y_neg_val, c_te, n_splits=5, alpha=40.0, seed=seed)

        oof_logabs, va_logabs = oof_target_encode_single(c_tr, y_logabs, c_va, n_splits=5, alpha=40.0, seed=seed)
        _, te_logabs = oof_target_encode_single(c_tr, y_logabs, c_te, n_splits=5, alpha=40.0, seed=seed)

        te_tr_list += [
            oof_mean[:, None], oof_neg[:, None], oof_zero[:, None], oof_pos[:, None],
            oof_ypos[:, None], oof_yneg[:, None], oof_logabs[:, None],
        ]
        te_va_list += [
            va_mean[:, None], va_neg[:, None], va_zero[:, None], va_pos[:, None],
            va_ypos[:, None], va_yneg[:, None], va_logabs[:, None],
        ]
        te_te_list += [
            te_mean[:, None], te_neg[:, None], te_zero[:, None], te_pos[:, None],
            te_ypos[:, None], te_yneg[:, None], te_logabs[:, None],
        ]
        te_names += [
            f"{cname}__te_ymean", f"{cname}__te_pneg", f"{cname}__te_pzero", f"{cname}__te_ppos",
            f"{cname}__te_ypos", f"{cname}__te_yneg", f"{cname}__te_logabs",
        ]

    te_tr = np.concatenate(te_tr_list, axis=1).astype(np.float32)
    te_va = np.concatenate(te_va_list, axis=1).astype(np.float32)
    te_te = np.concatenate(te_te_list, axis=1).astype(np.float32)
    return te_tr, te_va, te_te, te_names

def build_cat_count_features(train_df: pd.DataFrame, dfs, cat_cols):
    """
    Frequency encoding: log1p(count(category)) per user (take user's first value).
    """
    if not cat_cols:
        outs = []
        for df in dfs:
            outs.append((np.zeros((df[ID_COL].nunique(), 0), np.float32), []))
        return outs

    counts = {}
    for c in cat_cols:
        vc = train_df[c].fillna("UNK").astype(str).value_counts(dropna=False)
        counts[c] = vc

    names = [f"{c}__logcnt" for c in cat_cols]
    outs = []
    for df in dfs:
        users = df[ID_COL].drop_duplicates().values
        df2 = ensure_day_col(df).sort_values([ID_COL, DAY_COL]).drop_duplicates([ID_COL, DAY_COL], keep="last")
        first = df2.groupby(ID_COL)[cat_cols].first().reindex(users)
        X = np.zeros((len(users), len(cat_cols)), dtype=np.float32)
        for j, c in enumerate(cat_cols):
            vals = first[c].fillna("UNK").astype(str).values
            vc = counts[c]
            X[:, j] = np.log1p(np.array([vc.get(v, 0) for v in vals], dtype=np.float32))
        outs.append((X, names))
    return outs


# ============================================================
# LightGBM models
# ============================================================
def lgb_aw_obj(preds, train_data):
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

def _common_lgb_params(seed: int):
    return dict(
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=256,
        min_data_in_leaf=50,
        feature_fraction=0.70,
        feature_fraction_bynode=0.80,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=2.0,
        lambda_l1=0.0,
        min_gain_to_split=0.0,
        max_depth=-1,
        max_bin=255,
        verbosity=-1,
        seed=seed,
        num_threads=max(1, os.cpu_count() or 1),
        extra_trees=True,
    )

def train_lgb_reg_aw(Xtr, ytr, Xva, yva, cat_idx, seed=42):
    params = _common_lgb_params(seed)
    params.update(dict(
        objective="regression",
        metric=["rmse"],
    ))
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_idx, free_raw_data=False)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=7000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

def train_lgb_huber(Xtr, ytr, Xva, yva, cat_idx, seed=42):
    params = _common_lgb_params(seed)
    params.update(dict(
        objective="huber",
        alpha=1.0,
        metric=["rmse"],
        num_leaves=192,
        min_data_in_leaf=70,
        feature_fraction=0.72,
    ))
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

def train_lgb_quantile(Xtr, ytr, Xva, yva, cat_idx, alpha=0.20, seed=42):
    params = _common_lgb_params(seed)
    params.update(dict(
        objective="quantile",
        alpha=alpha,
        metric=["quantile"],
        num_leaves=160,
        min_data_in_leaf=80,
        feature_fraction=0.75,
        extra_trees=False,
    ))
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

    params = _common_lgb_params(seed)
    params.update(dict(
        objective="binary",
        metric=["auc", "binary_logloss"],
        num_leaves=128,
        min_data_in_leaf=120,
        feature_fraction=0.80,
        scale_pos_weight=scale_pos_weight,
        extra_trees=True,
    ))
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
    params = _common_lgb_params(seed)
    params.update(dict(
        objective="regression_l2",
        metric=["rmse"],
        num_leaves=160,
        min_data_in_leaf=90,
        feature_fraction=0.75,
        extra_trees=True,
    ))
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

def affine_fit(y_true, y_pred):
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

def mean_correct(pred: np.ndarray, anchor_mean: float) -> np.ndarray:
    pred = pred.astype(np.float32, copy=False)
    return (pred + (np.float32(anchor_mean) - np.float32(pred.mean()))).astype(np.float32)

def apply_policy_v2(pred_aw, pred_huber, pred_q20, pred_q80, pred_moe, pneg, pzero,
                    w_aw, w_huber, w_moe, w_q,
                    use_qmin, thr_gate, thr_zero, snap_eps, risk_scale):
    """
    Conservative mixture + risk shaping:
      - blend multiple regressors
      - optionally min with q20 lower bound
      - gate to <=0 for high negative-risk
      - snap small magnitudes to 0 (helps zero inflation)
      - optional risk_scale to damp positive predictions when pneg high
    """
    y = (w_aw * pred_aw + w_huber * pred_huber + w_moe * pred_moe + w_q * pred_q20).astype(np.float32)

    if use_qmin:
        y = np.minimum(y, pred_q20.astype(np.float32))

    if risk_scale > 0:
        gap = np.maximum(y - pred_q20.astype(np.float32), 0.0)
        y = y - (np.float32(risk_scale) * pneg.astype(np.float32) * gap)

    if thr_gate is not None:
        y = np.where(pneg > np.float32(thr_gate), np.minimum(y, 0.0), y)

    if thr_zero is not None:
        y = np.where((pzero > np.float32(thr_zero)) & (y > 0), 0.0, y)

    if snap_eps is not None and snap_eps > 0:
        y = np.where(np.abs(y) < np.float32(snap_eps), 0.0, y)

    y = np.where((pred_q80 < 0) & (y > 0), np.minimum(y, 0.0), y)

    return y.astype(np.float32)

def tune_on_val_v2(yva, pred_aw_va, pred_huber_va, pred_q20_va, pred_q80_va, pred_moe_va, pneg_va, pzero_va, anchor_mean: float):
    w_q_grid = [0.00, 0.05, 0.10, 0.15]
    w_aw_grid = [0.35, 0.50, 0.65]
    w_huber_grid = [0.00, 0.10, 0.20]
    thr_gate_grid = [0.45, 0.55, 0.65, 0.75]
    thr_zero_grid = [None, 0.75, 0.85, 0.92]
    use_qmin_grid = [True, False]
    snap_eps_grid = [0.0, 0.5, 1.0, 2.0]
    risk_scale_grid = [0.0, 0.2, 0.4]

    best = (-1e9, None)

    for use_qmin in use_qmin_grid:
        for thr_gate in thr_gate_grid:
            for thr_zero in thr_zero_grid:
                for snap_eps in snap_eps_grid:
                    for risk_scale in risk_scale_grid:
                        for w_q in w_q_grid:
                            for w_aw in w_aw_grid:
                                for w_huber in w_huber_grid:
                                    w_moe = 1.0 - w_aw - w_q - w_huber
                                    if w_moe < 0:
                                        continue

                                    y_raw = apply_policy_v2(
                                        pred_aw_va, pred_huber_va, pred_q20_va, pred_q80_va, pred_moe_va, pneg_va, pzero_va,
                                        w_aw=w_aw, w_huber=w_huber, w_moe=w_moe, w_q=w_q,
                                        use_qmin=use_qmin, thr_gate=thr_gate, thr_zero=thr_zero,
                                        snap_eps=snap_eps, risk_scale=risk_scale
                                    )

                                    a, b = affine_fit(yva, y_raw)
                                    y_cal = (a * y_raw + b).astype(np.float32)

                                    y_cal = mean_correct(y_cal, anchor_mean=anchor_mean)

                                    with contextlib.redirect_stdout(io.StringIO()):
                                        s = compute_pareto_multi_objective(yva, y_cal)

                                    if s > best[0]:
                                        best = (s, dict(
                                            use_qmin=bool(use_qmin),
                                            thr_gate=float(thr_gate),
                                            thr_zero=None if thr_zero is None else float(thr_zero),
                                            snap_eps=float(snap_eps),
                                            risk_scale=float(risk_scale),
                                            w_aw=float(w_aw),
                                            w_huber=float(w_huber),
                                            w_q=float(w_q),
                                            w_moe=float(w_moe),
                                            a=float(a),
                                            b=float(b),
                                        ))
    return best


# ============================================================
# Main pipeline
# ============================================================
def main():
    set_seed(42)

    print("=" * 70)
    print("Improved Method v2: Custom AWMSE LGB + richer temporal FE + q20/q80 + robust mean anchoring")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    train_df = ensure_day_col(train_df)
    val_df   = ensure_day_col(val_df)
    test_df  = ensure_day_col(test_df)

    num_cols, cat_cols = infer_columns(train_df)
    print(f"Detected num_cols={len(num_cols)}, cat_cols={len(cat_cols)}")

    tr_users = train_df[ID_COL].drop_duplicates().values
    va_users = val_df[ID_COL].drop_duplicates().values
    te_users = test_df[ID_COL].drop_duplicates().values

    ytr = train_df.groupby(ID_COL)[TARGET_COL].first().reindex(tr_users).values.astype(np.float32)
    yva = val_df.groupby(ID_COL)[TARGET_COL].first().reindex(va_users).values.astype(np.float32)
    yte = test_df.groupby(ID_COL)[TARGET_COL].first().reindex(te_users).values.astype(np.float32)

    anchor_mean = float(np.mean(np.concatenate([ytr, yva], axis=0)))
    print(f"Anchor mean (train+val) = {anchor_mean:.6f} | train mean={float(ytr.mean()):.6f} | val mean={float(yva.mean()):.6f}")

    cat_maps = build_cat_maps(train_df, cat_cols)
    Xc_tr, cat_names = encode_user_cats(train_df, tr_users, cat_cols, cat_maps)
    Xc_va, _ = encode_user_cats(val_df,   va_users, cat_cols, cat_maps)
    Xc_te, _ = encode_user_cats(test_df,  te_users, cat_cols, cat_maps)

    Xn_tr_3d, m_tr = build_user_numeric_tensor(train_df, tr_users, num_cols, n_days=N_DAYS)
    Xn_va_3d, m_va = build_user_numeric_tensor(val_df,   va_users, num_cols, n_days=N_DAYS)
    Xn_te_3d, m_te = build_user_numeric_tensor(test_df,  te_users, num_cols, n_days=N_DAYS)

    Xn_tr, n_names = numeric_time_features_v2(Xn_tr_3d, m_tr, num_cols, prefix_days=True, add_diffs=True, add_windows=True)
    Xn_va, _       = numeric_time_features_v2(Xn_va_3d, m_va, num_cols, prefix_days=True, add_diffs=True, add_windows=True)
    Xn_te, _       = numeric_time_features_v2(Xn_te_3d, m_te, num_cols, prefix_days=True, add_diffs=True, add_windows=True)

    daycnt_tr = m_tr.sum(axis=1).reshape(-1, 1).astype(np.float32)
    daycnt_va = m_va.sum(axis=1).reshape(-1, 1).astype(np.float32)
    daycnt_te = m_te.sum(axis=1).reshape(-1, 1).astype(np.float32)

    te_tr, te_va, te_te, te_names = build_target_encoding_features_v2(
        Xc_tr, ytr, Xc_va, Xc_te, cat_names, seed=42
    )

    (cnt_tr, cnt_names), (cnt_va, _), (cnt_te, _) = build_cat_count_features(train_df, [train_df, val_df, test_df], cat_cols)

    Xtr = np.concatenate([Xn_tr, daycnt_tr, te_tr, cnt_tr, Xc_tr.astype(np.float32)], axis=1).astype(np.float32)
    Xva = np.concatenate([Xn_va, daycnt_va, te_va, cnt_va, Xc_va.astype(np.float32)], axis=1).astype(np.float32)
    XteX = np.concatenate([Xn_te, daycnt_te, te_te, cnt_te, Xc_te.astype(np.float32)], axis=1).astype(np.float32)

    feature_names = (
        n_names + ["day_count"] + te_names + cnt_names + [f"{c}__catcode" for c in cat_names]
    )

    cat_start = Xtr.shape[1] - Xc_tr.shape[1]
    cat_idx = list(range(cat_start, Xtr.shape[1])) if Xc_tr.shape[1] > 0 else []

    Xtr, Xva, XteX, feature_names = drop_constant_columns(Xtr, Xva, XteX, feature_names)
    cat_idx = [i for i, n in enumerate(feature_names) if n.endswith("__catcode")]

    print(f"Final feature count = {Xtr.shape[1]} (categorical={len(cat_idx)})")
    print(f"Train/Val/Test users = {len(ytr)}/{len(yva)}/{len(yte)}")

    lo, hi = np.quantile(ytr, [0.001, 0.999])
    lo, hi = float(lo), float(hi)

    print("\n[Training] AWMSE regressors (standard regression)...")
    reg_aw_1 = train_lgb_reg_aw(Xtr, ytr, Xva, yva, cat_idx, seed=42)
    reg_aw_2 = train_lgb_reg_aw(Xtr, ytr, Xva, yva, cat_idx, seed=202)
    reg_aw_3 = train_lgb_reg_aw(Xtr, ytr, Xva, yva, cat_idx, seed=777)

    print("[Training] Huber regressor (robust, diversity)...")
    reg_huber = train_lgb_huber(Xtr, ytr, Xva, yva, cat_idx, seed=1337)

    print("[Training] Quantile regressors (q20 and q80)...")
    reg_q20_1 = train_lgb_quantile(Xtr, ytr, Xva, yva, cat_idx, alpha=0.20, seed=42)
    reg_q20_2 = train_lgb_quantile(Xtr, ytr, Xva, yva, cat_idx, alpha=0.20, seed=202)
    reg_q80   = train_lgb_quantile(Xtr, ytr, Xva, yva, cat_idx, alpha=0.80, seed=909)

    print("[Training] Negative classifier p(y<0)...")
    ytr_neg = (ytr < 0).astype(np.int32)
    yva_neg = (yva < 0).astype(np.int32)
    clf_neg = train_lgb_binary(Xtr, ytr_neg, Xva, yva_neg, cat_idx, seed=42)

    print("[Training] Zero classifier p(y==0 | y>=0)...")
    tr_mask_nneg = (ytr >= 0)
    va_mask_nneg = (yva >= 0)
    if np.sum(tr_mask_nneg) > 100 and np.sum(va_mask_nneg) > 0:
        ytr_zero = (ytr[tr_mask_nneg] == 0).astype(np.int32)
        yva_zero = (yva[va_mask_nneg] == 0).astype(np.int32)
        clf_zero = train_lgb_binary(Xtr[tr_mask_nneg], ytr_zero, Xva[va_mask_nneg], yva_zero, cat_idx, seed=99)
    else:
        clf_zero = None

    print("[Training] Pos/Neg magnitude experts...")
    tr_pos = ytr > 0
    tr_negm = ytr < 0
    va_pos = yva > 0
    va_negm = yva < 0

    reg_pos = None
    reg_negm = None
    if np.sum(tr_pos) > 200 and np.sum(va_pos) > 0:
        ytr_pos_mag = np.log1p(ytr[tr_pos].astype(np.float32))
        yva_pos_mag = np.log1p(np.clip(yva[va_pos], 0, None).astype(np.float32))
        reg_pos = train_lgb_mag_reg(Xtr[tr_pos], ytr_pos_mag, Xva[va_pos], yva_pos_mag, cat_idx, seed=7)

    if np.sum(tr_negm) > 200 and np.sum(va_negm) > 0:
        ytr_neg_mag = np.log1p((-ytr[tr_negm]).astype(np.float32))
        yva_neg_mag = np.log1p((-yva[va_negm]).astype(np.float32))
        reg_negm = train_lgb_mag_reg(Xtr[tr_negm], ytr_neg_mag, Xva[va_negm], yva_neg_mag, cat_idx, seed=8)

    pred_aw_va = (predict_lgb(reg_aw_1, Xva) + predict_lgb(reg_aw_2, Xva) + predict_lgb(reg_aw_3, Xva)) / 3.0
    pred_huber_va = predict_lgb(reg_huber, Xva)
    pred_q20_va = 0.5 * (predict_lgb(reg_q20_1, Xva) + predict_lgb(reg_q20_2, Xva))
    pred_q80_va = predict_lgb(reg_q80, Xva)

    pneg_va = np.clip(predict_lgb(clf_neg, Xva), 0, 1).astype(np.float32)

    if clf_zero is not None and np.sum(va_mask_nneg) > 0:
        pzero_va = np.zeros(len(yva), dtype=np.float32)
        pzero_va[va_mask_nneg] = np.clip(predict_lgb(clf_zero, Xva[va_mask_nneg]), 0, 1).astype(np.float32)
        pzero_va = (pzero_va * (1.0 - pneg_va)).astype(np.float32)
    else:
        pzero_va = np.zeros(len(yva), dtype=np.float32)

    if reg_pos is not None:
        pos_mag_hat_va = np.expm1(np.maximum(predict_lgb(reg_pos, Xva), 0.0)).astype(np.float32)
    else:
        pos_mag_hat_va = np.maximum(pred_aw_va, 0.0).astype(np.float32)

    if reg_negm is not None:
        neg_mag_hat_va = -np.expm1(np.maximum(predict_lgb(reg_negm, Xva), 0.0)).astype(np.float32)
    else:
        neg_mag_hat_va = np.minimum(pred_aw_va, 0.0).astype(np.float32)

    pred_moe_va = ((1.0 - pneg_va) * pos_mag_hat_va + pneg_va * neg_mag_hat_va).astype(np.float32)

    pred_aw_va = np.clip(pred_aw_va, lo, hi).astype(np.float32)
    pred_huber_va = np.clip(pred_huber_va, lo, hi).astype(np.float32)
    pred_q20_va = np.clip(pred_q20_va, lo, hi).astype(np.float32)
    pred_q80_va = np.clip(pred_q80_va, lo, hi).astype(np.float32)
    pred_moe_va = np.clip(pred_moe_va, lo, hi).astype(np.float32)

    print("\n[Tuning] Searching conservative policy on VAL (expanded but compact grid)...")
    best_val_score, best_params = tune_on_val_v2(
        yva=yva,
        pred_aw_va=pred_aw_va,
        pred_huber_va=pred_huber_va,
        pred_q20_va=pred_q20_va,
        pred_q80_va=pred_q80_va,
        pred_moe_va=pred_moe_va,
        pneg_va=pneg_va,
        pzero_va=pzero_va,
        anchor_mean=anchor_mean
    )
    print(f"Best VAL policy score = {best_val_score:.6f}")
    print("Best params:", best_params)

    pred_aw_te = (predict_lgb(reg_aw_1, XteX) + predict_lgb(reg_aw_2, XteX) + predict_lgb(reg_aw_3, XteX)) / 3.0
    pred_huber_te = predict_lgb(reg_huber, XteX)
    pred_q20_te = 0.5 * (predict_lgb(reg_q20_1, XteX) + predict_lgb(reg_q20_2, XteX))
    pred_q80_te = predict_lgb(reg_q80, XteX)

    pneg_te = np.clip(predict_lgb(clf_neg, XteX), 0, 1).astype(np.float32)

    if clf_zero is not None:
        pzero_te = np.clip(predict_lgb(clf_zero, XteX), 0, 1).astype(np.float32)
        pzero_te = (pzero_te * (1.0 - pneg_te)).astype(np.float32)
    else:
        pzero_te = np.zeros(len(te_users), dtype=np.float32)

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
    pred_huber_te = np.clip(pred_huber_te, lo, hi).astype(np.float32)
    pred_q20_te = np.clip(pred_q20_te, lo, hi).astype(np.float32)
    pred_q80_te = np.clip(pred_q80_te, lo, hi).astype(np.float32)
    pred_moe_te = np.clip(pred_moe_te, lo, hi).astype(np.float32)

    y_raw_te = apply_policy_v2(
        pred_aw=pred_aw_te,
        pred_huber=pred_huber_te,
        pred_q20=pred_q20_te,
        pred_q80=pred_q80_te,
        pred_moe=pred_moe_te,
        pneg=pneg_te,
        pzero=pzero_te,
        w_aw=best_params["w_aw"],
        w_huber=best_params["w_huber"],
        w_moe=best_params["w_moe"],
        w_q=best_params["w_q"],
        use_qmin=best_params["use_qmin"],
        thr_gate=best_params["thr_gate"],
        thr_zero=best_params["thr_zero"],
        snap_eps=best_params["snap_eps"],
        risk_scale=best_params["risk_scale"],
    )

    final_pred = (best_params["a"] * y_raw_te + best_params["b"]).astype(np.float32)

    final_pred = mean_correct(final_pred, anchor_mean=anchor_mean)

    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(final_pred, index=te_users).reindex(y_test.index).values

    print("\n" + "=" * 70)
    score_val, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 70)
    print(f"\n🎯 Final Score: {score_val:.6f}")

    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/improved_lgbm_temporal_results_v2.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "method": "Improved v2: custom AWMSE + richer temporal + q20/q80 + mean anchoring",
                "score": float(score_val),
                "metrics": metrics,
                "best_val_policy_score": float(best_val_score),
                "policy": best_params,
                "num_cols": int(len(num_cols)),
                "cat_cols": int(len(cat_cols)),
                "features": int(Xtr.shape[1]),
                "anchor_mean": float(anchor_mean),
                "clip_lo": float(lo),
                "clip_hi": float(hi),
            },
            f,
            indent=2,
        )

    # === SAVE MODEL ===
    node_dir = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_20260112_102800/162c38d7"
    os.makedirs(node_dir, exist_ok=True)
    
    # Try to save model if it exists
    try:
        if 'model' in dir() and hasattr(model, 'state_dict'):
            model_path = os.path.join(node_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved to: {model_path}")
        elif 'lgb_model' in dir():
            import joblib
            model_path = os.path.join(node_dir, "model.joblib")
            joblib.dump(lgb_model, model_path)
            print(f"✅ Model saved to: {model_path}")
        elif 'best_model' in dir():
            import joblib
            model_path = os.path.join(node_dir, "model.joblib")
            joblib.dump(best_model, model_path)
            print(f"✅ Model saved to: {model_path}")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")

    # === SAVE PREDICTIONS ===
    try:
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'DEVICE_ID': y_test.index if hasattr(y_test, 'index') else range(len(y_test)),
            'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
            'y_pred': pred_aligned
        })
        predictions_path = os.path.join(node_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✅ Predictions saved to: {predictions_path}")
    except Exception as e:
        print(f"⚠️ Could not save predictions: {e}")


    # ========== AUTO-ADDED SAVING CODE ==========
    try:
        from pathlib import Path
        import joblib
        save_dir = Path("/home/jupyter/saved_models/162c38d7")
        save_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        
        # Save all LightGBM models
        try:
            for name in dir():
                obj = eval(name)
                if isinstance(obj, lgb.Booster):
                    obj.save_model(str(save_dir / f"{name}.txt"))
                    saved.append(name)
                    print(f"✅ Saved {name}")
        except: pass
        
        # Save predictions
        for pn in ['pred_aligned', 'final_pred', 'test_predictions', 'y_pred']:
            if pn in dir():
                for tn in ['y_test', 'yte', 'y_true']:
                    if tn in dir():
                        for un in ['te_users', 'test_users']:
                            if un in dir():
                                pd.DataFrame({
                                    'user_id': eval(un),
                                    'y_pred': eval(pn),
                                    'y_true': eval(tn)
                                }).to_csv(save_dir / "predictions.csv", index=False)
                                print(f"✅ Saved predictions")
                                saved.append('predictions')
                                break
                            break
                        break
                break
        
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump({'node_id': '162c38d7', 'saved': saved}, f)
        print(f"🎉 Saved to {save_dir}")
    except Exception as e:
        print(f"⚠️ Save error: {e}")
    # ========== END SAVING CODE ==========

    return score_val, pred_aligned, y_test.values


if __name__ == "__main__":
    score, test_predictions, y_test = main()

# ========== CORRECTED SAVING CODE ==========
    # main() already executed and returned (score, y_test_values, test_predictions) 
    # or similar - capture them!
    import joblib
    from pathlib import Path
    
    save_dir = Path(f"/home/jupyter/saved_models_final/162c38d7")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions using the return values
    try:
        import pandas as pd
        pred_df = pd.DataFrame({
            'user_id': list(range(len(test_predictions))),
            'y_pred': test_predictions,
            'y_true': y_test_values
        })
        pred_df.to_csv(save_dir / "predictions.csv", index=False)
        print(f"✅✅✅ Saved predictions to {save_dir}/predictions.csv")
    except Exception as e:
        print(f"⚠️ Prediction save failed: {e}")
# ========== END CORRECTED SAVING CODE ==========

    print(f"score = {score}")

print(f"score = {score}")