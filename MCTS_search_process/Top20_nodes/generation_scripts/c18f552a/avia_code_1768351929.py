import warnings
warnings.filterwarnings("ignore")

import os
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import contextlib
import io

from sklearn.model_selection import KFold

import lightgbm as lgb

# -----------------------------
# Paths / constants
# -----------------------------
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

def standardize_3d(train_X: np.ndarray, train_mask: np.ndarray, X: np.ndarray, mask: np.ndarray, eps=1e-6):
    """
    Compute per-feature mean/std on train (masked), apply to any split.
    """
    m = train_mask.astype(np.float64)  # (U,T,1)
    Xtr = train_X.astype(np.float64)
    cnt = np.clip(m.sum(axis=(0, 1)), 1.0, None)  # (1,)
    mu = (Xtr * m).sum(axis=(0, 1)) / cnt  # (D,)
    var = ((Xtr - mu) ** 2 * m).sum(axis=(0, 1)) / cnt
    std = np.sqrt(np.maximum(var, eps))
    Xn = (X - mu[None, None, :]) / std[None, None, :]
    Xn = Xn * mask
    return Xn.astype(np.float32), mu.astype(np.float32), std.astype(np.float32)

def masked_mean_np(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cnt = np.clip(mask.sum(axis=1), 1.0, None)  # (U,1)
    return (X * mask).sum(axis=1) / cnt

def masked_std_np(X: np.ndarray, mask: np.ndarray, eps=1e-6) -> np.ndarray:
    mu = masked_mean_np(X, mask)
    cnt = np.clip(mask.sum(axis=1), 1.0, None)
    var = ((X - mu[:, None, :]) ** 2 * mask).sum(axis=1) / cnt
    return np.sqrt(np.maximum(var, eps)).astype(np.float32)

def masked_minmax_np(X: np.ndarray, mask: np.ndarray):
    big = np.float32(1e6)
    Xmin = np.min(np.where(mask > 0, X, big), axis=1)
    Xmax = np.max(np.where(mask > 0, X, -big), axis=1)
    # if a user had no days (shouldn't), fix:
    Xmin = np.where(np.isfinite(Xmin), Xmin, 0.0).astype(np.float32)
    Xmax = np.where(np.isfinite(Xmax), Xmax, 0.0).astype(np.float32)
    return Xmin, Xmax

def masked_first_last_np(X: np.ndarray, mask: np.ndarray):
    # T is tiny (7) -> simple loop is fine and robust
    U, T, D = X.shape
    first = np.zeros((U, D), dtype=np.float32)
    last = np.zeros((U, D), dtype=np.float32)
    first_set = np.zeros((U, 1), dtype=bool)
    last_set = np.zeros((U, 1), dtype=bool)

    for t in range(T):
        mt = (mask[:, t, 0] > 0)[:, None]
        take = mt & (~first_set)
        first[take[:, 0], :] = X[take[:, 0], t, :]
        first_set = first_set | mt

    for t in range(T - 1, -1, -1):
        mt = (mask[:, t, 0] > 0)[:, None]
        take = mt & (~last_set)
        last[take[:, 0], :] = X[take[:, 0], t, :]
        last_set = last_set | mt

    return first.astype(np.float32), last.astype(np.float32)

def masked_slope_np(X: np.ndarray, mask: np.ndarray, eps=1e-6) -> np.ndarray:
    # slope over days (centered time), per feature
    U, T, D = X.shape
    t = np.arange(T, dtype=np.float32)[None, :, None]
    m = mask.astype(np.float32)
    t_mean = (t * m).sum(axis=1) / np.clip(m.sum(axis=1), 1.0, None)
    t_center = t - t_mean[:, None, :]
    denom = (t_center ** 2 * m).sum(axis=1) + eps
    numer = (t_center * X * m).sum(axis=1)
    return (numer / denom).astype(np.float32)

def velocity_features_np(X: np.ndarray, mask: np.ndarray, eps=1e-6):
    # day-to-day diffs, masked
    dX = X[:, 1:, :] - X[:, :-1, :]
    m = (mask[:, 1:, :] * mask[:, :-1, :]).astype(np.float32)
    cnt = np.clip(m.sum(axis=1), 1.0, None)

    v_mean = (dX * m).sum(axis=1) / cnt
    v_absmean = (np.abs(dX) * m).sum(axis=1) / cnt

    v_mu = v_mean
    v_var = (((dX - v_mu[:, None, :]) ** 2) * m).sum(axis=1) / cnt
    v_std = np.sqrt(np.maximum(v_var, eps)).astype(np.float32)
    return v_mean.astype(np.float32), v_absmean.astype(np.float32), v_std.astype(np.float32)

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

def fpr_predicted_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_pos = y_pred > 0
    denom = float(np.sum(pred_pos))
    if denom <= 0:
        return 0.0
    fp = float(np.sum((y_true < 0) & pred_pos))
    return fp / denom

# ============================================================
# Target encoding (OOF) + category counts (kept from template)
# ============================================================
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
# Conservative policy (kept compatible with previous approach)
# ============================================================
def apply_policy(pred_mean, pred_q20, pneg, w_q=0.20, use_min=True, thr_gate=0.65, snap_eps=1.0, risk_scale=0.25):
    pred_mean = pred_mean.astype(np.float32)
    pred_q20 = pred_q20.astype(np.float32)
    pneg = pneg.astype(np.float32)

    y = ((1.0 - w_q) * pred_mean + w_q * pred_q20).astype(np.float32)
    if use_min:
        y = np.minimum(y, pred_q20)

    if risk_scale > 0:
        gap = np.maximum(y - pred_q20, 0.0).astype(np.float32)
        y = y - (np.float32(risk_scale) * pneg * gap)

    if thr_gate is not None:
        y = np.where(pneg > np.float32(thr_gate), np.minimum(y, 0.0), y).astype(np.float32)

    if snap_eps is not None and snap_eps > 0:
        y = np.where(np.abs(y) < np.float32(snap_eps), 0.0, y).astype(np.float32)

    return y.astype(np.float32)

def tune_delta_for_score(y_true, y_pred, fpr_target=0.40, grid=None):
    if grid is None:
        grid = np.concatenate([
            np.linspace(-150, 0, 16),
            np.linspace(0, 120, 25),
            np.linspace(130, 260, 14),
        ]).astype(np.float32)

    best_s = -1e18
    best_d = 0.0
    for d in grid:
        y_adj = (y_pred - d).astype(np.float32)
        if fpr_predicted_positive(y_true, y_adj) > fpr_target:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            s = compute_pareto_multi_objective(y_true, y_adj)
        if s > best_s:
            best_s = float(s)
            best_d = float(d)
    return best_s, best_d

# ============================================================
# LightGBM: custom AWMSE objective (Method 1)
# ============================================================
def awmse_lgb_obj(preds: np.ndarray, train_data: lgb.Dataset):
    y = train_data.get_label().astype(np.float32)
    p = preds.astype(np.float32)

    fp = (p > 0) & (y < 0)
    fn = (p < 0) & (y > 0)

    w = np.ones_like(y, dtype=np.float32)
    if np.any(fp):
        w[fp] = (2.5 + 0.02 * np.abs(y[fp])).astype(np.float32)
    if np.any(fn):
        w[fn] = (1.5 + 0.01 * np.clip(y[fn], 0, None)).astype(np.float32)

    grad = 2.0 * w * (p - y)
    hess = 2.0 * w
    return grad, hess

# ============================================================
# Build user-level model matrix (temporal + static)
# ============================================================
def build_user_matrix(X3d: np.ndarray, mask: np.ndarray, X_static: np.ndarray, X_cat: np.ndarray):
    """
    Returns a dense float32 matrix for GBDT:
      - flattened 7-day sequence
      - aggregated temporal stats
      - static engineered features (TE + cat counts)
      - raw cat codes (as float; TE already carries most signal)
    """
    U, T, D = X3d.shape

    # Flatten sequence
    X_flat = X3d.reshape(U, T * D).astype(np.float32)

    # Aggregates
    mu = masked_mean_np(X3d, mask)
    sd = masked_std_np(X3d, mask)
    mn, mx = masked_minmax_np(X3d, mask)
    first, last = masked_first_last_np(X3d, mask)
    delta = (last - first).astype(np.float32)
    slope = masked_slope_np(X3d, mask)
    v_mean, v_absmean, v_std = velocity_features_np(X3d, mask)

    daycnt = mask.sum(axis=1).astype(np.float32)  # (U,1)

    # Combine
    parts = [
        X_flat,
        mu, sd, mn, mx,
        first, last, delta, slope,
        v_mean, v_absmean, v_std,
        daycnt,
        X_static.astype(np.float32) if X_static is not None else np.zeros((U, 0), np.float32),
        X_cat.astype(np.float32) if X_cat is not None else np.zeros((U, 0), np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)

# ============================================================
# Training helpers
# ============================================================
def train_lgb_reg_aw(Xtr, ytr, Xva, yva, seed=42):
    params = dict(
        boosting_type="gbdt",
        objective="regression",  # ignored by fobj, but required
        metric="rmse",
        learning_rate=0.03,
        num_leaves=256,
        min_data_in_leaf=40,
        feature_fraction=0.75,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1e-3,
        max_bin=255,
        verbosity=-1,
        seed=seed,
    )
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=5000,
        valid_sets=[dva],
        fobj=awmse_lgb_obj,
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

def train_lgb_quantile(Xtr, ytr, Xva, yva, alpha=0.20, seed=43):
    params = dict(
        boosting_type="gbdt",
        objective="quantile",
        alpha=float(alpha),
        metric="quantile",
        learning_rate=0.03,
        num_leaves=256,
        min_data_in_leaf=40,
        feature_fraction=0.75,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1e-3,
        max_bin=255,
        verbosity=-1,
        seed=seed,
    )
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

def train_lgb_binary(Xtr, ytr_bin, Xva, yva_bin, seed=44):
    pos = float(np.sum(ytr_bin == 1))
    neg = float(np.sum(ytr_bin == 0))
    spw = neg / max(pos, 1.0)

    params = dict(
        boosting_type="gbdt",
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=128,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1e-3,
        max_bin=255,
        verbosity=-1,
        seed=seed,
        scale_pos_weight=spw,
    )
    dtr = lgb.Dataset(Xtr, label=ytr_bin.astype(np.float32))
    dva = lgb.Dataset(Xva, label=yva_bin.astype(np.float32), reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

def train_lgb_reg_log1p_positive(Xtr, ytr, Xva, yva, seed=45):
    # ytr, yva are strictly positive for training subsets
    ytr_t = np.log1p(ytr).astype(np.float32)
    yva_t = np.log1p(yva).astype(np.float32)

    params = dict(
        boosting_type="gbdt",
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=256,
        min_data_in_leaf=30,
        feature_fraction=0.75,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1e-3,
        max_bin=255,
        verbosity=-1,
        seed=seed,
    )
    dtr = lgb.Dataset(Xtr, label=ytr_t)
    dva = lgb.Dataset(Xva, label=yva_t, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return model

# ============================================================
# Main
# ============================================================
def main():
    set_seed(42)

    print("=" * 70)
    print("Improved Hybrid GBDT Ensemble: AWMSE mean + q20 + hurdle(sign) + conservative calibration")
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
    yte = test_df.groupby(ID_COL)[TARGET_COL].first().reindex(te_users).values.astype(np.float32)  # not used for tuning

    anchor_mean = float(np.mean(np.concatenate([ytr, yva], axis=0)))
    print(f"Anchor mean (train+val) = {anchor_mean:.6f}")

    # categorical encodings + TE + counts
    cat_maps = build_cat_maps(train_df, cat_cols)
    Xc_tr, cat_names = encode_user_cats(train_df, tr_users, cat_cols, cat_maps)
    Xc_va, _ = encode_user_cats(val_df,   va_users, cat_cols, cat_maps)
    Xc_te, _ = encode_user_cats(test_df,  te_users, cat_cols, cat_maps)

    te_tr, te_va, te_te, _ = build_target_encoding_features_v2(Xc_tr, ytr, Xc_va, Xc_te, cat_names, seed=42)
    (cnt_tr, _), (cnt_va, _), (cnt_te, _) = build_cat_count_features(train_df, [train_df, val_df, test_df], cat_cols)

    X_static_tr = np.concatenate([te_tr, cnt_tr], axis=1).astype(np.float32)
    X_static_va = np.concatenate([te_va, cnt_va], axis=1).astype(np.float32)
    X_static_te = np.concatenate([te_te, cnt_te], axis=1).astype(np.float32)

    # sequence numeric tensor (log-signed + standardize on train)
    Xn_tr_3d, m_tr = build_user_numeric_tensor(train_df, tr_users, num_cols, n_days=N_DAYS)
    Xn_va_3d, m_va = build_user_numeric_tensor(val_df,   va_users, num_cols, n_days=N_DAYS)
    Xn_te_3d, m_te = build_user_numeric_tensor(test_df,  te_users, num_cols, n_days=N_DAYS)

    Xn_tr_3d, mu_num, std_num = standardize_3d(Xn_tr_3d, m_tr, Xn_tr_3d, m_tr)
    Xn_va_3d = ((Xn_va_3d - mu_num[None, None, :]) / std_num[None, None, :]).astype(np.float32) * m_va
    Xn_te_3d = ((Xn_te_3d - mu_num[None, None, :]) / std_num[None, None, :]).astype(np.float32) * m_te

    # build dense matrices for GBDTs
    Xtr = build_user_matrix(Xn_tr_3d, m_tr, X_static_tr, Xc_tr)
    Xva = build_user_matrix(Xn_va_3d, m_va, X_static_va, Xc_va)
    Xte = build_user_matrix(Xn_te_3d, m_te, X_static_te, Xc_te)

    # train-derived clip on target for safety (pre-policy)
    lo, hi = np.quantile(ytr, [0.001, 0.999])
    lo, hi = float(lo), float(hi)
    print(f"Target clip: [{lo:.3f}, {hi:.3f}]")

    # -----------------------
    # Base models (train->val)
    # -----------------------
    print("\n[Training] LightGBM AWMSE mean regressor...")
    m_aw = train_lgb_reg_aw(Xtr, ytr, Xva, yva, seed=42)
    pred_aw_va = m_aw.predict(Xva, num_iteration=m_aw.best_iteration).astype(np.float32)
    pred_aw_te = m_aw.predict(Xte, num_iteration=m_aw.best_iteration).astype(np.float32)

    print("[Training] LightGBM Quantile (tau=0.20) regressor...")
    m_q20 = train_lgb_quantile(Xtr, ytr, Xva, yva, alpha=0.20, seed=43)
    pred_q_va = m_q20.predict(Xva, num_iteration=m_q20.best_iteration).astype(np.float32)
    pred_q_te = m_q20.predict(Xte, num_iteration=m_q20.best_iteration).astype(np.float32)

    print("[Training] LightGBM Negative-risk classifier P(y<0)...")
    ytr_neg = (ytr < 0).astype(np.int32)
    yva_neg = (yva < 0).astype(np.int32)
    m_neg = train_lgb_binary(Xtr, ytr_neg, Xva, yva_neg, seed=44)
    pneg_va = m_neg.predict(Xva, num_iteration=m_neg.best_iteration).astype(np.float32)
    pneg_te = m_neg.predict(Xte, num_iteration=m_neg.best_iteration).astype(np.float32)

    # Hurdle magnitude models (log1p magnitudes)
    print("[Training] Hurdle magnitude regressors (pos/neg magnitudes)...")
    tr_pos = ytr > 0
    va_pos = yva > 0
    tr_neg = ytr < 0
    va_neg = yva < 0

    # Positive regressor: predict log1p(y) for y>0
    m_pos = None
    if np.sum(tr_pos) > 500 and np.sum(va_pos) > 50:
        m_pos = train_lgb_reg_log1p_positive(Xtr[tr_pos], ytr[tr_pos], Xva[va_pos], yva[va_pos], seed=45)
        pos_va = np.expm1(m_pos.predict(Xva, num_iteration=m_pos.best_iteration)).astype(np.float32)
        pos_te = np.expm1(m_pos.predict(Xte, num_iteration=m_pos.best_iteration)).astype(np.float32)
    else:
        # fallback
        pos_va = np.maximum(pred_aw_va, 0.0).astype(np.float32)
        pos_te = np.maximum(pred_aw_te, 0.0).astype(np.float32)

    # Negative magnitude regressor: predict log1p(|y|) for y<0
    m_nmag = None
    if np.sum(tr_neg) > 500 and np.sum(va_neg) > 50:
        m_nmag = train_lgb_reg_log1p_positive(Xtr[tr_neg], np.abs(ytr[tr_neg]), Xva[va_neg], np.abs(yva[va_neg]), seed=46)
        nmag_va = np.expm1(m_nmag.predict(Xva, num_iteration=m_nmag.best_iteration)).astype(np.float32)
        nmag_te = np.expm1(m_nmag.predict(Xte, num_iteration=m_nmag.best_iteration)).astype(np.float32)
    else:
        nmag_va = np.maximum(-pred_aw_va, 0.0).astype(np.float32)
        nmag_te = np.maximum(-pred_aw_te, 0.0).astype(np.float32)

    pred_h_va = ((1.0 - pneg_va) * pos_va - pneg_va * nmag_va).astype(np.float32)
    pred_h_te = ((1.0 - pneg_te) * pos_te - pneg_te * nmag_te).astype(np.float32)

    # Clip base predictions (train-derived)
    pred_aw_va = np.clip(pred_aw_va, lo, hi).astype(np.float32)
    pred_aw_te = np.clip(pred_aw_te, lo, hi).astype(np.float32)
    pred_q_va  = np.clip(pred_q_va,  lo, hi).astype(np.float32)
    pred_q_te  = np.clip(pred_q_te,  lo, hi).astype(np.float32)
    pred_h_va  = np.clip(pred_h_va,  lo, hi).astype(np.float32)
    pred_h_te  = np.clip(pred_h_te,  lo, hi).astype(np.float32)

    # -----------------------
    # Tune ensemble policy on VAL
    # -----------------------
    print("\n[Tuning] Ensemble blend + conservative policy on VAL...")

    w_aw_grid = [0.55, 0.70, 0.85, 0.95]
    w_q_grid = [0.00, 0.10, 0.20, 0.30]
    use_min_grid = [True, False]
    thr_gate_grid = [0.55, 0.65, 0.75]
    snap_eps_grid = [0.0, 0.5, 1.0, 2.0]
    risk_scale_grid = [0.0, 0.20, 0.35]

    best = (-1e18, None)

    for w_aw in w_aw_grid:
        pred_mean_va = (np.float32(w_aw) * pred_aw_va + np.float32(1.0 - w_aw) * pred_h_va).astype(np.float32)

        for w_q in w_q_grid:
            for use_min in use_min_grid:
                for thr_gate in thr_gate_grid:
                    for snap_eps in snap_eps_grid:
                        for risk_scale in risk_scale_grid:
                            y_raw = apply_policy(
                                pred_mean_va, pred_q_va, pneg_va,
                                w_q=w_q, use_min=use_min, thr_gate=thr_gate,
                                snap_eps=snap_eps, risk_scale=risk_scale
                            )

                            a, b = affine_fit(yva, y_raw)
                            y_cal = (a * y_raw + b).astype(np.float32)
                            y_cal = mean_correct(y_cal, anchor_mean=anchor_mean)

                            s_best_delta, delta = tune_delta_for_score(yva, y_cal, fpr_target=0.40)
                            if s_best_delta > best[0]:
                                best = (s_best_delta, dict(
                                    w_aw=float(w_aw),
                                    w_q=float(w_q),
                                    use_min=bool(use_min),
                                    thr_gate=float(thr_gate),
                                    snap_eps=float(snap_eps),
                                    risk_scale=float(risk_scale),
                                    a=float(a),
                                    b=float(b),
                                    delta=float(delta),
                                ))

    best_val_score, best_params = best
    print(f"Best VAL score = {best_val_score:.6f}")
    print("Best params:", best_params)

    # -----------------------
    # Inference on TEST
    # -----------------------
    print("\n[Inference] Predicting TEST...")

    pred_mean_te = (np.float32(best_params["w_aw"]) * pred_aw_te + np.float32(1.0 - best_params["w_aw"]) * pred_h_te).astype(np.float32)

    y_raw_te = apply_policy(
        pred_mean_te, pred_q_te, pneg_te,
        w_q=best_params["w_q"],
        use_min=best_params["use_min"],
        thr_gate=best_params["thr_gate"],
        snap_eps=best_params["snap_eps"],
        risk_scale=best_params["risk_scale"],
    )

    final_pred = (best_params["a"] * y_raw_te + best_params["b"]).astype(np.float32)
    final_pred = mean_correct(final_pred, anchor_mean=anchor_mean)
    final_pred = (final_pred - np.float32(best_params["delta"])).astype(np.float32)

    # Align to test index order (user-level)
    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    test_predictions = pd.Series(final_pred, index=te_users).reindex(y_test.index).values.astype(np.float32)

    print("\n" + "=" * 70)
    score_val, metrics = compute_score(y_test.values, test_predictions)
    print("=" * 70)
    print(f"\n🎯 Final Score: {score_val:.6f}")

    # Optional artifact
    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/improved_gbdt_ensemble_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "method": "Hybrid GBDT Ensemble: AWMSE mean + q20 + hurdle(sign) + conservative calibration",
                "score": float(score_val),
                "metrics": metrics,
                "best_val_policy_score": float(best_val_score),
                "policy": best_params,
                "num_cols": int(len(num_cols)),
                "cat_cols": int(len(cat_cols)),
                "seq_days": int(N_DAYS),
                "X_dim": int(Xtr.shape[1]),
                "anchor_mean": float(anchor_mean),
                "clip_lo": float(lo),
                "clip_hi": float(hi),
                "lgb_aw_best_iter": int(m_aw.best_iteration),
                "lgb_q20_best_iter": int(m_q20.best_iteration),
                "lgb_pneg_best_iter": int(m_neg.best_iteration),
            },
            f,
            indent=2,
        )

    return test_predictions, y_test.values

if __name__ == "__main__":
    test_predictions, y_test = main()

    # CRITICAL: score must be computed on TEST and assigned to variable named exactly `score`
    score = compute_pareto_multi_objective(y_test, test_predictions)
    print(f"score = {score}")