#!/usr/bin/env python3
"""
AviaAgentMonty - Execution Node: 58d24a9b
Type: original
Generated: 2026-01-13T00:10:53.748082
Mutation: standard
Parent: None

DO NOT DELETE - This file is preserved for reproducibility.
"""
# Suppress warnings to prevent false failures
import warnings
warnings.filterwarnings("ignore")

#!/usr/bin/env python3
"""
Improved Method 17 -> Method 17.2:
- Fix LightGBM custom AWMSE objective usage (was defined but NOT used)
- Stronger categorical target encoding (mean + neg/zero rate + count/logcount + freq), OOF (no test leakage)
- CV bagging for base models to reduce bias mismatch between tuning and final inference
- Stacking meta-learner (RidgeCV) on OOF predictions + calibrated affine transform (a,b) learned on OOF
  to improve Error Rate (sum accuracy) without hurting rank metrics (Gini/Spearman)
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

import lightgbm as lgb

# -----------------------------
# Config
# -----------------------------
SEED = 42
np.random.seed(SEED)

TRAIN_PATH = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"
DAY_COL = "TDATE_RN"  # expected 1..7

# Keep the original list as a strong prior for temporal features
NUMERICAL_COLS = [
    "DEPOSIT_AMOUNT", "REC_USD", "REC_USD_CUM", "REC_USD_D6", "CPI",
    "RANK1_PLAY_CNT_ALL", "PLAY_CNT_ALL", "ACTUAL_ENTRY_FEE_CASH",
    "ACTUAL_REWARD_CASH", "PLAY_CNT_CASH", "HIGHFEE_PLAY_CNT_CASH",
    "CASH_RATIO", "ACTIVE_DAYS_ALL_CUM", "PLAY_CNT_ALL_CUM", "SESSION_CNT_ALL",
    "CLEAR_PLAY_CNT_ALL", "RANK_UNDER3_PLAY_CNT_ALL", "JN_PLAY_CNT", "FJ80_PLAY_CNT"
]

# -----------------------------
# CRITICAL: Scoring functions (PRESERVE EXACTLY)
# -----------------------------
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

# Optional wrapper (does not modify existing scoring)
def compute_pareto_multi_objective(y_true, y_pred):
    return compute_score(y_true, y_pred)[0]

# -----------------------------
# Helpers
# -----------------------------
def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def _detect_cols(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [TARGET_COL, ID_COL, DAY_COL]]

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET_COL, ID_COL, DAY_COL]]
    return num_cols, cat_cols

def create_features(df, is_train=True):
    """
    User-level features:
    - Strong numeric aggregates across all numeric columns
    - Temporal wide + trajectory features for a curated subset of temporal_cols
    - Keep first-observed categorical snapshot (target-encoded later)
    """
    df = df.copy()
    has_day = DAY_COL in df.columns

    num_cols_all, cat_cols = _detect_cols(df)

    if has_day:
        df = df.sort_values([ID_COL, DAY_COL])
        df = df.drop_duplicates(subset=[ID_COL, DAY_COL], keep="first")
    else:
        df = df.sort_values([ID_COL])

    # --- Base numeric aggregates ---
    agg_funcs = ["mean", "std", "min", "max", "sum", "first", "last", "median"]
    if len(num_cols_all) > 0:
        uf_num = df.groupby(ID_COL)[num_cols_all].agg(agg_funcs)
        uf_num.columns = ["_".join(c) for c in uf_num.columns]
        uf_num = uf_num.reset_index()
    else:
        uf_num = df[[ID_COL]].drop_duplicates().copy()

    # --- User-level categorical snapshots (first) ---
    if len(cat_cols) > 0:
        uf_cat = df.groupby(ID_COL)[cat_cols].first().reset_index()
        uf = uf_num.merge(uf_cat, on=ID_COL, how="left")
    else:
        uf = uf_num

    # --- Temporal trajectory features on curated subset ---
    temporal_cols = [c for c in NUMERICAL_COLS if c in num_cols_all]
    if has_day and len(temporal_cols) > 0:
        wide = df[[ID_COL, DAY_COL] + temporal_cols].set_index([ID_COL, DAY_COL])[temporal_cols].unstack(DAY_COL)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([temporal_cols, list(range(1, 8))]))
        wide.columns = [f"{c}_d{d}" for c, d in wide.columns]
        wide = wide.reset_index()

        derived = {ID_COL: wide[ID_COL].values}
        t = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
        t2 = float((t**2).sum())

        for c in temporal_cols:
            cols_d = [f"{c}_d{i}" for i in range(1, 8)]
            s = wide[cols_d].to_numpy(dtype=np.float32)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

            d1, d3, d5, d7 = s[:, 0], s[:, 2], s[:, 4], s[:, 6]
            mean_1_3 = s[:, 0:3].mean(axis=1)
            mean_5_7 = s[:, 4:7].mean(axis=1)

            vel = (mean_5_7 - mean_1_3) / 4.0
            v_early = (d3 - d1) / 2.0
            v_late = (d7 - d5) / 2.0
            acc = v_late - v_early

            diffs = np.diff(s, axis=1)
            vol = diffs.std(axis=1) / (np.abs(s.mean(axis=1)) + 1e-3)

            rec = _safe_div(d7, d1, eps=1e-3)
            peak_day = (np.argmax(s, axis=1) + 1).astype(np.float32)  # 1..7
            trough_day = (np.argmin(s, axis=1) + 1).astype(np.float32)
            peak_last2 = (peak_day >= 6).astype(np.float32)

            last3_mean = s[:, 4:7].mean(axis=1)
            first3_mean = s[:, 0:3].mean(axis=1)
            mid3_mean = s[:, 2:5].mean(axis=1)

            slope = (s * t[None, :]).sum(axis=1) / t2

            nonzero_days = (s != 0).sum(axis=1).astype(np.float32)
            last_minus_first = (d7 - d1).astype(np.float32)
            max_minus_min = (s.max(axis=1) - s.min(axis=1)).astype(np.float32)
            auc = s.sum(axis=1).astype(np.float32)

            derived[f"{c}_mean_1_3"] = mean_1_3
            derived[f"{c}_mean_5_7"] = mean_5_7
            derived[f"{c}_velocity"] = vel
            derived[f"{c}_acceleration"] = acc
            derived[f"{c}_volatility"] = vol
            derived[f"{c}_recency_ratio"] = rec
            derived[f"{c}_peak_day"] = peak_day
            derived[f"{c}_trough_day"] = trough_day
            derived[f"{c}_peak_last2"] = peak_last2
            derived[f"{c}_first3_mean"] = first3_mean
            derived[f"{c}_mid3_mean"] = mid3_mean
            derived[f"{c}_last3_mean"] = last3_mean
            derived[f"{c}_trend_slope"] = slope
            derived[f"{c}_nonzero_days"] = nonzero_days
            derived[f"{c}_last_minus_first"] = last_minus_first
            derived[f"{c}_max_minus_min"] = max_minus_min
            derived[f"{c}_auc7"] = auc

        uf = uf.merge(wide, on=ID_COL, how="left")
        uf = uf.merge(pd.DataFrame(derived), on=ID_COL, how="left")

    if is_train and TARGET_COL in df.columns:
        y = df.groupby(ID_COL)[TARGET_COL].first().reset_index()
        uf = uf.merge(y, on=ID_COL, how="left")

    uf = uf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return uf

def add_target_encodings(train_f, test_f, cat_cols, n_splits=5, smoothing=30.0, seed=42):
    """
    OOF target encoding on training set only; then apply full-train mapping to test.
    Adds:
      - TE mean
      - TE neg_rate  (y<0)
      - TE zero_rate (y==0)
      - TE count / logcount
      - category frequency
    """
    if len(cat_cols) == 0:
        return train_f, test_f

    train_f = train_f.reset_index(drop=True).copy()
    test_f = test_f.copy()

    y = train_f[TARGET_COL].values.astype(np.float32)
    global_mean = float(np.mean(y))
    global_neg = float(np.mean(y < 0))
    global_zero = float(np.mean(y == 0))

    # helper flags (train only)
    is_neg = (y < 0).astype(np.float32)
    is_zero = (y == 0).astype(np.float32)
    train_f["__is_neg"] = is_neg
    train_f["__is_zero"] = is_zero

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for c in cat_cols:
        oof_mean = np.zeros(len(train_f), dtype=np.float32)
        oof_neg = np.zeros(len(train_f), dtype=np.float32)
        oof_zero = np.zeros(len(train_f), dtype=np.float32)
        oof_cnt = np.zeros(len(train_f), dtype=np.float32)
        oof_freq = np.zeros(len(train_f), dtype=np.float32)

        for tr_idx, te_idx in kf.split(train_f):
            tr = train_f.iloc[tr_idx]
            te = train_f.iloc[te_idx]

            stats = tr.groupby(c).agg(
                mean=(TARGET_COL, "mean"),
                cnt=(TARGET_COL, "size"),
                neg_rate=("__is_neg", "mean"),
                zero_rate=("__is_zero", "mean"),
            )

            cnt = stats["cnt"].astype(np.float32)
            sm_mean = (stats["mean"].astype(np.float32) * cnt + global_mean * smoothing) / (cnt + smoothing)
            sm_neg = (stats["neg_rate"].astype(np.float32) * cnt + global_neg * smoothing) / (cnt + smoothing)
            sm_zero = (stats["zero_rate"].astype(np.float32) * cnt + global_zero * smoothing) / (cnt + smoothing)

            freq = (cnt / float(cnt.sum())).astype(np.float32)

            te_vals = te[c]
            oof_mean[te_idx] = te_vals.map(sm_mean).fillna(global_mean).astype(np.float32).values
            oof_neg[te_idx] = te_vals.map(sm_neg).fillna(global_neg).astype(np.float32).values
            oof_zero[te_idx] = te_vals.map(sm_zero).fillna(global_zero).astype(np.float32).values
            oof_cnt[te_idx] = te_vals.map(cnt).fillna(0.0).astype(np.float32).values
            oof_freq[te_idx] = te_vals.map(freq).fillna(0.0).astype(np.float32).values

        # fit on full train for test transform
        stats_full = train_f.groupby(c).agg(
            mean=(TARGET_COL, "mean"),
            cnt=(TARGET_COL, "size"),
            neg_rate=("__is_neg", "mean"),
            zero_rate=("__is_zero", "mean"),
        )
        cnt_full = stats_full["cnt"].astype(np.float32)
        sm_mean_full = (stats_full["mean"].astype(np.float32) * cnt_full + global_mean * smoothing) / (cnt_full + smoothing)
        sm_neg_full = (stats_full["neg_rate"].astype(np.float32) * cnt_full + global_neg * smoothing) / (cnt_full + smoothing)
        sm_zero_full = (stats_full["zero_rate"].astype(np.float32) * cnt_full + global_zero * smoothing) / (cnt_full + smoothing)
        freq_full = (cnt_full / float(cnt_full.sum())).astype(np.float32)

        train_f[f"{c}__te_mean"] = oof_mean
        train_f[f"{c}__te_neg_rate"] = oof_neg
        train_f[f"{c}__te_zero_rate"] = oof_zero
        train_f[f"{c}__cnt"] = oof_cnt
        train_f[f"{c}__logcnt"] = np.log1p(oof_cnt).astype(np.float32)
        train_f[f"{c}__freq"] = oof_freq

        test_f[f"{c}__te_mean"] = test_f[c].map(sm_mean_full).fillna(global_mean).astype(np.float32)
        test_f[f"{c}__te_neg_rate"] = test_f[c].map(sm_neg_full).fillna(global_neg).astype(np.float32)
        test_f[f"{c}__te_zero_rate"] = test_f[c].map(sm_zero_full).fillna(global_zero).astype(np.float32)
        test_cnt = test_f[c].map(cnt_full).fillna(0.0).astype(np.float32)
        test_f[f"{c}__cnt"] = test_cnt
        test_f[f"{c}__logcnt"] = np.log1p(test_cnt).astype(np.float32)
        test_f[f"{c}__freq"] = test_f[c].map(freq_full).fillna(0.0).astype(np.float32)

    # drop helpers / raw cats
    train_f = train_f.drop(columns=["__is_neg", "__is_zero"] + cat_cols, errors="ignore")
    test_f = test_f.drop(columns=cat_cols, errors="ignore")
    return train_f, test_f

# -----------------------------
# LightGBM: Custom AWMSE objective (Method 1)
# -----------------------------
def lgb_aw_obj(preds, train_data):
    y = train_data.get_label().astype(np.float32)
    p = preds.astype(np.float32)

    w = np.ones_like(y, dtype=np.float32)
    fp = (p > 0) & (y < 0)
    fn = (p < 0) & (y > 0)

    w[fp] = 2.5 + 0.02 * np.abs(y[fp])
    w[fn] = 1.5 + 0.01 * y[fn]
    w = np.clip(w, 0.1, 25.0)

    grad = 2.0 * w * (p - y)
    hess = 2.0 * w
    return grad, hess

def lgb_aw_eval(preds, train_data):
    y = train_data.get_label().astype(np.float32)
    p = preds.astype(np.float32)

    w = np.ones_like(y, dtype=np.float32)
    fp = (p > 0) & (y < 0)
    fn = (p < 0) & (y > 0)

    w[fp] = 2.5 + 0.02 * np.abs(y[fp])
    w[fn] = 1.5 + 0.01 * y[fn]
    w = np.clip(w, 0.1, 25.0)

    awmse = float(np.mean(w * (p - y) ** 2))
    return "awmse", awmse, False  # lower is better

def train_lgb_aw(X_tr, y_tr, X_va, y_va, seed=42):
    params = dict(
        objective="none",           # IMPORTANT for custom objective
        metric="None",
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=50,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.3,
        max_bin=255,
        verbose=-1,
        seed=seed,
        num_threads=-1,
        force_col_wise=True,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=6000,
        valid_sets=[dval],
        fobj=lgb_aw_obj,
        feval=lgb_aw_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=250, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model

def train_lgb_rmse(X_tr, y_tr, X_va, y_va, seed=42):
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.035,
        num_leaves=192,
        min_data_in_leaf=60,
        feature_fraction=0.75,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=0.2,
        max_bin=255,
        verbose=-1,
        seed=seed,
        num_threads=-1,
        force_col_wise=True,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=6000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=250, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model

def train_lgb_quantile(X_tr, y_tr, X_va, y_va, alpha=0.20, seed=42):
    params = dict(
        objective="quantile",
        alpha=float(alpha),
        metric="quantile",
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.3,
        max_bin=255,
        verbose=-1,
        seed=seed,
        num_threads=-1,
        force_col_wise=True,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=6000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=250, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model

def train_lgb_neg_classifier(X_tr, y_tr, X_va, y_va, seed=42):
    y_tr_bin = (y_tr < 0).astype(np.int32)
    y_va_bin = (y_va < 0).astype(np.int32)
    pos = float(y_tr_bin.mean())
    # We want to reduce "missed negatives" => upweight negative class (label=1)
    scale_pos_weight = float((1.0 - pos) / (pos + 1e-6)) * 1.25

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.04,
        num_leaves=64,
        min_data_in_leaf=100,
        feature_fraction=0.80,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=0.5,
        max_bin=255,
        verbose=-1,
        seed=seed,
        num_threads=-1,
        force_col_wise=True,
        scale_pos_weight=scale_pos_weight,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr_bin)
    dval = lgb.Dataset(X_va, label=y_va_bin, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model

# -----------------------------
# Meta features + calibration
# -----------------------------
def build_meta_features(p_aw, p_rmse, p_q20, p_q80, p_negprob):
    p_aw = p_aw.astype(np.float32)
    p_rmse = p_rmse.astype(np.float32)
    p_q20 = p_q20.astype(np.float32)
    p_q80 = p_q80.astype(np.float32)
    p_negprob = p_negprob.astype(np.float32)

    d1 = (p_aw - p_q20).astype(np.float32)
    d2 = (p_q80 - p_aw).astype(np.float32)
    spread = (p_q80 - p_q20).astype(np.float32)
    absdiff = np.abs(p_aw - p_rmse).astype(np.float32)

    X = np.stack(
        [p_aw, p_rmse, p_q20, p_q80, p_negprob, d1, d2, spread, absdiff],
        axis=1
    ).astype(np.float32)
    return X

def fit_affine_calibration(y, p):
    """
    Fit y ≈ a*p + b (OLS). This exactly matches the mean on the calibration data,
    improving Error Rate (sum accuracy) while preserving ranking if a>0.
    """
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    var = np.var(p)
    if var < 1e-12:
        a = 1.0
    else:
        a = float(np.cov(p, y, bias=True)[0, 1] / var)
    if not np.isfinite(a):
        a = 1.0
    a = float(np.clip(a, 0.05, 5.0))
    b = float(y.mean() - a * p.mean())
    return a, b

# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print("Improved Method 17.2: CV-bagged LGB ensemble + stacking + affine calibration")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Build user-level features
    train_f = create_features(train_df, is_train=True)
    val_f   = create_features(val_df, is_train=True)
    test_f  = create_features(test_df, is_train=False)

    # Combine train+val for stronger training & more stable sum calibration (no test leakage)
    full_f = pd.concat([train_f, val_f], axis=0, ignore_index=True)
    y_full = full_f[TARGET_COL].values.astype(np.float32)

    # Target encode categoricals using FULL TRAINING only (train+val); apply to test
    cat_cols = full_f.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [ID_COL, TARGET_COL]]
    full_f, test_f = add_target_encodings(full_f, test_f, cat_cols, n_splits=5, smoothing=35.0, seed=SEED)

    # Align features
    feat_cols = [c for c in full_f.columns if c not in [ID_COL, TARGET_COL]]
    X_full = full_f[feat_cols].values.astype(np.float32)
    X_test = test_f[feat_cols].values.astype(np.float32)
    test_ids = test_f[ID_COL].values

    print(f"Users: full_train={len(full_f)}, test={len(test_f)}")
    print(f"Features = {len(feat_cols)}")

    # -----------------------------
    # CV bagging: generate OOF preds for stacking + avg test preds
    # -----------------------------
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_aw = np.zeros(len(full_f), dtype=np.float32)
    oof_rmse = np.zeros(len(full_f), dtype=np.float32)
    oof_q20 = np.zeros(len(full_f), dtype=np.float32)
    oof_q80 = np.zeros(len(full_f), dtype=np.float32)
    oof_negp = np.zeros(len(full_f), dtype=np.float32)

    test_aw = np.zeros(len(test_f), dtype=np.float32)
    test_rmse = np.zeros(len(test_f), dtype=np.float32)
    test_q20 = np.zeros(len(test_f), dtype=np.float32)
    test_q80 = np.zeros(len(test_f), dtype=np.float32)
    test_negp = np.zeros(len(test_f), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
        X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
        X_va, y_va = X_full[va_idx], y_full[va_idx]

        seed_fold = SEED + fold * 17
        print(f"\n--- Fold {fold}/{n_splits} ---")

        m_aw = train_lgb_aw(X_tr, y_tr, X_va, y_va, seed=seed_fold)
        p_va_aw = m_aw.predict(X_va, num_iteration=m_aw.best_iteration).astype(np.float32)
        p_te_aw = m_aw.predict(X_test, num_iteration=m_aw.best_iteration).astype(np.float32)

        m_rmse = train_lgb_rmse(X_tr, y_tr, X_va, y_va, seed=seed_fold + 1)
        p_va_rmse = m_rmse.predict(X_va, num_iteration=m_rmse.best_iteration).astype(np.float32)
        p_te_rmse = m_rmse.predict(X_test, num_iteration=m_rmse.best_iteration).astype(np.float32)

        m_q20 = train_lgb_quantile(X_tr, y_tr, X_va, y_va, alpha=0.20, seed=seed_fold + 2)
        p_va_q20 = m_q20.predict(X_va, num_iteration=m_q20.best_iteration).astype(np.float32)
        p_te_q20 = m_q20.predict(X_test, num_iteration=m_q20.best_iteration).astype(np.float32)

        m_q80 = train_lgb_quantile(X_tr, y_tr, X_va, y_va, alpha=0.80, seed=seed_fold + 3)
        p_va_q80 = m_q80.predict(X_va, num_iteration=m_q80.best_iteration).astype(np.float32)
        p_te_q80 = m_q80.predict(X_test, num_iteration=m_q80.best_iteration).astype(np.float32)

        m_neg = train_lgb_neg_classifier(X_tr, y_tr, X_va, y_va, seed=seed_fold + 4)
        p_va_neg = m_neg.predict(X_va, num_iteration=m_neg.best_iteration).astype(np.float32)
        p_te_neg = m_neg.predict(X_test, num_iteration=m_neg.best_iteration).astype(np.float32)

        oof_aw[va_idx] = p_va_aw
        oof_rmse[va_idx] = p_va_rmse
        oof_q20[va_idx] = p_va_q20
        oof_q80[va_idx] = p_va_q80
        oof_negp[va_idx] = p_va_neg

        test_aw += p_te_aw / n_splits
        test_rmse += p_te_rmse / n_splits
        test_q20 += p_te_q20 / n_splits
        test_q80 += p_te_q80 / n_splits
        test_negp += p_te_neg / n_splits

    # -----------------------------
    # Meta learner (stacking) trained on OOF
    # -----------------------------
    X_meta_oof = build_meta_features(oof_aw, oof_rmse, oof_q20, oof_q80, oof_negp)
    X_meta_test = build_meta_features(test_aw, test_rmse, test_q20, test_q80, test_negp)

    scaler = StandardScaler()
    X_meta_oof_s = scaler.fit_transform(X_meta_oof)
    X_meta_test_s = scaler.transform(X_meta_test)

    meta = RidgeCV(alphas=np.logspace(-3, 3, 25), fit_intercept=True)
    meta.fit(X_meta_oof_s, y_full.astype(np.float64))
    p_oof_meta = meta.predict(X_meta_oof_s).astype(np.float32)
    p_test_meta = meta.predict(X_meta_test_s).astype(np.float32)

    # Affine calibration on OOF (improves Error Rate / sum accuracy)
    a, b = fit_affine_calibration(y_full, p_oof_meta)
    p_oof_final = (a * p_oof_meta + b).astype(np.float32)
    p_test_final = (a * p_test_meta + b).astype(np.float32)

    print("\n" + "=" * 60)
    print("OOF (train+val) diagnostic score (should be optimistic but stable):")
    _ = compute_score(y_full, p_oof_final)
    print("=" * 60)

    # -----------------------------
    # Score on TEST (required)
    # -----------------------------
    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(p_test_final, index=test_ids).reindex(y_test.index).values.astype(np.float32)

    print("\n" + "=" * 60)
    score, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 60)

    result = {
        "method": "Improved Method 17.2: CV-bagged LGB + stacking + affine calibration",
        "score": float(score),
        "metrics": metrics,
        "meta_alpha": float(getattr(meta, "alpha_", np.nan)),
        "affine_a": float(a),
        "affine_b": float(b),
        "n_features": int(len(feat_cols)),
        "n_train_full": int(len(full_f)),
    }
    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/method_17_2_results.json"
    try:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    print(f"score = {score}")
    return score, y_test.values, pred_aligned

if __name__ == "__main__":
    final_score, y_test_values, test_predictions = main()

    # =============================================================================
    # SCORE CALCULATION - MANDATORY (for EVALUATION, not training)
    # =============================================================================
    # NOTE: This is the EVALUATION metric. You can use ANY training loss you prefer.
    # But the final score MUST be calculated using this function.
    # The variable MUST be named exactly 'score' for the system to read it.
    # Using the pareto_multi_objective metric (higher is better)

    score = compute_pareto_multi_objective(y_test_values, test_predictions)
    print(f"score = {score}")  # This will be parsed by the system