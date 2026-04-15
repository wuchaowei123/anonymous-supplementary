# Suppress warnings to prevent false failures
import warnings
warnings.filterwarnings('ignore')

#!/usr/bin/env python3
"""
Method: Physics-inspired Deep Residual Time-Stepping (Lecture Notes: Deep Learning & Computational Physics)
- Implement a sequence model as an explicit time integrator (ResNet/Euler/RK2 viewpoint):
    h_{t+1} = h_t + dt * f(h_t, u_t)  (Forward Euler / RK2(Heun))
  where u_t are day-t features (7-day sequence), f is a small MLP "dynamics" operator, dt is a learnable step.
- Multi-task heads (aligned with task principles):
    (1) Mean regression with AWMSE (asymmetric business costs)
    (2) Conservative quantile (tau=0.20) with Pinball loss
    (3) Negative-risk classifier with weighted BCE
- Keep strong tabular/GBDT baselines (LightGBM) and stack with Ridge meta-learner + affine calibration.

CRITICAL: Scoring functions preserved EXACTLY as provided.
"""

import warnings
warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_PATH = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"
DAY_COL = "TDATE_RN"  # expected 1..7

NUMERICAL_COLS = [
    "DEPOSIT_AMOUNT", "REC_USD", "REC_USD_CUM", "REC_USD_D6", "CPI",
    "RANK1_PLAY_CNT_ALL", "PLAY_CNT_ALL", "ACTUAL_ENTRY_FEE_CASH",
    "ACTUAL_REWARD_CASH", "PLAY_CNT_CASH", "HIGHFEE_PLAY_CNT_CASH",
    "CASH_RATIO", "ACTIVE_DAYS_ALL_CUM", "PLAY_CNT_ALL_CUM", "SESSION_CNT_ALL",
    "CLEAR_PLAY_CNT_ALL", "RANK_UNDER3_PLAY_CNT_ALL", "JN_PLAY_CNT", "FJ80_PLAY_CNT"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def compute_pareto_multi_objective(y_true, y_pred):
    return compute_score(y_true, y_pred)[0]


# -----------------------------
# Helpers: feature engineering (tabular user-level)
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
    - Numeric aggregates across all numeric columns
    - Temporal wide + trajectory features for a curated subset of temporal_cols
    - First-observed categorical snapshot (target-encoded later)
    """
    df = df.copy()
    has_day = DAY_COL in df.columns

    num_cols_all, cat_cols = _detect_cols(df)

    if has_day:
        df = df.sort_values([ID_COL, DAY_COL])
        df = df.drop_duplicates(subset=[ID_COL, DAY_COL], keep="first")
    else:
        df = df.sort_values([ID_COL])

    agg_funcs = ["mean", "std", "min", "max", "sum", "first", "last", "median"]
    if len(num_cols_all) > 0:
        uf_num = df.groupby(ID_COL)[num_cols_all].agg(agg_funcs)
        uf_num.columns = ["_".join(c) for c in uf_num.columns]
        uf_num = uf_num.reset_index()
    else:
        uf_num = df[[ID_COL]].drop_duplicates().copy()

    if len(cat_cols) > 0:
        uf_cat = df.groupby(ID_COL)[cat_cols].first().reset_index()
        uf = uf_num.merge(uf_cat, on=ID_COL, how="left")
    else:
        uf = uf_num

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
            peak_day = (np.argmax(s, axis=1) + 1).astype(np.float32)
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

    train_f = train_f.drop(columns=["__is_neg", "__is_zero"] + cat_cols, errors="ignore")
    test_f = test_f.drop(columns=cat_cols, errors="ignore")
    return train_f, test_f


# -----------------------------
# Sequence building (7-day tensor per user) for physics-inspired model
# -----------------------------
def build_category_maps(df_full_train, cat_cols):
    """
    Build category->index mapping using TRAIN ONLY (train+val), no target usage.
    index 0 reserved for UNK.
    """
    maps = {}
    for c in cat_cols:
        vals = df_full_train[c].astype("object").fillna("__MISSING__").values
        uniq = pd.unique(vals)
        mp = {v: i+1 for i, v in enumerate(uniq)}  # start at 1
        maps[c] = mp
    return maps

def build_user_sequences(df_rows, user_ids, num_cols, cat_cols, cat_maps):
    """
    Build:
      X_num: (n_users, 7, n_num) float32
      X_cat: (n_users, 7, n_cat) int64 (embedding indices)
    """
    n_users = len(user_ids)
    n_days = 7
    n_num = len(num_cols)
    n_cat = len(cat_cols)

    id_to_idx = {u: i for i, u in enumerate(user_ids)}
    X_num = np.zeros((n_users, n_days, n_num), dtype=np.float32)
    X_cat = np.zeros((n_users, n_days, n_cat), dtype=np.int64)

    if n_users == 0:
        return X_num, X_cat

    df = df_rows.copy()
    df = df.sort_values([ID_COL, DAY_COL])
    df = df.drop_duplicates(subset=[ID_COL, DAY_COL], keep="first")

    # numeric fill
    if n_num > 0:
        df_num = df[[ID_COL, DAY_COL] + num_cols].copy()
        df_num[num_cols] = df_num[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        idx = df_num[ID_COL].map(id_to_idx).values
        day = (df_num[DAY_COL].values.astype(np.int64) - 1)
        ok = (idx >= 0) & (day >= 0) & (day < 7)
        vals = df_num.loc[ok, num_cols].to_numpy(dtype=np.float32, copy=False)
        X_num[idx[ok], day[ok], :] = vals

    # categorical fill
    for j, c in enumerate(cat_cols):
        mp = cat_maps.get(c, {})
        ser = df[c].astype("object").fillna("__MISSING__")
        idx = df[ID_COL].map(id_to_idx).values
        day = (df[DAY_COL].values.astype(np.int64) - 1)
        ok = (idx >= 0) & (day >= 0) & (day < 7)
        mapped = ser.map(mp).fillna(0).astype(np.int64).values
        X_cat[idx[ok], day[ok], j] = mapped[ok]

    return X_num, X_cat


# -----------------------------
# LightGBM: Custom AWMSE objective (Method 1) - FIXED
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
    return "awmse", awmse, False

def train_lgb_aw(X_tr, y_tr, X_va, y_va, seed=42):
    params = dict(
        objective="regression",
        metric="rmse",
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
    dtrain = lgb.Dataset(X_tr, label=y_tr.astype(np.float32))
    dval = lgb.Dataset(X_va, label=y_va.astype(np.float32), reference=dtrain)

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
    dtrain = lgb.Dataset(X_tr, label=y_tr.astype(np.float32))
    dval = lgb.Dataset(X_va, label=y_va.astype(np.float32), reference=dtrain)

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
    dtrain = lgb.Dataset(X_tr, label=y_tr.astype(np.float32))
    dval = lgb.Dataset(X_va, label=y_va.astype(np.float32), reference=dtrain)

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
# Physics-inspired sequence network (time-stepping ResNet / RK2)
# -----------------------------
def awmse_torch(y, p):
    """
    Asymmetric weighted MSE with piecewise weights (torch).
    """
    y = y.float()
    p = p.float()
    w = torch.ones_like(y)
    fp = (p > 0) & (y < 0)
    fn = (p < 0) & (y > 0)
    w = torch.where(fp, 2.5 + 0.02 * torch.abs(y), w)
    w = torch.where(fn, 1.5 + 0.01 * y, w)
    w = torch.clamp(w, 0.1, 25.0)
    return torch.mean(w * (p - y) ** 2)

def pinball_loss(y, q, tau=0.20):
    y = y.float()
    q = q.float()
    e = y - q
    return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))

class IndexSeqDataset(Dataset):
    def __init__(self, X_num_seq, X_cat_seq, X_static, y, indices):
        self.X_num_seq = X_num_seq
        self.X_cat_seq = X_cat_seq
        self.X_static = X_static
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        xnum = self.X_num_seq[idx]   # (7, n_num)
        xcat = self.X_cat_seq[idx]   # (7, n_cat)
        xst = self.X_static[idx]     # (n_static,)
        y = self.y[idx]
        return xnum, xcat, xst, y

class IndexSeqDatasetNoY(Dataset):
    def __init__(self, X_num_seq, X_cat_seq, X_static, indices):
        self.X_num_seq = X_num_seq
        self.X_cat_seq = X_cat_seq
        self.X_static = X_static
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.X_num_seq[idx], self.X_cat_seq[idx], self.X_static[idx]

class PhysicsTimeStepper(nn.Module):
    """
    Physics-inspired explicit time integrator:
      h_{t+1} = h_t + dt * f([h_t, u_t])     (Forward Euler)
    optionally RK2 (Heun):
      k1 = f(h, u); k2 = f(h + dt*k1, u); h <- h + dt*(k1+k2)/2

    u_t is a learned projection of day-t inputs + static forcing.
    """
    def __init__(self, n_num, cat_cardinalities, n_static,
                 hidden_dim=96, dyn_dim=192, dropout=0.10, use_rk2=True):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.n_static = n_static
        self.hidden_dim = hidden_dim
        self.use_rk2 = use_rk2

        # categorical embeddings (entity embeddings)
        self.cat_embeds = nn.ModuleList()
        cat_out_dims = []
        for card in cat_cardinalities:
            # small, robust rule-of-thumb
            dim = int(min(16, max(4, round(card ** 0.25) * 4)))
            self.cat_embeds.append(nn.Embedding(num_embeddings=int(card) + 1, embedding_dim=dim))
            cat_out_dims.append(dim)
        self.cat_total_dim = int(sum(cat_out_dims))

        # static forcing projection
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # project per-day inputs to u_t
        self.u_proj = nn.Sequential(
            nn.Linear(n_num + self.cat_total_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # dynamics operator f([h, u])
        self.f = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, dyn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dyn_dim, hidden_dim),
        )

        self.ln = nn.LayerNorm(hidden_dim)

        # learnable time step (positive)
        self.raw_dt = nn.Parameter(torch.tensor(0.0))  # softplus(raw_dt) ~ 0.69 initially

        # heads
        self.head_mean = nn.Linear(hidden_dim, 1)
        self.head_q20 = nn.Linear(hidden_dim, 1)
        self.head_neg = nn.Linear(hidden_dim, 1)

    def forward(self, x_num, x_cat, x_static):
        """
        x_num: (B, 7, n_num)
        x_cat: (B, 7, n_cat) long
        x_static: (B, n_static)
        """
        B, T, _ = x_num.shape

        # embed categoricals
        if self.n_cat > 0:
            embs = []
            for j, emb in enumerate(self.cat_embeds):
                embs.append(emb(x_cat[:, :, j]))
            x_cat_e = torch.cat(embs, dim=-1)  # (B, 7, cat_total_dim)
        else:
            x_cat_e = torch.zeros((B, T, 0), device=x_num.device, dtype=x_num.dtype)

        s = self.static_proj(x_static)  # (B, hidden)
        s_rep = s.unsqueeze(1).expand(-1, T, -1)

        u = self.u_proj(torch.cat([x_num, x_cat_e, s_rep], dim=-1))  # (B, 7, hidden)

        # integrate
        h = torch.zeros((B, self.hidden_dim), device=x_num.device, dtype=x_num.dtype)
        dt = torch.clamp(torch.nn.functional.softplus(self.raw_dt), 0.05, 1.5)

        energy = 0.0
        for t in range(T):
            ut = u[:, t, :]
            if self.use_rk2:
                k1 = self.f(torch.cat([h, ut], dim=-1))
                h_temp = h + dt * k1
                k2 = self.f(torch.cat([h_temp, ut], dim=-1))
                dh = 0.5 * (k1 + k2)
            else:
                dh = self.f(torch.cat([h, ut], dim=-1))

            h = h + dt * dh
            h = self.ln(h)
            energy = energy + torch.mean(dh * dh)

        energy = energy / float(T)

        mean = self.head_mean(h).squeeze(-1)
        q20 = self.head_q20(h).squeeze(-1)
        neg_logit = self.head_neg(h).squeeze(-1)
        return mean, q20, neg_logit, energy

@torch.no_grad()
def predict_physics(model, loader, num_mean, num_std, st_mean, st_std):
    model.eval()
    preds_mean = []
    preds_q20 = []
    preds_negp = []
    num_mean_t = torch.as_tensor(num_mean, device=DEVICE).view(1, 1, -1)
    num_std_t  = torch.as_tensor(num_std, device=DEVICE).view(1, 1, -1)
    st_mean_t  = torch.as_tensor(st_mean, device=DEVICE).view(1, -1)
    st_std_t   = torch.as_tensor(st_std, device=DEVICE).view(1, -1)

    for batch in loader:
        if len(batch) == 4:
            xnum, xcat, xst, _ = batch
        else:
            xnum, xcat, xst = batch

        xnum = xnum.to(DEVICE, non_blocking=True).float()
        xcat = xcat.to(DEVICE, non_blocking=True).long()
        xst = xst.to(DEVICE, non_blocking=True).float()

        xnum = (xnum - num_mean_t) / num_std_t
        xst = (xst - st_mean_t) / st_std_t

        pm, pq, nlogit, _ = model(xnum, xcat, xst)
        preds_mean.append(pm.detach().cpu().numpy())
        preds_q20.append(pq.detach().cpu().numpy())
        preds_negp.append(torch.sigmoid(nlogit).detach().cpu().numpy())

    return (np.concatenate(preds_mean).astype(np.float32),
            np.concatenate(preds_q20).astype(np.float32),
            np.concatenate(preds_negp).astype(np.float32))

def train_physics_fold(X_num_seq, X_cat_seq, X_static, y,
                       tr_idx, va_idx, test_idx,
                       cat_cardinalities,
                       seed=42,
                       hidden_dim=96, dyn_dim=192,
                       batch_size=1024, max_epochs=20, patience=3,
                       lr=1e-3, wd=1e-4,
                       lambda_q=0.25, lambda_bce=0.20, lambda_energy=1e-4,
                       tau=0.20):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # fold scalers (train fold only)
    Xn_tr = X_num_seq[tr_idx].reshape(-1, X_num_seq.shape[-1]).astype(np.float32)
    num_mean = Xn_tr.mean(axis=0)
    num_std = Xn_tr.std(axis=0) + 1e-6

    Xs_tr = X_static[tr_idx].astype(np.float32)
    st_mean = Xs_tr.mean(axis=0)
    st_std = Xs_tr.std(axis=0) + 1e-6

    y_tr = y[tr_idx].astype(np.float32)
    neg_rate = float(np.mean(y_tr < 0))
    # pos_weight in BCEWithLogitsLoss weights positive labels (negatives here, label=1)
    pos_weight = float((1.0 - neg_rate) / (neg_rate + 1e-6)) * 1.25
    pos_weight_t = torch.tensor([pos_weight], device=DEVICE)

    model = PhysicsTimeStepper(
        n_num=X_num_seq.shape[-1],
        cat_cardinalities=cat_cardinalities,
        n_static=X_static.shape[-1],
        hidden_dim=hidden_dim,
        dyn_dim=dyn_dim,
        dropout=0.10,
        use_rk2=True,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    train_ds = IndexSeqDataset(X_num_seq, X_cat_seq, X_static, y, tr_idx)
    val_ds = IndexSeqDataset(X_num_seq, X_cat_seq, X_static, y, va_idx)
    test_ds = IndexSeqDatasetNoY(X_num_seq, X_cat_seq, X_static, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    best_loss = float("inf")
    best_state = None
    bad = 0

    num_mean_t = torch.as_tensor(num_mean, device=DEVICE).view(1, 1, -1)
    num_std_t  = torch.as_tensor(num_std, device=DEVICE).view(1, 1, -1)
    st_mean_t  = torch.as_tensor(st_mean, device=DEVICE).view(1, -1)
    st_std_t   = torch.as_tensor(st_std, device=DEVICE).view(1, -1)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xnum, xcat, xst, yy in train_loader:
            xnum = xnum.to(DEVICE, non_blocking=True).float()
            xcat = xcat.to(DEVICE, non_blocking=True).long()
            xst = xst.to(DEVICE, non_blocking=True).float()
            yy = yy.to(DEVICE, non_blocking=True).float()

            xnum = (xnum - num_mean_t) / num_std_t
            xst = (xst - st_mean_t) / st_std_t

            pm, pq, nlogit, energy = model(xnum, xcat, xst)
            loss_m = awmse_torch(yy, pm)
            loss_q = pinball_loss(yy, pq, tau=tau)

            yneg = (yy < 0).float()
            loss_b = bce(nlogit, yneg)

            loss = loss_m + lambda_q * loss_q + lambda_bce * loss_b + lambda_energy * energy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xnum, xcat, xst, yy in val_loader:
                xnum = xnum.to(DEVICE, non_blocking=True).float()
                xcat = xcat.to(DEVICE, non_blocking=True).long()
                xst = xst.to(DEVICE, non_blocking=True).float()
                yy = yy.to(DEVICE, non_blocking=True).float()

                xnum = (xnum - num_mean_t) / num_std_t
                xst = (xst - st_mean_t) / st_std_t

                pm, pq, nlogit, energy = model(xnum, xcat, xst)
                loss_m = awmse_torch(yy, pm)
                loss_q = pinball_loss(yy, pq, tau=tau)
                yneg = (yy < 0).float()
                loss_b = bce(nlogit, yneg)
                loss = loss_m + lambda_q * loss_q + lambda_bce * loss_b + lambda_energy * energy
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if len(val_losses) else float("inf")
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predict OOF/TEST
    oof_mean, oof_q20, oof_negp = predict_physics(model, val_loader, num_mean, num_std, st_mean, st_std)
    te_mean, te_q20, te_negp = predict_physics(model, test_loader, num_mean, num_std, st_mean, st_std)
    return oof_mean, oof_q20, oof_negp, te_mean, te_q20, te_negp


# -----------------------------
# Meta features + calibration
# -----------------------------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)

def build_meta_features(p_aw, p_rmse, p_q20, p_q80, p_negprob,
                        p_phy_mean, p_phy_q20, p_phy_negprob):
    """
    Stack-friendly features: base predictions + uncertainty spreads + disagreement signals.
    """
    p_aw = p_aw.astype(np.float32)
    p_rmse = p_rmse.astype(np.float32)
    p_q20 = p_q20.astype(np.float32)
    p_q80 = p_q80.astype(np.float32)
    p_negprob = p_negprob.astype(np.float32)

    p_phy_mean = p_phy_mean.astype(np.float32)
    p_phy_q20 = p_phy_q20.astype(np.float32)
    p_phy_negprob = p_phy_negprob.astype(np.float32)

    spread_lgb = (p_q80 - p_q20).astype(np.float32)
    d_aw_q20 = (p_aw - p_q20).astype(np.float32)
    d_q80_aw = (p_q80 - p_aw).astype(np.float32)
    abs_aw_rmse = np.abs(p_aw - p_rmse).astype(np.float32)

    spread_cross = (p_aw - p_phy_mean).astype(np.float32)
    abs_cross = np.abs(spread_cross).astype(np.float32)
    phy_gap = (p_phy_mean - p_phy_q20).astype(np.float32)

    X = np.stack([
        p_aw, p_rmse, p_q20, p_q80, p_negprob,
        _logit(p_negprob),
        spread_lgb, d_aw_q20, d_q80_aw, abs_aw_rmse,

        p_phy_mean, p_phy_q20, p_phy_negprob,
        _logit(p_phy_negprob),
        abs_cross, spread_cross, phy_gap,
        (p_phy_q20 - p_q20).astype(np.float32),
        (p_phy_mean - p_aw).astype(np.float32),
    ], axis=1).astype(np.float32)
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
    print("Physics-inspired time-stepping DNN (RK2) + LGB ensemble + stacking + affine calibration")
    print("=" * 60)
    print(f"DEVICE = {DEVICE}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # --- user-level tabular features for GBDT and as static forcing for physics net
    train_f = create_features(train_df, is_train=True)
    val_f   = create_features(val_df, is_train=True)
    test_f  = create_features(test_df, is_train=False)

    full_f = pd.concat([train_f, val_f], axis=0, ignore_index=True)
    y_full = full_f[TARGET_COL].values.astype(np.float32)

    # target encode categoricals (train+val only), apply to test
    cat_cols_user = full_f.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols_user = [c for c in cat_cols_user if c not in [ID_COL, TARGET_COL]]
    full_f, test_f = add_target_encodings(full_f, test_f, cat_cols_user, n_splits=5, smoothing=35.0, seed=SEED)

    feat_cols = [c for c in full_f.columns if c not in [ID_COL, TARGET_COL]]
    X_full = full_f[feat_cols].values.astype(np.float32)
    X_test = test_f[feat_cols].values.astype(np.float32)
    full_ids = full_f[ID_COL].values
    test_ids = test_f[ID_COL].values

    print(f"Users: full_train={len(full_f)}, test={len(test_f)}")
    print(f"Static features = {len(feat_cols)}")

    # --- build 7-day sequences for physics model from row-level data (train+val only for maps)
    full_rows = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    num_cols_seq, cat_cols_seq = _detect_cols(full_rows)
    # ensure day exists and is numeric
    num_cols_seq = [c for c in num_cols_seq if c not in [DAY_COL]]
    cat_cols_seq = [c for c in cat_cols_seq if c not in [DAY_COL]]

    cat_maps = build_category_maps(full_rows, cat_cols_seq)
    cat_cardinalities = [int(len(cat_maps[c])) for c in cat_cols_seq]

    X_num_full, X_cat_full = build_user_sequences(full_rows, full_ids, num_cols_seq, cat_cols_seq, cat_maps)
    X_num_test, X_cat_test = build_user_sequences(test_df, test_ids, num_cols_seq, cat_cols_seq, cat_maps)

    # static forcing for physics model
    X_static_full = X_full.astype(np.float32)
    X_static_test = X_test.astype(np.float32)

    print(f"Sequence numeric cols = {len(num_cols_seq)}, sequence cat cols = {len(cat_cols_seq)}")
    print(f"X_num_full shape = {X_num_full.shape}, X_static_full shape = {X_static_full.shape}")

    # -----------------------------
    # CV bagging
    # -----------------------------
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_aw = np.zeros(len(full_f), dtype=np.float32)
    oof_rmse = np.zeros(len(full_f), dtype=np.float32)
    oof_q20 = np.zeros(len(full_f), dtype=np.float32)
    oof_q80 = np.zeros(len(full_f), dtype=np.float32)
    oof_negp = np.zeros(len(full_f), dtype=np.float32)

    oof_phy_mean = np.zeros(len(full_f), dtype=np.float32)
    oof_phy_q20 = np.zeros(len(full_f), dtype=np.float32)
    oof_phy_negp = np.zeros(len(full_f), dtype=np.float32)

    test_aw = np.zeros(len(test_f), dtype=np.float32)
    test_rmse = np.zeros(len(test_f), dtype=np.float32)
    test_q20 = np.zeros(len(test_f), dtype=np.float32)
    test_q80 = np.zeros(len(test_f), dtype=np.float32)
    test_negp = np.zeros(len(test_f), dtype=np.float32)

    test_phy_mean = np.zeros(len(test_f), dtype=np.float32)
    test_phy_q20 = np.zeros(len(test_f), dtype=np.float32)
    test_phy_negp = np.zeros(len(test_f), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
        X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
        X_va, y_va = X_full[va_idx], y_full[va_idx]

        seed_fold = SEED + fold * 17
        print(f"\n--- Fold {fold}/{n_splits} ---")

        # LightGBM base learners
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

        # Physics-inspired DNN learner (sequence)
        # Build fold-local arrays by concatenating full+test into one tensor space to reuse the same trainer API
        # We'll train on full indices and predict on val fold + test indices (offset by len(full)).
        X_num_all = np.concatenate([X_num_full, X_num_test], axis=0)
        X_cat_all = np.concatenate([X_cat_full, X_cat_test], axis=0)
        X_static_all = np.concatenate([X_static_full, X_static_test], axis=0)
        y_all = np.concatenate([y_full, np.zeros(len(test_f), dtype=np.float32)], axis=0)

        test_idx = np.arange(len(full_f), len(full_f) + len(test_f), dtype=np.int64)

        phy_oof_mean, phy_oof_q20, phy_oof_negp, phy_te_mean, phy_te_q20, phy_te_negp = train_physics_fold(
            X_num_all, X_cat_all, X_static_all, y_all,
            tr_idx=tr_idx, va_idx=va_idx, test_idx=test_idx,
            cat_cardinalities=cat_cardinalities,
            seed=seed_fold + 10,
            hidden_dim=96, dyn_dim=192,
            batch_size=1024, max_epochs=20, patience=3,
            lr=1e-3, wd=1e-4,
            lambda_q=0.25, lambda_bce=0.20, lambda_energy=1e-4,
            tau=0.20
        )

        oof_phy_mean[va_idx] = phy_oof_mean
        oof_phy_q20[va_idx] = phy_oof_q20
        oof_phy_negp[va_idx] = phy_oof_negp

        test_phy_mean += phy_te_mean / n_splits
        test_phy_q20 += phy_te_q20 / n_splits
        test_phy_negp += phy_te_negp / n_splits

    # -----------------------------
    # Meta learner (stacking) trained on OOF
    # -----------------------------
    X_meta_oof = build_meta_features(
        oof_aw, oof_rmse, oof_q20, oof_q80, oof_negp,
        oof_phy_mean, oof_phy_q20, oof_phy_negp
    )
    X_meta_test = build_meta_features(
        test_aw, test_rmse, test_q20, test_q80, test_negp,
        test_phy_mean, test_phy_q20, test_phy_negp
    )

    scaler = StandardScaler()
    X_meta_oof_s = scaler.fit_transform(X_meta_oof).astype(np.float32)
    X_meta_test_s = scaler.transform(X_meta_test).astype(np.float32)

    meta = RidgeCV(alphas=np.logspace(-3, 3, 31), fit_intercept=True)
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
        "method": "Physics-inspired RK2 time-stepping DNN + LGB ensemble + Ridge stacking + affine calibration",
        "score": float(score),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "meta_alpha": float(getattr(meta, "alpha_", np.nan)),
        "affine_a": float(a),
        "affine_b": float(b),
        "n_static_features": int(len(feat_cols)),
        "n_seq_num_features": int(len(num_cols_seq)),
        "n_seq_cat_features": int(len(cat_cols_seq)),
        "n_train_full": int(len(full_f)),
        "device": str(DEVICE),
    }
    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/physics_timestep_results.json"
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
    print(f"score = {score}")