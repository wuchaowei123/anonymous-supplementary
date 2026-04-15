#!/usr/bin/env python3
"""
AviaAgentMonty - Execution Node: 59394c83
Type: original
Generated: 2026-01-13T12:50:31.332905
Mutation: replicate
Parent: None

DO NOT DELETE - This file is preserved for reproducibility.
"""
# Suppress warnings to prevent false failures
import warnings
warnings.filterwarnings("ignore")

#!/usr/bin/env python3
"""
Physics-Inspired Deep Time-Stepping (RK2 Neural ODE / ResNet limit) + Leak-free Fold Target Encoding + LightGBM + Meta + Conservative Calibration

Implements key ideas aligned with "Deep Learning and Computational Physics (Lecture Notes)":
- Residual networks as explicit time-stepping schemes (ODE viewpoint).
- Stable explicit integrator (RK2 / midpoint) across 7 daily steps (dt=1).
- Regularization inspired by numerical stability: penalize large step-to-step state changes ("smooth dynamics").

Adaptation to this task:
- Use the raw 7-day rows per user as a short time sequence.
- Multi-head outputs to address signed/zero-inflated targets:
  - sign logits (neg/zero/pos),
  - conservative quantiles (tau=0.2 and 0.8),
  - positive/negative magnitudes (softplus),
  - mixture prediction via soft gating.
- Asymmetric business cost handled via AWMSE-style loss inside the deep model.
- Keep strong LightGBM baselines (leak-free fold target encoding).
- Meta learner blends GBDT + Deep physics model outputs.
- Final post-hoc calibration: affine (mean match) + optional conservative shift tuned on train/val only.
- CRITICAL: scoring/metric functions are preserved EXACTLY.

NOTE: Do NOT use test labels for any model selection. Test labels are used only at the end for required scoring.
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# curated temporal columns used by create_features (kept)
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

def compute_pareto_multi_objective(y_true, y_pred):
    return compute_score(y_true, y_pred)[0]


# -----------------------------
# Helpers: feature engineering (tabular user-level) (kept baseline)
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
    - First-observed categorical snapshot (target-encoded per fold later)
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


# -----------------------------
# Fold target encoding (leak-free)
# -----------------------------
def fold_target_encode(tr_df, other_df, cat_cols, smoothing=35.0):
    """
    Fit TE on tr_df only (uses TARGET_COL), apply to other_df.
    Returns: (tr_encoded, other_encoded)
    Adds:
      - __te_mean, __te_neg_rate, __te_zero_rate, __cnt, __logcnt, __freq
    Drops raw categorical columns.
    """
    if len(cat_cols) == 0:
        return tr_df.copy(), other_df.copy()

    tr = tr_df.copy()
    ot = other_df.copy()

    y = tr[TARGET_COL].values.astype(np.float32)
    global_mean = float(np.mean(y))
    global_neg = float(np.mean(y < 0))
    global_zero = float(np.mean(y == 0))

    tr["__is_neg"] = (y < 0).astype(np.float32)
    tr["__is_zero"] = (y == 0).astype(np.float32)

    for c in cat_cols:
        tr[c] = tr[c].astype("object").fillna("__MISSING__")
        ot[c] = ot[c].astype("object").fillna("__MISSING__")

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

        tr_cnt = tr[c].map(cnt).fillna(0.0).astype(np.float32).values
        ot_cnt = ot[c].map(cnt).fillna(0.0).astype(np.float32).values

        tr[f"{c}__te_mean"] = tr[c].map(sm_mean).fillna(global_mean).astype(np.float32).values
        tr[f"{c}__te_neg_rate"] = tr[c].map(sm_neg).fillna(global_neg).astype(np.float32).values
        tr[f"{c}__te_zero_rate"] = tr[c].map(sm_zero).fillna(global_zero).astype(np.float32).values
        tr[f"{c}__cnt"] = tr_cnt
        tr[f"{c}__logcnt"] = np.log1p(tr_cnt).astype(np.float32)
        tr[f"{c}__freq"] = tr[c].map(freq).fillna(0.0).astype(np.float32).values

        ot[f"{c}__te_mean"] = ot[c].map(sm_mean).fillna(global_mean).astype(np.float32).values
        ot[f"{c}__te_neg_rate"] = ot[c].map(sm_neg).fillna(global_neg).astype(np.float32).values
        ot[f"{c}__te_zero_rate"] = ot[c].map(sm_zero).fillna(global_zero).astype(np.float32).values
        ot[f"{c}__cnt"] = ot_cnt
        ot[f"{c}__logcnt"] = np.log1p(ot_cnt).astype(np.float32)
        ot[f"{c}__freq"] = ot[c].map(freq).fillna(0.0).astype(np.float32).values

    tr = tr.drop(columns=["__is_neg", "__is_zero"] + cat_cols, errors="ignore")
    ot = ot.drop(columns=cat_cols, errors="ignore")
    tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0)
    ot = ot.replace([np.inf, -np.inf], np.nan).fillna(0)
    return tr, ot


# -----------------------------
# LightGBM (keep a smaller strong suite)
# -----------------------------
def train_lgb_aw(X_tr, y_tr, X_va, y_va, seed=42):
    # Approximate asymmetric weighting via sample weights (static in y)
    w_tr = np.ones_like(y_tr, dtype=np.float32)
    fp = (y_tr < 0)
    fn = (y_tr > 0)
    w_tr[fp] = 2.5 + 0.02 * np.abs(y_tr[fp])
    w_tr[fn] = 1.5 + 0.01 * y_tr[fn]
    w_tr = np.clip(w_tr, 0.1, 25.0)

    w_va = np.ones_like(y_va, dtype=np.float32)
    fp_va = (y_va < 0)
    fn_va = (y_va > 0)
    w_va[fp_va] = 2.5 + 0.02 * np.abs(y_va[fp_va])
    w_va[fn_va] = 1.5 + 0.01 * y_va[fn_va]
    w_va = np.clip(w_va, 0.1, 25.0)

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=192,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.4,
        max_bin=255,
        verbose=-1,
        seed=seed,
        num_threads=-1,
        force_col_wise=True,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr.astype(np.float32), weight=w_tr)
    dval = lgb.Dataset(X_va, label=y_va.astype(np.float32), weight=w_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=8000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
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
        min_data_in_leaf=120,
        feature_fraction=0.80,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=0.6,
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
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=250, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


# -----------------------------
# Physics-inspired deep sequence model (RK2 time-stepping)
# -----------------------------
def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return np.sign(x) * np.log1p(np.abs(x))

def build_category_maps(df_list, cat_cols):
    """
    Build category->index maps from train+val only (no target usage).
    Index 0 reserved for unknown/missing.
    """
    maps = {}
    for c in cat_cols:
        vals = []
        for df in df_list:
            if c in df.columns:
                v = df[c].astype("object").fillna("__MISSING__").astype(str).values
                vals.append(v)
        if len(vals) == 0:
            maps[c] = {"__MISSING__": 1}
            continue
        allv = np.concatenate(vals, axis=0)
        uniq = pd.unique(allv)
        mapping = {k: i + 1 for i, k in enumerate(uniq.tolist())}  # start at 1
        if "__MISSING__" not in mapping:
            mapping["__MISSING__"] = len(mapping) + 1
        maps[c] = mapping
    return maps

def encode_cats(df, cat_cols, cat_maps):
    out = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        if c not in df.columns:
            continue
        s = df[c].astype("object").fillna("__MISSING__").astype(str)
        mp = cat_maps[c]
        # fast-ish mapping via pandas .map on Series
        codes = s.map(mp).fillna(0).astype(np.int64).values
        out[:, j] = codes
    return out

def make_aligned_sequences(df_daily, user_ids, num_cols, cat_cols, cat_maps, scaler, add_time_features=True):
    """
    Create aligned tensors:
      X_num: (N, 7, F_num)
      X_cat: (N, 7, F_cat)
    where order matches user_ids.
    """
    idx = pd.Index(user_ids)
    df = df_daily.copy()

    # ensure clean daily grid (dedupe per user/day)
    df = df.sort_values([ID_COL, DAY_COL])
    df = df.drop_duplicates(subset=[ID_COL, DAY_COL], keep="first")

    # filter to users in user_ids
    pos = idx.get_indexer(df[ID_COL].values)
    mask = pos >= 0
    df = df.loc[mask].copy()
    pos = pos[mask]

    day = df[DAY_COL].values.astype(np.int64) - 1
    day = np.clip(day, 0, 6)

    # numeric
    Xn = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    Xn = signed_log1p(Xn)
    Xn = scaler.transform(Xn).astype(np.float32)

    # optional time features as in computational physics (explicit time coordinate)
    if add_time_features:
        t = (day.astype(np.float32) / 6.0) * 2.0 - 1.0  # [-1,1]
        t_sin = np.sin(np.pi * t).astype(np.float32)
        t_cos = np.cos(np.pi * t).astype(np.float32)
        Xn = np.concatenate([Xn, t_sin[:, None], t_cos[:, None]], axis=1).astype(np.float32)

    # categorical
    Xc = encode_cats(df, cat_cols, cat_maps)  # (rows, C)

    N = len(user_ids)
    Fnum = Xn.shape[1]
    Fcat = Xc.shape[1]

    X_num = np.zeros((N, 7, Fnum), dtype=np.float32)
    X_cat = np.zeros((N, 7, Fcat), dtype=np.int64)

    # scatter into (user, day)
    X_num[pos, day, :] = Xn
    X_cat[pos, day, :] = Xc
    return X_num, X_cat

class SeqDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, i):
        if self.y is None:
            return self.X_num[i], self.X_cat[i]
        return self.X_num[i], self.X_cat[i], self.y[i]

class RK2TimeStepperNet(nn.Module):
    """
    Physics-inspired: hidden state evolves by explicit RK2 time stepping:
        h_{t+1} = h_t + dt * f(h_t + 0.5*dt*f(h_t,u_t), u_t)
    with shared f across time (autonomous dynamics + forcing u_t).
    """
    def __init__(self, n_num, cat_cardinalities, d_model=192, d_state=192, f_hidden=256, dropout=0.10):
        super().__init__()
        self.n_num = n_num
        self.cat_cardinalities = cat_cardinalities

        # entity embeddings for categorical columns
        self.embs = nn.ModuleList()
        emb_dims = []
        for card in cat_cardinalities:
            # heuristic: small embeddings for stability
            dim = int(np.clip(np.ceil(np.log2(max(card, 2)) + 1), 3, 12))
            self.embs.append(nn.Embedding(card + 1, dim, padding_idx=0))
            emb_dims.append(dim)
        self.emb_out = int(sum(emb_dims))
        in_dim = n_num + self.emb_out

        self.u_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.h0 = nn.Parameter(torch.zeros(d_state))

        self.f_net = nn.Sequential(
            nn.Linear(d_state + d_model, f_hidden),
            nn.LayerNorm(f_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(f_hidden, d_state),
        )
        self.dt = 1.0

        # readout uses last + mean states (short sequence)
        z_dim = 2 * d_state

        self.head_sign = nn.Linear(z_dim, 3)     # (neg, zero, pos)
        self.head_pos  = nn.Linear(z_dim, 1)     # positive magnitude (softplus)
        self.head_neg  = nn.Linear(z_dim, 1)     # negative magnitude (softplus)
        self.head_q20  = nn.Linear(z_dim, 1)     # conservative quantile
        self.head_q80  = nn.Linear(z_dim, 1)     # upper quantile

    def _f(self, h, u):
        return self.f_net(torch.cat([h, u], dim=-1))

    def _rk2_step(self, h, u):
        k1 = self._f(h, u)
        k2 = self._f(h + 0.5 * self.dt * k1, u)
        return h + self.dt * k2

    def forward(self, x_num, x_cat):
        """
        x_num: (B, 7, n_num)
        x_cat: (B, 7, n_cat)
        returns:
          pred_mean, q20, q80, p_neg, p_zero, smooth_reg
        """
        B, T, _ = x_num.shape

        # cat embeddings per day
        embs = []
        for j, emb in enumerate(self.embs):
            embs.append(emb(x_cat[:, :, j]))
        x_cat_e = torch.cat(embs, dim=-1) if len(embs) else None

        x_in = torch.cat([x_num, x_cat_e], dim=-1) if x_cat_e is not None else x_num
        u = self.u_proj(x_in)  # (B,T,d_model)

        h = self.h0[None, :].expand(B, -1)  # (B,d_state)
        states = []
        for t in range(T):
            h = self._rk2_step(h, u[:, t, :])
            states.append(h)
        H = torch.stack(states, dim=1)      # (B,T,d_state)
        h_last = H[:, -1, :]
        h_mean = H.mean(dim=1)
        z = torch.cat([h_last, h_mean], dim=-1)

        logits = self.head_sign(z)  # (B,3)
        probs = F.softmax(logits, dim=1)
        p_neg = probs[:, 0]
        p_zero = probs[:, 1]
        p_pos = probs[:, 2]

        pos_mag = F.softplus(self.head_pos(z).squeeze(1))
        neg_mag = F.softplus(self.head_neg(z).squeeze(1))

        # mixture prediction
        pred = p_pos * pos_mag - p_neg * neg_mag

        q20 = self.head_q20(z).squeeze(1)
        q80 = self.head_q80(z).squeeze(1)

        # numerical-stability-inspired smoothness (penalize violent dynamics)
        dH = H[:, 1:, :] - H[:, :-1, :]
        smooth = (dH ** 2).mean()

        return pred, q20, q80, logits, smooth

def awmse_torch(y, yhat):
    """
    Asymmetric Weighted MSE with weights depending on sign mismatch (yhat vs y).
    """
    w = torch.ones_like(yhat)
    w = torch.where((yhat > 0) & (y < 0), 2.5 + 0.02 * torch.abs(y), w)
    w = torch.where((yhat < 0) & (y > 0), 1.5 + 0.01 * y, w)
    w = torch.clamp(w, 0.1, 25.0)
    return (w * (yhat - y) ** 2).mean()

def pinball_loss(y, q, tau):
    e = y - q
    return torch.maximum(tau * e, (tau - 1.0) * e).mean()

def train_physics_model(
    Xn_tr, Xc_tr, y_tr,
    Xn_va, Xc_va, y_va,
    cat_cardinalities,
    seed=42,
    max_epochs=25,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    patience=4
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RK2TimeStepperNet(
        n_num=Xn_tr.shape[-1],
        cat_cardinalities=cat_cardinalities,
        d_model=192,
        d_state=192,
        f_hidden=256,
        dropout=0.10,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = np.inf
    best_state = None
    bad = 0

    tr_ds = SeqDataset(Xn_tr, Xc_tr, y_tr)
    va_ds = SeqDataset(Xn_va, Xc_va, y_va)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # cost-sensitive neg vs non-neg BCE from sign logits
    y_tr_neg = (y_tr < 0).astype(np.float32)
    pos_weight = float((1.0 - y_tr_neg.mean()) / (y_tr_neg.mean() + 1e-6))
    pos_weight = float(np.clip(pos_weight * 1.20, 1.0, 8.0))
    bce_neg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        ntr = 0

        for xb_num, xb_cat, yb in tr_ld:
            xb_num = xb_num.to(device=device, dtype=torch.float32)
            xb_cat = xb_cat.to(device=device, dtype=torch.long)
            yb = yb.to(device=device, dtype=torch.float32)

            pred, q20, q80, logits, smooth = model(xb_num, xb_cat)

            # neg logit as log-odds neg vs (zero/pos)
            logit_neg = logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1)
            y_neg = (yb < 0).float()

            loss_reg = awmse_torch(yb, pred)
            loss_q = 0.60 * pinball_loss(yb, q20, 0.20) + 0.20 * pinball_loss(yb, q80, 0.80)
            loss_cls = bce_neg(logit_neg, y_neg)
            loss_smooth = smooth

            loss = loss_reg + 0.20 * loss_q + 0.20 * loss_cls + 1e-3 * loss_smooth

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

            bs = int(yb.shape[0])
            tr_loss += float(loss.detach().cpu().item()) * bs
            ntr += bs

        tr_loss /= max(ntr, 1)

        model.eval()
        va_loss = 0.0
        nva = 0
        with torch.no_grad():
            for xb_num, xb_cat, yb in va_ld:
                xb_num = xb_num.to(device=device, dtype=torch.float32)
                xb_cat = xb_cat.to(device=device, dtype=torch.long)
                yb = yb.to(device=device, dtype=torch.float32)

                pred, q20, q80, logits, smooth = model(xb_num, xb_cat)
                logit_neg = logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1)
                y_neg = (yb < 0).float()

                loss_reg = awmse_torch(yb, pred)
                loss_q = 0.60 * pinball_loss(yb, q20, 0.20) + 0.20 * pinball_loss(yb, q80, 0.80)
                loss_cls = bce_neg(logit_neg, y_neg)
                loss_smooth = smooth
                loss = loss_reg + 0.20 * loss_q + 0.20 * loss_cls + 1e-3 * loss_smooth

                bs = int(yb.shape[0])
                va_loss += float(loss.detach().cpu().item()) * bs
                nva += bs

        va_loss /= max(nva, 1)

        if va_loss + 1e-6 < best:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def predict_physics_model(model, Xn, Xc, batch_size=1024):
    device = next(model.parameters()).device
    ds = SeqDataset(Xn, Xc, y=None)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    preds = []
    q20s = []
    q80s = []
    p_negs = []
    p_zeros = []

    for xb_num, xb_cat in ld:
        xb_num = xb_num.to(device=device, dtype=torch.float32)
        xb_cat = xb_cat.to(device=device, dtype=torch.long)

        pred, q20, q80, logits, _smooth = model(xb_num, xb_cat)
        probs = F.softmax(logits, dim=1)
        p_neg = probs[:, 0]
        p_zero = probs[:, 1]

        preds.append(pred.detach().cpu().numpy().astype(np.float32))
        q20s.append(q20.detach().cpu().numpy().astype(np.float32))
        q80s.append(q80.detach().cpu().numpy().astype(np.float32))
        p_negs.append(p_neg.detach().cpu().numpy().astype(np.float32))
        p_zeros.append(p_zero.detach().cpu().numpy().astype(np.float32))

    return (
        np.concatenate(preds),
        np.concatenate(q20s),
        np.concatenate(q80s),
        np.concatenate(p_negs),
        np.concatenate(p_zeros),
    )


# -----------------------------
# Meta features + calibration
# -----------------------------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)

def build_meta_matrix(features: dict):
    """
    Flexible meta feature builder: stack all provided base predictions and derived interactions.
    """
    # required keys (some may be None, but should exist)
    def f(name):
        v = features.get(name, None)
        if v is None:
            raise ValueError(f"Missing meta feature: {name}")
        return v.astype(np.float32)

    lgb_aw = f("lgb_aw")
    lgb_q20 = f("lgb_q20")
    lgb_q80 = f("lgb_q80")
    lgb_negp = np.clip(f("lgb_negp"), 1e-6, 1 - 1e-6)

    dnn_mean = f("dnn_mean")
    dnn_q20 = f("dnn_q20")
    dnn_q80 = f("dnn_q80")
    dnn_negp = np.clip(f("dnn_negp"), 1e-6, 1 - 1e-6)
    dnn_zerop = np.clip(f("dnn_zerop"), 1e-6, 1 - 1e-6)

    spread_lgb = (lgb_q80 - lgb_q20).astype(np.float32)
    spread_dnn = (dnn_q80 - dnn_q20).astype(np.float32)

    X = np.stack([
        lgb_aw, lgb_q20, lgb_q80, spread_lgb,
        lgb_negp, _logit(lgb_negp),

        dnn_mean, dnn_q20, dnn_q80, spread_dnn,
        dnn_negp, _logit(dnn_negp),
        dnn_zerop, _logit(dnn_zerop),

        (lgb_aw - lgb_q20).astype(np.float32),
        (dnn_mean - dnn_q20).astype(np.float32),
        np.abs(lgb_aw - dnn_mean).astype(np.float32),
        0.5 * (lgb_aw + dnn_mean).astype(np.float32),
        0.5 * (lgb_q20 + dnn_q20).astype(np.float32),
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

def align_test_distribution(p_test, ref_mean, ref_std, clip_scale=(0.6, 1.6)):
    """
    Rank-preserving linear map to match (mean,std) to reference.
    Helps Error Rate stability under mean/scale drift without using y_test.
    """
    p_test = p_test.astype(np.float32)
    mu = float(np.mean(p_test))
    sd = float(np.std(p_test))
    if sd < 1e-6:
        sd = 1e-6
    scale = float(ref_std / sd)
    scale = float(np.clip(scale, clip_scale[0], clip_scale[1]))
    out = (p_test - mu) * scale + float(ref_mean)
    return out.astype(np.float32)

def train_meta_lgb(X_meta_oof, y, seed=42):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_meta_oof, y.astype(np.float32),
        test_size=0.12, random_state=seed, shuffle=True
    )
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=32,
        max_depth=5,
        min_data_in_leaf=250,
        feature_fraction=0.95,
        bagging_fraction=0.90,
        bagging_freq=1,
        lambda_l2=2.0,
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
        num_boost_round=4000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model

def tune_conservative_shift(y, p, deltas=None, seed=42):
    """
    Tune a global downward shift delta on a train-only split to reduce risk near zero.
    Uses compute_score on the split (NO test usage).
    """
    if deltas is None:
        deltas = np.linspace(-250, 100, 36, dtype=np.float32)

    X_tr, X_va, y_tr, y_va = train_test_split(
        p.astype(np.float32), y.astype(np.float32),
        test_size=0.20, random_state=seed, shuffle=True
    )
    best_delta = 0.0
    best_sc = -1e9
    for d in deltas:
        sc, _ = compute_score(y_va, (X_va - d).astype(np.float32))
        if sc > best_sc:
            best_sc = sc
            best_delta = float(d)
    return best_delta


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print("Physics-inspired RK2 time-stepping DNN + Leak-free TE LightGBM + LGB meta + conservative calibration")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # -----------------------------
    # 1) User-level features for LightGBM (baseline, strong)
    # -----------------------------
    train_f = create_features(train_df, is_train=True)
    val_f   = create_features(val_df, is_train=True)
    test_f  = create_features(test_df, is_train=False)

    full_f = pd.concat([train_f, val_f], axis=0, ignore_index=True)
    y_full = full_f[TARGET_COL].values.astype(np.float32)
    full_ids = full_f[ID_COL].values
    test_ids = test_f[ID_COL].values

    cat_cols_user = full_f.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols_user = [c for c in cat_cols_user if c not in [ID_COL, TARGET_COL]]

    print(f"Users: full_train={len(full_f)}, test={len(test_f)}")
    print(f"Raw user-level categorical cols (for TE) = {len(cat_cols_user)}")

    # -----------------------------
    # 2) Build deep sequence tensors (from daily rows) aligned to full_ids/test_ids
    #    (No target leakage: only uses X columns and train+val distribution for scalers/mappings)
    # -----------------------------
    full_daily = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    # detect daily numeric/cat
    daily_num_cols = full_daily.select_dtypes(include=[np.number]).columns.tolist()
    daily_num_cols = [c for c in daily_num_cols if c not in [TARGET_COL, ID_COL, DAY_COL]]
    daily_cat_cols = full_daily.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    daily_cat_cols = [c for c in daily_cat_cols if c not in [TARGET_COL, ID_COL, DAY_COL]]

    print(f"Daily numeric cols for DNN = {len(daily_num_cols)}, daily categorical cols for DNN = {len(daily_cat_cols)}")

    # Fit scaler on train+val daily X only
    Xn_fit = full_daily[daily_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    Xn_fit = signed_log1p(Xn_fit)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(Xn_fit)

    # Categorical maps from train+val only
    cat_maps = build_category_maps([train_df, val_df], daily_cat_cols)
    cat_cardinalities = [max(cat_maps[c].values()) for c in daily_cat_cols]  # max index

    Xn_full, Xc_full = make_aligned_sequences(
        full_daily, full_ids, daily_num_cols, daily_cat_cols, cat_maps, scaler, add_time_features=True
    )
    Xn_test, Xc_test = make_aligned_sequences(
        test_df, test_ids, daily_num_cols, daily_cat_cols, cat_maps, scaler, add_time_features=True
    )

    # -----------------------------
    # 3) LightGBM CV (leak-free TE) -> base predictions
    # -----------------------------
    n_splits_lgb = 5
    kf = KFold(n_splits=n_splits_lgb, shuffle=True, random_state=SEED)

    oof_lgb_aw = np.zeros(len(full_f), dtype=np.float32)
    oof_lgb_q20 = np.zeros(len(full_f), dtype=np.float32)
    oof_lgb_q80 = np.zeros(len(full_f), dtype=np.float32)
    oof_lgb_negp = np.zeros(len(full_f), dtype=np.float32)

    test_lgb_aw = np.zeros(len(test_f), dtype=np.float32)
    test_lgb_q20 = np.zeros(len(test_f), dtype=np.float32)
    test_lgb_q80 = np.zeros(len(test_f), dtype=np.float32)
    test_lgb_negp = np.zeros(len(test_f), dtype=np.float32)

    feature_cols_final = None

    for fold, (tr_idx, va_idx) in enumerate(kf.split(full_f), 1):
        seed_fold = SEED + fold * 17
        print(f"\n--- LGB Fold {fold}/{n_splits_lgb} ---")

        tr_raw = full_f.iloc[tr_idx].copy()
        va_raw = full_f.iloc[va_idx].copy()
        te_raw = test_f.copy()

        tr_enc, va_enc = fold_target_encode(tr_raw, va_raw, cat_cols_user, smoothing=35.0)
        _, te_enc = fold_target_encode(tr_raw, te_raw, cat_cols_user, smoothing=35.0)

        feat_cols = [c for c in tr_enc.columns if c not in [ID_COL, TARGET_COL]]
        if feature_cols_final is None:
            feature_cols_final = feat_cols
        else:
            feat_cols = feature_cols_final
            for df_ in (tr_enc, va_enc, te_enc):
                for c in feat_cols:
                    if c not in df_.columns:
                        df_[c] = 0.0

        X_tr = tr_enc[feat_cols].values.astype(np.float32)
        y_tr = tr_enc[TARGET_COL].values.astype(np.float32)
        X_va = va_enc[feat_cols].values.astype(np.float32)
        y_va = va_enc[TARGET_COL].values.astype(np.float32)
        X_te = te_enc[feat_cols].values.astype(np.float32)

        m_aw = train_lgb_aw(X_tr, y_tr, X_va, y_va, seed=seed_fold)
        p_va_aw = m_aw.predict(X_va, num_iteration=m_aw.best_iteration).astype(np.float32)
        p_te_aw = m_aw.predict(X_te, num_iteration=m_aw.best_iteration).astype(np.float32)

        m_q20 = train_lgb_quantile(X_tr, y_tr, X_va, y_va, alpha=0.20, seed=seed_fold + 1)
        p_va_q20 = m_q20.predict(X_va, num_iteration=m_q20.best_iteration).astype(np.float32)
        p_te_q20 = m_q20.predict(X_te, num_iteration=m_q20.best_iteration).astype(np.float32)

        m_q80 = train_lgb_quantile(X_tr, y_tr, X_va, y_va, alpha=0.80, seed=seed_fold + 2)
        p_va_q80 = m_q80.predict(X_va, num_iteration=m_q80.best_iteration).astype(np.float32)
        p_te_q80 = m_q80.predict(X_te, num_iteration=m_q80.best_iteration).astype(np.float32)

        m_neg = train_lgb_neg_classifier(X_tr, y_tr, X_va, y_va, seed=seed_fold + 3)
        p_va_neg = m_neg.predict(X_va, num_iteration=m_neg.best_iteration).astype(np.float32)
        p_te_neg = m_neg.predict(X_te, num_iteration=m_neg.best_iteration).astype(np.float32)

        oof_lgb_aw[va_idx] = p_va_aw
        oof_lgb_q20[va_idx] = p_va_q20
        oof_lgb_q80[va_idx] = p_va_q80
        oof_lgb_negp[va_idx] = p_va_neg

        test_lgb_aw += p_te_aw / n_splits_lgb
        test_lgb_q20 += p_te_q20 / n_splits_lgb
        test_lgb_q80 += p_te_q80 / n_splits_lgb
        test_lgb_negp += p_te_neg / n_splits_lgb

    # -----------------------------
    # 4) Physics-inspired DNN CV (fewer folds for speed)
    # -----------------------------
    n_splits_dnn = 3
    kf_dnn = KFold(n_splits=n_splits_dnn, shuffle=True, random_state=SEED + 123)

    oof_dnn_mean = np.zeros(len(full_f), dtype=np.float32)
    oof_dnn_q20 = np.zeros(len(full_f), dtype=np.float32)
    oof_dnn_q80 = np.zeros(len(full_f), dtype=np.float32)
    oof_dnn_negp = np.zeros(len(full_f), dtype=np.float32)
    oof_dnn_zerop = np.zeros(len(full_f), dtype=np.float32)

    test_dnn_mean = np.zeros(len(test_f), dtype=np.float32)
    test_dnn_q20 = np.zeros(len(test_f), dtype=np.float32)
    test_dnn_q80 = np.zeros(len(test_f), dtype=np.float32)
    test_dnn_negp = np.zeros(len(test_f), dtype=np.float32)
    test_dnn_zerop = np.zeros(len(test_f), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf_dnn.split(full_f), 1):
        seed_fold = SEED + 900 + fold * 31
        print(f"\n--- DNN Fold {fold}/{n_splits_dnn} ---")

        Xn_tr = Xn_full[tr_idx]
        Xc_tr = Xc_full[tr_idx]
        y_tr = y_full[tr_idx]

        Xn_va = Xn_full[va_idx]
        Xc_va = Xc_full[va_idx]
        y_va = y_full[va_idx]

        model = train_physics_model(
            Xn_tr, Xc_tr, y_tr,
            Xn_va, Xc_va, y_va,
            cat_cardinalities=cat_cardinalities,
            seed=seed_fold,
            max_epochs=25,
            batch_size=512,
            lr=1e-3,
            weight_decay=1e-4,
            patience=4,
        )

        p_va, q20_va, q80_va, negp_va, zerop_va = predict_physics_model(model, Xn_va, Xc_va, batch_size=1024)
        p_te, q20_te, q80_te, negp_te, zerop_te = predict_physics_model(model, Xn_test, Xc_test, batch_size=1024)

        oof_dnn_mean[va_idx] = p_va
        oof_dnn_q20[va_idx] = q20_va
        oof_dnn_q80[va_idx] = q80_va
        oof_dnn_negp[va_idx] = negp_va
        oof_dnn_zerop[va_idx] = zerop_va

        test_dnn_mean += p_te / n_splits_dnn
        test_dnn_q20 += q20_te / n_splits_dnn
        test_dnn_q80 += q80_te / n_splits_dnn
        test_dnn_negp += negp_te / n_splits_dnn
        test_dnn_zerop += zerop_te / n_splits_dnn

    # -----------------------------
    # 5) Meta learner on OOF
    # -----------------------------
    feats_oof = dict(
        lgb_aw=oof_lgb_aw, lgb_q20=oof_lgb_q20, lgb_q80=oof_lgb_q80, lgb_negp=oof_lgb_negp,
        dnn_mean=oof_dnn_mean, dnn_q20=oof_dnn_q20, dnn_q80=oof_dnn_q80, dnn_negp=oof_dnn_negp, dnn_zerop=oof_dnn_zerop,
    )
    feats_test = dict(
        lgb_aw=test_lgb_aw, lgb_q20=test_lgb_q20, lgb_q80=test_lgb_q80, lgb_negp=test_lgb_negp,
        dnn_mean=test_dnn_mean, dnn_q20=test_dnn_q20, dnn_q80=test_dnn_q80, dnn_negp=test_dnn_negp, dnn_zerop=test_dnn_zerop,
    )

    X_meta_oof = build_meta_matrix(feats_oof)
    X_meta_test = build_meta_matrix(feats_test)

    meta = train_meta_lgb(X_meta_oof, y_full, seed=SEED + 999)
    p_oof_meta = meta.predict(X_meta_oof, num_iteration=meta.best_iteration).astype(np.float32)
    p_test_meta = meta.predict(X_meta_test, num_iteration=meta.best_iteration).astype(np.float32)

    # 6) Affine calibration on OOF
    a, b = fit_affine_calibration(y_full, p_oof_meta)
    p_oof_cal = (a * p_oof_meta + b).astype(np.float32)
    p_test_cal = (a * p_test_meta + b).astype(np.float32)

    # 7) Conservative shift tuned ONLY on train/val (OOF)
    delta = tune_conservative_shift(y_full, p_oof_cal, seed=SEED + 777)
    p_oof_shift = (p_oof_cal - delta).astype(np.float32)
    p_test_shift = (p_test_cal - delta).astype(np.float32)

    # 8) Align TEST distribution to OOF-calibrated distribution (rank-preserving)
    ref_mean = float(np.mean(p_oof_shift))
    ref_std = float(np.std(p_oof_shift) + 1e-6)
    p_test_final = align_test_distribution(p_test_shift, ref_mean=ref_mean, ref_std=ref_std, clip_scale=(0.6, 1.6))
    p_oof_final = p_oof_shift

    print("\n" + "=" * 60)
    print("OOF (train+val) diagnostic score (optimistic but useful):")
    _ = compute_score(y_full, p_oof_final)
    print(f"Affine a={a:.4f}, b={b:.2f}, conservative_delta={delta:.2f}")
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
        "method": "Physics-inspired RK2 time-stepping DNN + leak-free TE LightGBM + LGB meta + affine+shift+dist-align",
        "score": float(score),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "affine_a": float(a),
        "affine_b": float(b),
        "conservative_delta": float(delta),
        "n_meta_features": int(X_meta_oof.shape[1]),
        "n_feature_cols_lgb": int(len(feature_cols_final) if feature_cols_final is not None else -1),
        "n_train_full": int(len(full_f)),
        "n_test": int(len(test_f)),
        "n_dnn_num_cols": int(Xn_full.shape[-1]),
        "n_dnn_cat_cols": int(Xc_full.shape[-1]),
    }
    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/physics_rk2_hybrid_results.json"
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
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