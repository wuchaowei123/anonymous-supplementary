# Suppress warnings to prevent false failures
import warnings
warnings.filterwarnings('ignore')

#!/usr/bin/env python3
"""
Hybrid: Retrieval-Augmented Tabular + TCN Temporal Embeddings + LightGBM (AWMSE) + Quantile + RF + Conservative Blend

Key ideas combined:
- Solution A strength: kNN retrieval features from similar users (uses historical labels in train only) + strong tabular aggregation.
- Solution B strength: sequence-aware TCN to encode 7-day dynamics (temporal patterns lost by pure aggregation).
- Final: tabular user features + retrieval stats + TCN embeddings/preds -> train (1) custom AWMSE LightGBM, (2) quantile LightGBM (tau=0.2),
  (3) RF. Then blend weights + conservative bias tuned on VAL only, refit base models on TRAIN+VAL, evaluate on TEST.

CRITICAL:
- Scoring/metric functions are preserved EXACTLY as given (calc_gini, compute_score).
- Final score computed on TEST and assigned to variable named exactly `score`.
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

import contextlib, io

# Paths / constants
TRAIN_PATH = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL, TEMPORAL_COL = "REC_USD_D60", "DEVICE_ID", "TDATE_RN"

# From Solution A (used as retrieval base)
NUMERICAL_COLS = ['DEPOSIT_AMOUNT', 'REC_USD', 'REC_USD_CUM', 'REC_USD_D6', 'CPI',
    'RANK1_PLAY_CNT_ALL', 'PLAY_CNT_ALL', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_REWARD_CASH', 'PLAY_CNT_CASH', 'HIGHFEE_PLAY_CNT_CASH',
    'CASH_RATIO', 'ACTIVE_DAYS_ALL_CUM', 'PLAY_CNT_ALL_CUM', 'SESSION_CNT_ALL']

# From Solution B (used for sequences; we will expand slightly by intersecting with available numeric columns)
TEMPORAL_FEATURES = ['DEPOSIT_AMOUNT', 'REC_USD', 'PLAY_CNT_CASH', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_REWARD_CASH', 'PLAY_CNT_ALL', 'SESSION_CNT_ALL', 'RANK1_PLAY_CNT_ALL',
    'AVG_SCORE_ALL', 'AD_PLAY_CNT_ALL', 'CLEAR_PLAY_CNT_ALL', 'HIGHFEE_PLAY_CNT_CASH']


# =========================
# SCORING FUNCTIONS (DO NOT MODIFY)  <-- preserved exactly from provided code
# =========================
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


# Required by task context (wrapper; does not alter original scoring functions)
def compute_pareto_multi_objective(y_true, y_pred):
    s, _ = compute_score(y_true, y_pred)
    return s


def compute_score_silent(y_true, y_pred):
    with contextlib.redirect_stdout(io.StringIO()):
        s, _ = compute_score(y_true, y_pred)
    return float(s)


# =========================
# Feature engineering
# =========================
def detect_columns_from_trainval(train_df: pd.DataFrame, val_df: pd.DataFrame):
    full = pd.concat([train_df, val_df], ignore_index=True)

    # numeric candidates (from train+val only; do NOT use test to choose columns)
    numeric_cols = full.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [TARGET_COL, TEMPORAL_COL]]
    numeric_cols = [c for c in numeric_cols if c != ID_COL]

    # categorical candidates
    cat_cols = []
    for c in full.columns:
        if c in [ID_COL, TARGET_COL, TEMPORAL_COL]:
            continue
        if full[c].dtype == "object" or str(full[c].dtype).startswith("category"):
            cat_cols.append(c)

    return numeric_cols, cat_cols


def make_user_level_features(df: pd.DataFrame, numeric_cols, cat_cols, is_train: bool):
    """
    User-level aggregates from daily rows.
    - Sort by time so 'last' is meaningful.
    - Adds light temporal deltas for a few key columns.
    """
    df_sorted = df.sort_values([ID_COL, TEMPORAL_COL]).copy()

    # Numeric aggregates
    agg_funcs = ['sum', 'mean', 'max', 'min', 'std', 'last']
    agg = {c: agg_funcs for c in numeric_cols if c in df_sorted.columns}
    uf = df_sorted.groupby(ID_COL).agg(agg)
    uf.columns = ['_'.join(c) for c in uf.columns]
    uf = uf.reset_index()

    # Categorical: take last observed (time-sorted)
    if len(cat_cols) > 0:
        cat_last = df_sorted.groupby(ID_COL)[cat_cols].last().reset_index()
        uf = uf.merge(cat_last, on=ID_COL, how='left')

    # Simple derived temporal deltas for key signals if present
    key_cols = [c for c in ['REC_USD', 'PLAY_CNT_ALL', 'SESSION_CNT_ALL', 'DEPOSIT_AMOUNT'] if c in df_sorted.columns]
    if key_cols:
        first = df_sorted.groupby(ID_COL)[key_cols].first().add_suffix("_first").reset_index()
        last = df_sorted.groupby(ID_COL)[key_cols].last().add_suffix("_lastraw").reset_index()
        tmp = first.merge(last, on=ID_COL, how="left")
        for c in key_cols:
            tmp[f"{c}_delta"] = tmp[f"{c}_lastraw"] - tmp[f"{c}_first"]
        tmp = tmp[[ID_COL] + [f"{c}_delta" for c in key_cols]]
        uf = uf.merge(tmp, on=ID_COL, how="left")

    # Active days (robust helper)
    uf["N_DAYS_OBS"] = df_sorted.groupby(ID_COL)[TEMPORAL_COL].nunique().reindex(uf[ID_COL]).values

    # Attach target if requested
    if is_train and TARGET_COL in df_sorted.columns:
        y = df_sorted.groupby(ID_COL)[TARGET_COL].first().reset_index()
        uf = uf.merge(y, on=ID_COL, how="left")

    return uf.fillna(0)


def add_freq_target_encoding(train_uf: pd.DataFrame, other_uf: pd.DataFrame, cat_cols, global_mean):
    """
    Frequency encoding + simple target mean encoding using TRAIN ONLY (no leakage from val/test).
    For target encoding we use raw mean per category with smoothing.
    """
    out = other_uf.copy()
    n_train = len(train_uf)

    for c in cat_cols:
        if c not in train_uf.columns or c not in out.columns:
            continue
        # Frequency
        freq = train_uf[c].astype(str).value_counts(dropna=False)
        freq = (freq / max(n_train, 1)).to_dict()
        out[f"{c}_freq"] = out[c].astype(str).map(freq).fillna(0).astype(np.float32)

        # Target mean (smoothing)
        te_stats = train_uf.groupby(train_uf[c].astype(str))[TARGET_COL].agg(['mean', 'count'])
        te_mean = te_stats['mean']
        te_cnt = te_stats['count']
        k = 20.0  # smoothing strength
        te_smooth = (te_mean * te_cnt + global_mean * k) / (te_cnt + k)
        te_map = te_smooth.to_dict()
        out[f"{c}_te"] = out[c].astype(str).map(te_map).fillna(global_mean).astype(np.float32)

    return out


# =========================
# Retrieval features (vectorized)
# =========================
def retrieval_features_from_neighbors(y_neighbors: np.ndarray, distances: np.ndarray) -> np.ndarray:
    d = distances.astype(np.float32) + 1e-8
    w = (1.0 / d)
    w = w / np.sum(w, axis=1, keepdims=True)

    wmean = np.sum(y_neighbors * w, axis=1)
    med = np.median(y_neighbors, axis=1)
    std = np.std(y_neighbors, axis=1)
    mn = np.min(y_neighbors, axis=1)
    mx = np.max(y_neighbors, axis=1)
    neg_rate = np.mean(y_neighbors < 0, axis=1)
    q25 = np.quantile(y_neighbors, 0.25, axis=1)
    q75 = np.quantile(y_neighbors, 0.75, axis=1)

    return np.column_stack([wmean, med, std, mn, mx, neg_rate, q25, q75]).astype(np.float32)


def make_retrieval_features_train_loo(X_train: np.ndarray, y_train: np.ndarray, k: int = 30) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', n_jobs=-1)
    nn.fit(Xs)
    dist, idx = nn.kneighbors(Xs)

    dist = dist[:, 1:k+1]
    idx = idx[:, 1:k+1]

    y_neighbors = y_train[idx]
    return retrieval_features_from_neighbors(y_neighbors, dist)


def make_retrieval_features_query(X_query: np.ndarray, X_db: np.ndarray, y_db: np.ndarray, k: int = 30):
    scaler = StandardScaler()
    X_db_s = scaler.fit_transform(X_db)
    X_q_s = scaler.transform(X_query)

    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    nn.fit(X_db_s)
    dist, idx = nn.kneighbors(X_q_s)

    y_neighbors = y_db[idx]
    return retrieval_features_from_neighbors(y_neighbors, dist)


# =========================
# TCN (sequence model) -> embeddings + prediction
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_sequence_data(df, temporal_features, is_train=True):
    """Create sequential data for TCN (vectorized)."""
    temp_feats = [c for c in temporal_features if c in df.columns]
    df_sorted = df.sort_values([ID_COL, TEMPORAL_COL]).reset_index(drop=True)
    users = df_sorted[ID_COL].unique()
    n_users, n_timesteps, n_feats = len(users), 7, len(temp_feats)
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    seq_data = np.zeros((n_users, n_timesteps, n_feats), dtype=np.float32)
    targets = np.zeros(n_users, dtype=np.float32) if is_train else None
    
    df_sorted['user_idx'] = df_sorted[ID_COL].map(user_to_idx)
    df_sorted['time_step'] = df_sorted.groupby(ID_COL).cumcount()
    df_filtered = df_sorted[df_sorted['time_step'] < n_timesteps]
    
    user_indices = df_filtered['user_idx'].values
    time_indices = df_filtered['time_step'].values
    
    for j, feat in enumerate(temp_feats):
        values = df_filtered[feat].fillna(0).values.astype(np.float32)
        seq_data[user_indices, time_indices, j] = values
    
    if is_train:
        last_rows = df_sorted.groupby(ID_COL).last().reset_index()
        last_rows['user_idx'] = last_rows[ID_COL].map(user_to_idx)
        targets[last_rows['user_idx'].values] = last_rows[TARGET_COL].fillna(0).values
    
    return seq_data, targets, users


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
    
    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        res = self.residual(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return torch.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.2, emb_dim=64):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.emb = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(emb_dim, 1)
    
    def forward(self, x, return_embed=False):
        x = x.transpose(1, 2)         # (B, F, T)
        h = self.tcn(x)               # (B, H, T)
        h_last = h[:, :, -1]          # (B, H)
        z = self.emb(h_last)          # (B, emb_dim)
        y = self.head(z).squeeze(-1)  # (B,)
        if return_embed:
            return y, z
        return y


def awmse_torch_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # weights depend on (y_true, y_pred) sign crossing
    w = torch.ones_like(y_true)
    fp = (y_pred > 0) & (y_true < 0)
    fn = (y_pred < 0) & (y_true > 0)
    w = torch.where(fp, 2.5 + 0.02 * torch.abs(y_true), w)
    w = torch.where(fn, 1.5 + 0.01 * y_true, w)
    return torch.mean(w * (y_pred - y_true) ** 2)


def train_tcn_with_early_stopping(X_train, y_train, X_val, y_val, epochs=60, batch_size=256, patience=10):
    scaler = StandardScaler()
    n_train, n_steps, n_feats = X_train.shape
    X_train_s = scaler.fit_transform(X_train.reshape(-1, n_feats)).reshape(n_train, n_steps, n_feats)
    X_val_s = scaler.transform(X_val.reshape(-1, n_feats)).reshape(X_val.shape)

    X_train_t = torch.FloatTensor(X_train_s).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_s).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    model = TCNEncoder(input_dim=n_feats, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.20, emb_dim=64).to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = awmse_torch_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = awmse_torch_loss(val_pred, y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            bad = 0
        else:
            bad += 1

        if (ep + 1) % 10 == 0:
            print(f"   [TCN] epoch {ep+1}/{epochs} val_awMSE={val_loss:.4f} best={best_val:.4f}")

        if bad >= patience:
            break

    model.load_state_dict(best_state)
    return model, scaler, best_epoch


def tcn_predict_and_embed(model, scaler, X_seq):
    model.eval()
    n, t, f = X_seq.shape
    Xs = scaler.transform(X_seq.reshape(-1, f)).reshape(n, t, f)
    with torch.no_grad():
        yhat, z = model(torch.FloatTensor(Xs).to(device), return_embed=True)
    return yhat.detach().cpu().numpy().astype(np.float32), z.detach().cpu().numpy().astype(np.float32)


# =========================
# LightGBM models
# =========================
def awmse_lgb_obj(y_pred, train_data):
    y_true = train_data.get_label()
    w = np.ones_like(y_true, dtype=np.float64)

    fp = (y_pred > 0) & (y_true < 0)
    fn = (y_pred < 0) & (y_true > 0)

    w[fp] = 2.5 + 0.02 * np.abs(y_true[fp])
    w[fn] = 1.5 + 0.01 * y_true[fn]

    grad = 2.0 * w * (y_pred - y_true)
    hess = 2.0 * w
    return grad, hess


def train_lgb_aw(train_X, train_y, val_X, val_y):
    params = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        metric="l2",
        seed=42,
        force_col_wise=True,
    )
    dtr = lgb.Dataset(train_X, label=train_y)
    dva = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=6000,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)],
        feval=lambda preds, train_data: ("awmse", np.mean((preds - train_data.get_label())**2), False),
    )
    return model


def train_lgb_quantile(train_X, train_y, val_X, val_y, alpha=0.20):
    params = dict(
        objective="quantile",
        alpha=alpha,
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        metric="quantile",
        seed=43,
        force_col_wise=True,
    )
    dtr = lgb.Dataset(train_X, label=train_y)
    dva = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=6000,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)],
    )
    return model


# =========================
# Blend tuning (VAL only)
# =========================
def tune_blend_on_val(y_val, preds_matrix, seed=123, n_iter=220, deltas=None):
    """
    preds_matrix: shape (n_val, m)
    returns best_weights, best_delta, best_score
    """
    rng = np.random.default_rng(seed)
    m = preds_matrix.shape[1]
    if deltas is None:
        deltas = np.arange(-150, 151, 15, dtype=np.float32)

    best_s = -1e9
    best_w = None
    best_d = 0.0

    # also include a few deterministic candidates
    candidates = []
    candidates.append(np.ones(m, dtype=np.float32) / m)
    if m >= 2:
        w = np.zeros(m, dtype=np.float32); w[0] = 0.8; w[1] = 0.2
        candidates.append(w)
        w = np.zeros(m, dtype=np.float32); w[0] = 0.6; w[1] = 0.4
        candidates.append(w)

    for _ in range(n_iter):
        candidates.append(rng.dirichlet(np.ones(m)).astype(np.float32))

    for w in candidates:
        blend = preds_matrix @ w
        for d in deltas:
            s = compute_score_silent(y_val, blend - d)
            if s > best_s:
                best_s = s
                best_w = w.copy()
                best_d = float(d)

    return best_w.astype(np.float32), float(best_d), float(best_s)


# =========================
# Main
# =========================
def main():
    print("=" * 60)
    print("Hybrid: Retrieval + TCN Embeddings + LightGBM(AWMSE) + Quantile + RF")
    print("=" * 60)
    print(f"🔧 Using device: {device}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # Column detection (train+val only; no test)
    numeric_cols, cat_cols = detect_columns_from_trainval(train_df, val_df)

    # ---------- User-level features ----------
    train_uf = make_user_level_features(train_df, numeric_cols, cat_cols, is_train=True)
    val_uf   = make_user_level_features(val_df,   numeric_cols, cat_cols, is_train=True)
    test_uf  = make_user_level_features(test_df,  numeric_cols, cat_cols, is_train=False)

    global_mean = float(train_uf[TARGET_COL].mean())

    # Add encodings (train-only)
    val_uf  = add_freq_target_encoding(train_uf, val_uf,  cat_cols, global_mean)
    test_uf = add_freq_target_encoding(train_uf, test_uf, cat_cols, global_mean)

    # Drop raw cats after encoding (keep encodings only)
    drop_cats = [c for c in cat_cols if c in train_uf.columns]
    # but keep for train too to align; encode train as well
    train_uf_enc = add_freq_target_encoding(train_uf, train_uf, cat_cols, global_mean)
    train_uf = train_uf_enc

    for df_ in (train_uf, val_uf, test_uf):
        for c in drop_cats:
            if c in df_.columns:
                df_.drop(columns=[c], inplace=True)

    # ---------- TCN training (train -> val early stopping) ----------
    # Use intersection of temporal features with available numeric columns for stability
    seq_feats = [c for c in TEMPORAL_FEATURES if c in train_df.columns]
    if len(seq_feats) < 4:
        # fallback: choose a few strongest likely present
        seq_feats = [c for c in ['REC_USD', 'PLAY_CNT_ALL', 'SESSION_CNT_ALL', 'DEPOSIT_AMOUNT'] if c in train_df.columns]

    X_tr_seq, y_tr_seq, ids_tr_seq = create_sequence_data(train_df, seq_feats, True)
    X_va_seq, y_va_seq, ids_va_seq = create_sequence_data(val_df,   seq_feats, True)
    X_te_seq, _,       ids_te_seq = create_sequence_data(test_df,  seq_feats, False)

    print(f"[TCN] seq features={len(seq_feats)}, train_seq={X_tr_seq.shape}, val_seq={X_va_seq.shape}")

    tcn_model, tcn_scaler, best_epoch = train_tcn_with_early_stopping(
        X_tr_seq, y_tr_seq, X_va_seq, y_va_seq,
        epochs=60, batch_size=256, patience=10
    )
    print(f"[TCN] best_epoch={best_epoch}")

    # TCN preds/embeddings (for blend training on val)
    tr_tcn_pred, tr_tcn_emb = tcn_predict_and_embed(tcn_model, tcn_scaler, X_tr_seq)
    va_tcn_pred, va_tcn_emb = tcn_predict_and_embed(tcn_model, tcn_scaler, X_va_seq)
    te_tcn_pred, te_tcn_emb = tcn_predict_and_embed(tcn_model, tcn_scaler, X_te_seq)

    # Align to user feature frames
    train_uf = train_uf.copy()
    val_uf = val_uf.copy()
    test_uf = test_uf.copy()

    train_uf["TCN_PRED"] = pd.Series(tr_tcn_pred, index=ids_tr_seq).reindex(train_uf[ID_COL]).values.astype(np.float32)
    val_uf["TCN_PRED"]   = pd.Series(va_tcn_pred, index=ids_va_seq).reindex(val_uf[ID_COL]).values.astype(np.float32)
    test_uf["TCN_PRED"]  = pd.Series(te_tcn_pred, index=ids_te_seq).reindex(test_uf[ID_COL]).values.astype(np.float32)

    # Add embedding dims
    emb_dim = tr_tcn_emb.shape[1] if tr_tcn_emb is not None else 0
    for j in range(emb_dim):
        col = f"TCN_EMB_{j}"
        train_uf[col] = pd.Series(tr_tcn_emb[:, j], index=ids_tr_seq).reindex(train_uf[ID_COL]).values.astype(np.float32)
        val_uf[col]   = pd.Series(va_tcn_emb[:, j], index=ids_va_seq).reindex(val_uf[ID_COL]).values.astype(np.float32)
        test_uf[col]  = pd.Series(te_tcn_emb[:, j], index=ids_te_seq).reindex(test_uf[ID_COL]).values.astype(np.float32)

    # ---------- Retrieval features ----------
    # Use a stable base subset for kNN (from Solution A list)
    # We build base vectors from the already aggregated columns corresponding to NUMERICAL_COLS aggregates.
    base_cols = []
    for c in NUMERICAL_COLS:
        for suf in ["sum", "mean", "max", "min", "std", "last"]:
            col = f"{c}_{suf}"
            if col in train_uf.columns:
                base_cols.append(col)
    if len(base_cols) < 8:
        # fallback to all numeric (excluding target)
        base_cols = [c for c in train_uf.columns if c not in [ID_COL, TARGET_COL] and train_uf[c].dtype != "object"]

    X_train_base = train_uf[base_cols].values.astype(np.float32)
    y_train_user = train_uf[TARGET_COL].values.astype(np.float32)
    X_val_base   = val_uf[base_cols].values.astype(np.float32)

    print(f"[kNN] base dims={X_train_base.shape[1]} users_train={X_train_base.shape[0]} users_val={X_val_base.shape[0]}")

    train_knn = make_retrieval_features_train_loo(X_train_base, y_train_user, k=30)
    val_knn   = make_retrieval_features_query(X_val_base, X_train_base, y_train_user, k=30)

    # attach to train/val
    knn_cols = ["KNN_WMEAN", "KNN_MED", "KNN_STD", "KNN_MIN", "KNN_MAX", "KNN_NEG_RATE", "KNN_Q25", "KNN_Q75"]
    for i, c in enumerate(knn_cols):
        train_uf[c] = train_knn[:, i]
        val_uf[c]   = val_knn[:, i]

    # ---------- Build model matrices (train->val tuning) ----------
    feat_cols = [c for c in train_uf.columns if c not in [ID_COL, TARGET_COL]]
    X_tr = train_uf[feat_cols].values.astype(np.float32)
    y_tr = train_uf[TARGET_COL].values.astype(np.float32)

    X_va = val_uf[feat_cols].values.astype(np.float32)
    y_va = val_uf[TARGET_COL].values.astype(np.float32)

    # ---------- Base models (train only, validate on val for early stopping + blend tuning) ----------
    print("[LGB] training AWMSE model...")
    lgb_aw = train_lgb_aw(X_tr, y_tr, X_va, y_va)
    p_va_aw = lgb_aw.predict(X_va, num_iteration=lgb_aw.best_iteration).astype(np.float32)

    print("[LGB] training Quantile (tau=0.20) model...")
    lgb_q = train_lgb_quantile(X_tr, y_tr, X_va, y_va, alpha=0.20)
    p_va_q = lgb_q.predict(X_va, num_iteration=lgb_q.best_iteration).astype(np.float32)

    print("[RF] training RandomForest...")
    # light asymmetric sample weighting (label-based only) to bias against costly users
    sw = np.ones_like(y_tr, dtype=np.float32)
    neg = y_tr < 0
    sw[neg] = np.clip(2.5 + 0.02 * np.abs(y_tr[neg]), 0.1, 10.0)

    rf = RandomForestRegressor(
        n_estimators=450,
        max_depth=18,
        min_samples_leaf=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_tr, y_tr, sample_weight=sw)
    p_va_rf = rf.predict(X_va).astype(np.float32)

    # Also include raw retrieval wmean and raw TCN prediction as blend candidates
    p_va_knn = val_uf["KNN_WMEAN"].values.astype(np.float32)
    p_va_tcn = val_uf["TCN_PRED"].values.astype(np.float32)

    # ---------- Tune blend weights + conservative shift on VAL only ----------
    preds_val_mat = np.vstack([p_va_aw, p_va_q, p_va_rf, p_va_knn, p_va_tcn]).T.astype(np.float32)
    names = ["LGB_AW", "LGB_Q20", "RF", "KNN_WMEAN", "TCN"]

    best_w, best_delta, best_val_score = tune_blend_on_val(
        y_va, preds_val_mat,
        seed=123,
        n_iter=220,
        deltas=np.arange(-150, 151, 15, dtype=np.float32)
    )
    print(f"[BLEND] best_val_score={best_val_score:.6f} best_delta={best_delta} weights:")
    for n, w in zip(names, best_w):
        print(f"   {n:10s}: {w:.4f}")

    # ---------- Refit on FULL (train+val) for TEST predictions ----------
    print("\n[REFIT] Re-training base models on TRAIN+VAL and predicting TEST...")

    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # User-level full
    full_uf = make_user_level_features(full_df, numeric_cols, cat_cols, is_train=True)
    global_mean_full = float(full_uf[TARGET_COL].mean())
    full_uf = add_freq_target_encoding(full_uf, full_uf, cat_cols, global_mean_full)
    test_uf_full = make_user_level_features(test_df, numeric_cols, cat_cols, is_train=False)
    test_uf_full = add_freq_target_encoding(full_uf, test_uf_full, cat_cols, global_mean_full)

    # drop raw cats
    drop_cats_full = [c for c in cat_cols if c in full_uf.columns]
    for df_ in (full_uf, test_uf_full):
        for c in drop_cats_full:
            if c in df_.columns:
                df_.drop(columns=[c], inplace=True)

    # Re-train TCN on full using fixed epochs = best_epoch (from train/val early stopping)
    # (keeps sequence signal but uses more data; does not use test)
    X_full_seq, y_full_seq, ids_full_seq = create_sequence_data(full_df, seq_feats, True)
    X_test_seq, _, ids_test_seq = create_sequence_data(test_df, seq_feats, False)

    # Fit scaler on full and train for best_epoch (no early stopping here to avoid leaking test; epochs fixed)
    scaler_full = StandardScaler()
    n_full, n_steps, n_feats = X_full_seq.shape
    X_full_s = scaler_full.fit_transform(X_full_seq.reshape(-1, n_feats)).reshape(n_full, n_steps, n_feats)

    model_full = TCNEncoder(input_dim=n_feats, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.20, emb_dim=64).to(device)
    opt_full = AdamW(model_full.parameters(), lr=1e-3, weight_decay=1e-2)
    loader_full = DataLoader(
        TensorDataset(torch.FloatTensor(X_full_s).to(device), torch.FloatTensor(y_full_seq).to(device)),
        batch_size=256, shuffle=True
    )

    model_full.train()
    for ep in range(max(1, int(best_epoch))):
        for xb, yb in loader_full:
            opt_full.zero_grad(set_to_none=True)
            pred = model_full(xb)
            loss = awmse_torch_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
            opt_full.step()

    # Full TCN preds/embeddings
    def predict_embed_with_full_scaler(model, scaler, X_seq):
        model.eval()
        n, t, f = X_seq.shape
        Xs = scaler.transform(X_seq.reshape(-1, f)).reshape(n, t, f)
        with torch.no_grad():
            yhat, z = model(torch.FloatTensor(Xs).to(device), return_embed=True)
        return yhat.detach().cpu().numpy().astype(np.float32), z.detach().cpu().numpy().astype(np.float32)

    full_tcn_pred, full_tcn_emb = predict_embed_with_full_scaler(model_full, scaler_full, X_full_seq)
    test_tcn_pred2, test_tcn_emb2 = predict_embed_with_full_scaler(model_full, scaler_full, X_test_seq)

    full_uf["TCN_PRED"] = pd.Series(full_tcn_pred, index=ids_full_seq).reindex(full_uf[ID_COL]).values.astype(np.float32)
    test_uf_full["TCN_PRED"] = pd.Series(test_tcn_pred2, index=ids_test_seq).reindex(test_uf_full[ID_COL]).values.astype(np.float32)

    emb_dim2 = full_tcn_emb.shape[1]
    for j in range(emb_dim2):
        col = f"TCN_EMB_{j}"
        full_uf[col] = pd.Series(full_tcn_emb[:, j], index=ids_full_seq).reindex(full_uf[ID_COL]).values.astype(np.float32)
        test_uf_full[col] = pd.Series(test_tcn_emb2[:, j], index=ids_test_seq).reindex(test_uf_full[ID_COL]).values.astype(np.float32)

    # Retrieval on full (LOO for full, query for test)
    base_cols_full = []
    for c in NUMERICAL_COLS:
        for suf in ["sum", "mean", "max", "min", "std", "last"]:
            col = f"{c}_{suf}"
            if col in full_uf.columns:
                base_cols_full.append(col)
    if len(base_cols_full) < 8:
        base_cols_full = [c for c in full_uf.columns if c not in [ID_COL, TARGET_COL] and full_uf[c].dtype != "object"]

    X_full_base = full_uf[base_cols_full].values.astype(np.float32)
    y_full_user = full_uf[TARGET_COL].values.astype(np.float32)
    X_test_base = test_uf_full[base_cols_full].values.astype(np.float32)

    full_knn = make_retrieval_features_train_loo(X_full_base, y_full_user, k=30)
    test_knn = make_retrieval_features_query(X_test_base, X_full_base, y_full_user, k=30)

    for i, c in enumerate(knn_cols):
        full_uf[c] = full_knn[:, i]
        test_uf_full[c] = test_knn[:, i]

    # Matrices for final models
    feat_cols_full = [c for c in full_uf.columns if c not in [ID_COL, TARGET_COL]]
    X_full = full_uf[feat_cols_full].values.astype(np.float32)
    y_full = full_uf[TARGET_COL].values.astype(np.float32)
    X_test = test_uf_full[feat_cols_full].values.astype(np.float32)
    test_ids = test_uf_full[ID_COL].values

    # Train base models on full (no val early stopping; reuse best_iteration from earlier fits)
    # (This is a pragmatic compromise: keeps the tuned capacity while leveraging more data.)
    print("[REFIT-LGB] training AWMSE on full...")
    params_aw = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        metric="l2",
        seed=42,
        force_col_wise=True,
    )
    dfull = lgb.Dataset(X_full, label=y_full)
    lgb_aw_full = lgb.train(
        params_aw, dfull, num_boost_round=int(max(200, lgb_aw.best_iteration or 200)),
        valid_sets=[dfull], valid_names=["train"], callbacks=[lgb.log_evaluation(period=0)]
    )
    p_te_aw = lgb_aw_full.predict(X_test).astype(np.float32)

    print("[REFIT-LGB] training Quantile on full...")
    params_q = dict(
        objective="quantile",
        alpha=0.20,
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        metric="quantile",
        seed=43,
        force_col_wise=True,
    )
    lgb_q_full = lgb.train(
        params_q, dfull, num_boost_round=int(max(200, lgb_q.best_iteration or 200)),
        valid_sets=[dfull], valid_names=["train"], callbacks=[lgb.log_evaluation(period=0)]
    )
    p_te_q = lgb_q_full.predict(X_test).astype(np.float32)

    print("[REFIT-RF] training RF on full...")
    sw_full = np.ones_like(y_full, dtype=np.float32)
    neg_full = y_full < 0
    sw_full[neg_full] = np.clip(2.5 + 0.02 * np.abs(y_full[neg_full]), 0.1, 10.0)

    rf_full = RandomForestRegressor(
        n_estimators=550,
        max_depth=18,
        min_samples_leaf=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf_full.fit(X_full, y_full, sample_weight=sw_full)
    p_te_rf = rf_full.predict(X_test).astype(np.float32)

    p_te_knn = test_uf_full["KNN_WMEAN"].values.astype(np.float32)
    p_te_tcn = test_uf_full["TCN_PRED"].values.astype(np.float32)

    preds_test_mat = np.vstack([p_te_aw, p_te_q, p_te_rf, p_te_knn, p_te_tcn]).T.astype(np.float32)
    final_pred = (preds_test_mat @ best_w).astype(np.float32) - float(best_delta)

    # ---------- Score on TEST (required) ----------
    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(final_pred, index=test_ids).reindex(y_test.index).values

    print("\n" + "=" * 60)
    score, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 60)

    # Save artifacts (optional)
    out = {
        "method": "Hybrid: Retrieval+TCN+LGB(AWMSE)+Quantile+RF",
        "score": float(score),
        "metrics": metrics,
        "blend": {
            "names": names,
            "weights": [float(x) for x in best_w],
            "delta": float(best_delta),
            "val_score_at_tune": float(best_val_score),
        },
        "tcn": {"seq_features": seq_feats, "best_epoch_train_val": int(best_epoch)},
        "features": {"n_feat_train": int(len(feat_cols)), "n_feat_full": int(len(feat_cols_full))}
    }
    out_path = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/hybrid_retrieval_tcn_results.json"
    try:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] wrote {out_path}")
    except Exception as e:
        print(f"[WARN] could not write json: {e}")

    print(f"score = {score}")
    
    # Return values needed for final score calculation
    return y_test.values, pred_aligned, score


if __name__ == "__main__":
    y_test, test_predictions, score = main()

# =============================================================================
# SCORE CALCULATION - MANDATORY (for EVALUATION, not training)
# =============================================================================
# NOTE: This is the EVALUATION metric. You can use ANY training loss you prefer.
# But the final score MUST be calculated using this function.
# The variable MUST be named exactly 'score' for the system to read it.
# Using the pareto_multi_objective metric (higher is better)

score = compute_pareto_multi_objective(y_test, test_predictions)
print(f"score = {score}")
