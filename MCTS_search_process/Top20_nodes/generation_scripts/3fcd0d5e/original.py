#!/usr/bin/env python3
"""
AviaAgentMonty - Execution Node: 3fcd0d5e
Type: original
Generated: 2026-01-12T22:42:48.690358
Mutation: expert
Parent: None

DO NOT DELETE - This file is preserved for reproducibility.
"""
#!/usr/bin/env python3
"""Improved Method 17: Temporal Fusion Transformer (TFT) + Strong temporal feature engineering + GBDT ensemble"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW


# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_PATH = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"
DAY_COL = "TDATE_RN"  # expected 1..7

# Keep the original list as a strong prior for temporal features
NUMERICAL_COLS = ['DEPOSIT_AMOUNT', 'REC_USD', 'REC_USD_CUM', 'REC_USD_D6', 'CPI',
    'RANK1_PLAY_CNT_ALL', 'PLAY_CNT_ALL', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_REWARD_CASH', 'PLAY_CNT_CASH', 'HIGHFEE_PLAY_CNT_CASH',
    'CASH_RATIO', 'ACTIVE_DAYS_ALL_CUM', 'PLAY_CNT_ALL_CUM', 'SESSION_CNT_ALL',
    'CLEAR_PLAY_CNT_ALL', 'RANK_UNDER3_PLAY_CNT_ALL', 'JN_PLAY_CNT', 'FJ80_PLAY_CNT']


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
# Utilities
# -----------------------------
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def _detect_cols(df):
    # numeric columns (excluding target/id/day)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [TARGET_COL, ID_COL, DAY_COL]]

    # categorical columns (excluding id/day/target)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET_COL, ID_COL, DAY_COL]]

    return num_cols, cat_cols


# -----------------------------
# Feature Engineering (applies temporal advice)
# -----------------------------
def create_features(df, is_train=True):
    """
    User-level features with:
    - Base aggregates (mean/std/min/max/sum/first/last) on numeric features
    - Temporal trajectory features on a curated subset:
        velocity, acceleration, volatility, recency ratio, peak timing, rolling means
    - User-level categorical (first observed) columns (later target-encoded)
    """
    df = df.copy()
    has_day = DAY_COL in df.columns

    num_cols_all, cat_cols = _detect_cols(df)

    # Ensure stable first/last by sorting time
    if has_day:
        df = df.sort_values([ID_COL, DAY_COL])
    else:
        df = df.sort_values([ID_COL])

    # --- Base numeric aggregates ---
    # Use a compact but strong set of aggregates
    agg_funcs = ["mean", "std", "min", "max", "sum", "first", "last"]
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
    # (keep manageable dimensionality; focus on known high-signal columns)
    temporal_cols = [c for c in NUMERICAL_COLS if c in num_cols_all]
    if has_day and len(temporal_cols) > 0:
        # Wide: (user, day) -> columns (feature, day)
        wide = df[[ID_COL, DAY_COL] + temporal_cols].set_index([ID_COL, DAY_COL])[temporal_cols].unstack(DAY_COL)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([temporal_cols, list(range(1, 8))]))
        wide.columns = [f"{c}_d{d}" for c, d in wide.columns]
        wide = wide.reset_index()

        # Derived temporal features per column (vectorized)
        derived = {ID_COL: wide[ID_COL].values}
        for c in temporal_cols:
            cols_d = [f"{c}_d{i}" for i in range(1, 8)]
            s = wide[cols_d].to_numpy(dtype=np.float32)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

            d1 = s[:, 0]
            d3 = s[:, 2]
            d5 = s[:, 4]
            d7 = s[:, 6]

            mean_1_3 = s[:, 0:3].mean(axis=1)
            mean_5_7 = s[:, 4:7].mean(axis=1)

            # 1) Velocity: (Mean(days5-7) - Mean(days1-3)) / 4
            vel = (mean_5_7 - mean_1_3) / 4.0

            # 2) Acceleration: change in velocity over time (late slope - early slope)
            v_early = (d3 - d1) / 2.0
            v_late = (d7 - d5) / 2.0
            acc = v_late - v_early

            # 3) Volatility: StdDev(daily_changes) / Mean
            diffs = np.diff(s, axis=1)
            vol = diffs.std(axis=1) / (np.abs(s.mean(axis=1)) + 1e-3)

            # 4) Recency: Day7 / Day1 ratio (robust)
            rec = _safe_div(d7, d1, eps=1e-3)

            # 5) Peak timing: max in last 2 days
            peak_last2 = (np.argmax(s, axis=1) >= 5).astype(np.float32)

            # 6) Rolling/cumulative-style extras
            last3_mean = s[:, 4:7].mean(axis=1)
            first3_mean = s[:, 0:3].mean(axis=1)
            mid3_mean = s[:, 2:5].mean(axis=1)

            # Simple trend slope using centered days [-3..3] dot product
            # slope ~ sum(t * x) / sum(t^2)
            t = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
            slope = (s * t[None, :]).sum(axis=1) / (t**2).sum()

            nonzero_days = (s != 0).sum(axis=1).astype(np.float32)

            derived[f"{c}_mean_1_3"] = mean_1_3
            derived[f"{c}_mean_5_7"] = mean_5_7
            derived[f"{c}_velocity"] = vel
            derived[f"{c}_acceleration"] = acc
            derived[f"{c}_volatility"] = vol
            derived[f"{c}_recency_ratio"] = rec
            derived[f"{c}_peak_last2"] = peak_last2
            derived[f"{c}_first3_mean"] = first3_mean
            derived[f"{c}_mid3_mean"] = mid3_mean
            derived[f"{c}_last3_mean"] = last3_mean
            derived[f"{c}_trend_slope"] = slope
            derived[f"{c}_nonzero_days"] = nonzero_days

        uf = uf.merge(wide, on=ID_COL, how="left")
        uf = uf.merge(pd.DataFrame(derived), on=ID_COL, how="left")

    # Attach target at user-level (first row's target, same for all 7 days)
    if is_train and TARGET_COL in df.columns:
        y = df.groupby(ID_COL)[TARGET_COL].first().reset_index()
        uf = uf.merge(y, on=ID_COL, how="left")

    # Clean
    uf = uf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return uf


def add_target_encodings(train_f, val_f, test_f, cat_cols, n_splits=5, smoothing=20.0, seed=42):
    """
    Out-of-fold target encoding on TRAIN only (no leakage),
    then apply mapping trained on full train to val/test.
    Adds:
      - TE mean of target
      - TE negative-rate (P(y<0))
      - category frequency
    """
    if len(cat_cols) == 0:
        return train_f, val_f, test_f

    train_f = train_f.reset_index(drop=True).copy()
    val_f = val_f.copy()
    test_f = test_f.copy()

    y = train_f[TARGET_COL].values.astype(np.float32)
    global_mean = float(np.mean(y))
    global_neg = float(np.mean(y < 0))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for c in cat_cols:
        # OOF arrays
        oof_mean = np.zeros(len(train_f), dtype=np.float32)
        oof_neg  = np.zeros(len(train_f), dtype=np.float32)

        for tr_idx, te_idx in kf.split(train_f):
            tr = train_f.iloc[tr_idx]
            te = train_f.iloc[te_idx]

            stats = tr.groupby(c)[TARGET_COL].agg(["mean", "count"])
            stats["neg_rate"] = tr.groupby(c)[TARGET_COL].apply(lambda s: float(np.mean(s.values < 0)))

            cnt = stats["count"].astype(np.float32)
            sm_mean = (stats["mean"].astype(np.float32) * cnt + global_mean * smoothing) / (cnt + smoothing)
            sm_neg  = (stats["neg_rate"].astype(np.float32) * cnt + global_neg * smoothing) / (cnt + smoothing)

            oof_mean[te_idx] = te[c].map(sm_mean).fillna(global_mean).astype(np.float32).values
            oof_neg[te_idx]  = te[c].map(sm_neg).fillna(global_neg).astype(np.float32).values

        # Fit on full train for val/test transform
        stats_full = train_f.groupby(c)[TARGET_COL].agg(["mean", "count"])
        stats_full["neg_rate"] = train_f.groupby(c)[TARGET_COL].apply(lambda s: float(np.mean(s.values < 0)))

        cnt_full = stats_full["count"].astype(np.float32)
        sm_mean_full = (stats_full["mean"].astype(np.float32) * cnt_full + global_mean * smoothing) / (cnt_full + smoothing)
        sm_neg_full  = (stats_full["neg_rate"].astype(np.float32) * cnt_full + global_neg * smoothing) / (cnt_full + smoothing)

        freq_full = (cnt_full / cnt_full.sum()).astype(np.float32)

        te_mean_col = f"{c}__te_mean"
        te_neg_col  = f"{c}__te_neg_rate"
        te_freq_col = f"{c}__freq"

        train_f[te_mean_col] = oof_mean
        train_f[te_neg_col]  = oof_neg
        train_f[te_freq_col] = train_f[c].map(freq_full).fillna(0.0).astype(np.float32)

        val_f[te_mean_col] = val_f[c].map(sm_mean_full).fillna(global_mean).astype(np.float32)
        val_f[te_neg_col]  = val_f[c].map(sm_neg_full).fillna(global_neg).astype(np.float32)
        val_f[te_freq_col] = val_f[c].map(freq_full).fillna(0.0).astype(np.float32)

        test_f[te_mean_col] = test_f[c].map(sm_mean_full).fillna(global_mean).astype(np.float32)
        test_f[te_neg_col]  = test_f[c].map(sm_neg_full).fillna(global_neg).astype(np.float32)
        test_f[te_freq_col] = test_f[c].map(freq_full).fillna(0.0).astype(np.float32)

    # Drop raw cat columns after encoding
    train_f = train_f.drop(columns=cat_cols)
    val_f   = val_f.drop(columns=cat_cols)
    test_f  = test_f.drop(columns=cat_cols)
    return train_f, val_f, test_f


# -----------------------------
# Models
# -----------------------------
class GatedResidualNetwork(nn.Module):
    """GRN - core TFT component"""
    def __init__(self, input_dim, hidden_dim, output_dim, context_dim=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x, context=None):
        skip = self.skip(x)
        h = self.fc1(x)
        if context is not None and self.context_dim:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.layer_norm(skip + gate * out)


class VariableSelectionNetwork(nn.Module):
    """VSN for automated feature selection"""
    def __init__(self, num_vars, hidden_dim, context_dim=None, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_vars)
        ])
        
        self.selection_grn = GatedResidualNetwork(
            num_vars * hidden_dim, hidden_dim, num_vars, context_dim, dropout
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, context=None):
        batch = x.size(0)
        var_outputs = []
        for i in range(self.num_vars):
            var_out = self.var_grns[i](x[:, i:i+1])
            var_outputs.append(var_out)
        
        stacked = torch.stack(var_outputs, dim=1)  # (batch, num_vars, hidden_dim)
        flat = stacked.view(batch, -1)
        weights = self.softmax(self.selection_grn(flat, context))  # (batch, num_vars)
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        return selected, weights


class SimplifiedTFT(nn.Module):
    """Simplified TFT for tabular LTV prediction (with VSN + attention)"""
    def __init__(self, num_features, hidden_dim=48, num_heads=4, dropout=0.15):
        super().__init__()
        self.vsn = VariableSelectionNetwork(num_features, hidden_dim, dropout=dropout)
        self.static_encoder = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.post_attn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        selected, weights = self.vsn(x)
        context = self.static_encoder(selected)

        # Attention over a singleton token still helps as a learned interaction block
        context_seq = context.unsqueeze(1)
        attn_out, _ = self.self_attention(context_seq, context_seq, context_seq)
        attn_out = self.attention_norm(context_seq + attn_out).squeeze(1)

        out = self.post_attn(attn_out)
        return self.output(out).squeeze(-1), weights


def awmse_torch(y_true, y_pred):
    """
    Asymmetric Weighted MSE (piecewise by FP/FN) in ORIGINAL target space.
    - False Positive (pred>0, y<0): heavier
    - False Negative (pred<0, y>0): moderate
    - Else: 1.0
    """
    w = torch.ones_like(y_true)
    fp = (y_pred > 0) & (y_true < 0)
    fn = (y_pred < 0) & (y_true > 0)

    w[fp] = 2.5 + 0.02 * torch.abs(y_true[fp])
    w[fn] = 1.5 + 0.01 * y_true[fn]
    w = torch.clamp(w, 0.1, 25.0)
    return (w * (y_pred - y_true) ** 2).mean()


def train_tft(X_train, y_train, X_val, y_val, epochs=80, batch_size=512):
    print("Training Simplified TFT...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # scale target for stability, but compute loss in ORIGINAL scale for true asymmetry
    y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train) + 1e-6)
    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = SimplifiedTFT(X_train.shape[1]).to(device)
    optimizer = AdamW(model.parameters(), lr=7e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=6, factor=0.5)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb_s in train_loader:
            Xb = Xb.to(device)
            yb_s = yb_s.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_s, _ = model(Xb)

            # compute loss in ORIGINAL target space
            y_true = yb_s * y_std + y_mean
            y_pred = pred_s * y_std + y_mean
            loss = awmse_torch(y_true, y_pred)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_s, _ = model(X_val_t)
            y_true = y_val_t * y_std + y_mean
            y_pred = pred_s * y_std + y_mean
            val_loss = awmse_torch(y_true, y_pred).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: val_awMSE={val_loss:.4f}")

        if patience_counter >= 18:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, scaler, y_mean, y_std


# -----------------------------
# LightGBM (custom AWMSE objective)
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


def train_lgb_aw(X_train, y_train, X_val, y_val):
    params = dict(
        objective="none",
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_data_in_leaf=80,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.2,
        lambda_l2=0.2,
        metric="rmse",
        verbose=-1,
        seed=42,
    )

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=8000,
        valid_sets=[dval],
        fobj=lgb_aw_obj,
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            lgb.log_evaluation(period=400)
        ],
    )
    return model


def train_lgb_quantile(X_train, y_train, X_val, y_val, alpha=0.20):
    params = dict(
        objective="quantile",
        alpha=alpha,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_data_in_leaf=80,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.2,
        lambda_l2=0.2,
        metric="quantile",
        verbose=-1,
        seed=42,
    )

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=8000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            lgb.log_evaluation(period=400)
        ],
    )
    return model


def sample_weights_for_tree(y):
    # emphasize costly users; keep bounded
    w = np.ones_like(y, dtype=np.float32)
    m = y < 0
    w[m] = 2.5 + 0.02 * np.abs(y[m])
    return np.clip(w, 0.1, 25.0)


def choose_top_features_for_tft(train_f, max_features=220):
    """
    Keep TFT computationally reasonable by selecting top features by absolute Spearman to target.
    (uses TRAIN only; no leakage)
    """
    cols = [c for c in train_f.columns if c not in [ID_COL, TARGET_COL]]
    y = train_f[TARGET_COL].values
    scores = []
    for c in cols:
        x = train_f[c].values
        if np.std(x) < 1e-8:
            continue
        s = spearmanr(x, y)[0]
        if np.isnan(s):
            s = 0.0
        scores.append((abs(s), c))
    scores.sort(reverse=True)
    selected = [c for _, c in scores[:max_features]]
    return selected


def random_simplex_weights(n, rng):
    w = rng.random(n).astype(np.float32)
    w /= (w.sum() + 1e-9)
    return w


def tune_blend_on_val(y_val, preds_val_list, n_iter=2500, deltas=None, seed=42):
    """
    Tune non-negative blend weights + global bias shift delta on VAL only.
    """
    rng = np.random.default_rng(seed)
    m = len(preds_val_list)
    P = np.stack(preds_val_list, axis=1)  # (n, m)

    if deltas is None:
        deltas = np.arange(-200, 201, 10, dtype=np.float32)

    best = (-1e18, None, None)
    for _ in range(n_iter):
        w = random_simplex_weights(m, rng)
        blended = P @ w
        for d in deltas:
            pred = blended - d
            sc, _ = compute_score(y_val, pred)
            if sc > best[0]:
                best = (sc, w.copy(), float(d))

    best_score, best_w, best_delta = best
    print(f"Best VAL blend score={best_score:.6f}, weights={best_w}, delta={best_delta}")
    return best_w, best_delta


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print("Improved Method 17: Temporal features + TFT + LightGBM ensemble")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # User-level features
    train_f = create_features(train_df, is_train=True)
    val_f   = create_features(val_df, is_train=True)
    test_f  = create_features(test_df, is_train=False)

    # Detect categorical columns at user-level and target encode (train-only, OOF)
    cat_cols = train_f.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [ID_COL, TARGET_COL]]
    train_f, val_f, test_f = add_target_encodings(train_f, val_f, test_f, cat_cols, n_splits=5, smoothing=30.0, seed=42)

    # Align columns (safety)
    feat_cols = [c for c in train_f.columns if c not in [ID_COL, TARGET_COL]]
    val_f = val_f[[ID_COL, TARGET_COL] + feat_cols]
    test_f = test_f[[ID_COL] + feat_cols]

    # Matrices
    X_train = train_f[feat_cols].values.astype(np.float32)
    y_train = train_f[TARGET_COL].values.astype(np.float32)
    X_val   = val_f[feat_cols].values.astype(np.float32)
    y_val   = val_f[TARGET_COL].values.astype(np.float32)
    X_test  = test_f[feat_cols].values.astype(np.float32)
    test_ids = test_f[ID_COL].values

    print(f"Users: train={len(train_f)}, val={len(val_f)}, test={len(test_f)}")
    print(f"Features (tree/full) = {len(feat_cols)}")

    # --- Feature subset for TFT (computational) ---
    tft_cols = choose_top_features_for_tft(train_f, max_features=220)
    print(f"Features (TFT) = {len(tft_cols)}")

    X_train_tft = train_f[tft_cols].values.astype(np.float32)
    X_val_tft   = val_f[tft_cols].values.astype(np.float32)
    X_test_tft  = test_f[tft_cols].values.astype(np.float32)

    # -----------------------------
    # Train base models (train -> validate)
    # -----------------------------
    print("\nTraining LightGBM AWMSE...")
    lgb_aw = train_lgb_aw(X_train, y_train, X_val, y_val)
    p_val_lgb = lgb_aw.predict(X_val, num_iteration=lgb_aw.best_iteration).astype(np.float32)
    p_test_lgb = lgb_aw.predict(X_test, num_iteration=lgb_aw.best_iteration).astype(np.float32)

    print("\nTraining LightGBM Quantile (tau=0.20)...")
    lgb_q = train_lgb_quantile(X_train, y_train, X_val, y_val, alpha=0.20)
    p_val_q = lgb_q.predict(X_val, num_iteration=lgb_q.best_iteration).astype(np.float32)
    p_test_q = lgb_q.predict(X_test, num_iteration=lgb_q.best_iteration).astype(np.float32)

    print("\nTraining RandomForest (weighted)...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train, sample_weight=sample_weights_for_tree(y_train))
    p_val_rf = rf.predict(X_val).astype(np.float32)
    p_test_rf = rf.predict(X_test).astype(np.float32)

    print("\nTraining TFT (on top features)...")
    tft_model, tft_scaler, y_mean, y_std = train_tft(X_train_tft, y_train, X_val_tft, y_val)

    tft_model.eval()
    with torch.no_grad():
        X_val_s = tft_scaler.transform(X_val_tft)
        p_val_tft_s, _ = tft_model(torch.tensor(X_val_s, dtype=torch.float32, device=device))
        p_val_tft = (p_val_tft_s.detach().cpu().numpy().astype(np.float32) * y_std + y_mean)

        X_test_s = tft_scaler.transform(X_test_tft)
        p_test_tft_s, _ = tft_model(torch.tensor(X_test_s, dtype=torch.float32, device=device))
        p_test_tft = (p_test_tft_s.detach().cpu().numpy().astype(np.float32) * y_std + y_mean)

    # -----------------------------
    # Tune blend on VAL (no test leakage)
    # -----------------------------
    preds_val = [p_val_lgb, p_val_q, p_val_rf, p_val_tft]
    best_w, best_delta = tune_blend_on_val(y_val, preds_val, n_iter=2200, deltas=np.arange(-200, 201, 10), seed=42)

    # Optional: retrain base models on (train+val) to improve final generalization
    print("\nRetraining base models on (train+val) for final TEST predictions...")
    X_full = np.vstack([X_train, X_val]).astype(np.float32)
    y_full = np.concatenate([y_train, y_val]).astype(np.float32)

    # LGB AWMSE full
    lgb_aw_full = train_lgb_aw(X_full, y_full, X_val, y_val)  # reuse val for early-stop anchor
    p_test_lgb = lgb_aw_full.predict(X_test, num_iteration=lgb_aw_full.best_iteration).astype(np.float32)

    # LGB quantile full
    lgb_q_full = train_lgb_quantile(X_full, y_full, X_val, y_val, alpha=0.20)
    p_test_q = lgb_q_full.predict(X_test, num_iteration=lgb_q_full.best_iteration).astype(np.float32)

    # RF full
    rf_full = RandomForestRegressor(
        n_estimators=650,
        max_depth=18,
        min_samples_leaf=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf_full.fit(X_full, y_full, sample_weight=sample_weights_for_tree(y_full))
    p_test_rf = rf_full.predict(X_test).astype(np.float32)

    # TFT full (train using val as anchor still; keep weights consistent)
    X_full_tft = np.vstack([X_train_tft, X_val_tft]).astype(np.float32)
    y_full_tft = np.concatenate([y_train, y_val]).astype(np.float32)
    tft_full, tft_scaler_full, y_mean_full, y_std_full = train_tft(X_full_tft, y_full_tft, X_val_tft, y_val)

    tft_full.eval()
    with torch.no_grad():
        X_test_s = tft_scaler_full.transform(X_test_tft)
        p_test_tft_s, _ = tft_full(torch.tensor(X_test_s, dtype=torch.float32, device=device))
        p_test_tft = (p_test_tft_s.detach().cpu().numpy().astype(np.float32) * y_std_full + y_mean_full)

    # Final blend + delta
    P_test = np.stack([p_test_lgb, p_test_q, p_test_rf, p_test_tft], axis=1).astype(np.float32)
    final_pred = (P_test @ best_w).astype(np.float32) - np.float32(best_delta)

    # -----------------------------
    # Score on TEST (required)
    # -----------------------------
    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(final_pred, index=test_ids).reindex(y_test.index).values.astype(np.float32)

    print("\n" + "=" * 60)
    score, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 60)

    result = {
        "method": "Improved Method 17: Temporal FE + TFT + LGB ensemble",
        "score": float(score),
        "metrics": metrics,
        "blend_weights": [float(x) for x in best_w],
        "delta": float(best_delta),
        "n_features_tree": int(len(feat_cols)),
        "n_features_tft": int(len(tft_cols)),
    }
    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/method_17_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"score = {score}")


if __name__ == "__main__":
    main()

# =============================================================================
# SCORE CALCULATION - MANDATORY (for EVALUATION, not training)
# =============================================================================
# NOTE: This is the EVALUATION metric. You can use ANY training loss you prefer.
# But the final score MUST be calculated using this function.
# The variable MUST be named exactly 'score' for the system to read it.
# Using the pareto_multi_objective metric (higher is better)

score = compute_pareto_multi_objective(y_test, test_predictions)
print(f"score = {score}")  # This will be parsed by the system
