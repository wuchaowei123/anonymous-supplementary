# Suppress warnings to prevent false failures
import warnings
warnings.filterwarnings('ignore')

#!/usr/bin/env python3
"""
Method 25 (Rewritten): Physics-Inspired Euler/ResNet Time-Stepping Network

Based on:
Deep Learning and Computational Physics (Lecture Notes) (Ray, Pinti, Oberai, 2023)

Key ideas adapted to this task:
1) ResNets as ODE discretizations (Forward Euler time stepping):
      h_{t+1} = h_t + Δt * f(h_t, x_t)
2) Stabilization via damping (diffusion-like term) and energy regularization:
      h_{t+1} = (1-γ) h_t + Δt f(...)
   plus an "energy" penalty on ||f||^2 (smooth/stable dynamics)
3) Sequence modeling across 7 days (no naive aggregation as the only signal)
4) Multi-head outputs to support conservative boundary handling:
   - mean regression head (AWMSE loss)
   - quantile head (pinball loss, τ=0.20)
   - sign-risk head (weighted BCE for P(y<0))
5) Post-hoc conservative calibration tuned on VAL only (no test leakage)

CRITICAL:
- Metric/scoring functions are preserved EXACTLY as provided.
- Final score is computed on TEST and assigned to variable named `score`,
  then printed with print(f"score = {score}").

================================================================================
Deep Research Method Implementation for BT_IOS_2503_Pareto
================================================================================
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

# -----------------------------------------------------------------------------
# Device / Repro
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_PATH = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"
OUTPUT_JSON = os.environ.get('anonymous_institution_OUTPUT_JSON', '/tmp/anonymous_institution_score_output.json')

TARGET_COL = "REC_USD_D60"
ID_COL = "DEVICE_ID"
TEMPORAL_COL = "TDATE_RN"

# Kept (but we will auto-detect numeric columns too; these are used as a fallback list)
NUMERICAL_COLS = [
    'DEPOSIT_AMOUNT', 'REC_USD', 'REC_USD_CUM', 'REC_USD_D6', 'CPI',
    'RANK1_PLAY_CNT_ALL', 'RANK_UNDER3_PLAY_CNT_ALL', 'PLAY_CNT_ALL', 'PLAY_CNT_TICKET', 'AD_PLAY_CNT_ALL',
    'CLEAR_PLAY_CNT_ALL', 'AVG_SCORE_ALL', 'SESSION_CNT_ALL', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_BONUS_ENTRY_FEE_CASH', 'ACTUAL_REWARD_CASH', 'ACTUAL_BONUS_REWARD_CASH', 'PLAY_CNT_CASH',
    'RANK1_PLAY_CNT_CASH', 'HIGHFEE_PLAY_CNT_CASH', 'JN_PLAY_CNT', 'FJ80_PLAY_CNT',
    'CASH_RATIO', 'ACTIVE_DAYS_ALL_CUM', 'PLAY_CNT_ALL_CUM', 'PLAY_CNT_CASH_CUM',
]
CATEGORICAL_COLS = ['MEDIA_SOURCE', 'COUNTRY', 'DEVICE_TYPE']

# =============================================================================
# METRIC FUNCTIONS (⚠️ PRESERVE EXACTLY)
# =============================================================================
def calculate_norm_gini(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) == 0 or np.sum(y_true) <= 0:
        return 0.0
    
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    cumsum = np.cumsum(y_true_sorted)
    total = cumsum[-1]
    
    if total == 0:
        return 0.0
    
    lorenz = cumsum / total
    n = len(y_true)
    gini_actual = 2 * np.sum(lorenz) / n - 1
    
    y_true_perfect = np.sort(y_true)[::-1]
    cumsum_perfect = np.cumsum(y_true_perfect)
    lorenz_perfect = cumsum_perfect / total
    gini_perfect = 2 * np.sum(lorenz_perfect) / n - 1
    
    if gini_perfect == 0:
        return 1.0
    
    return gini_actual / gini_perfect


def calculate_error_rate(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    true_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)
    
    if true_sum == 0:
        return float('inf') if pred_sum > 0 else 0.0
    
    return abs(true_sum - pred_sum) / abs(true_sum)


def calculate_spearman(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def calculate_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_composite_score(y_true, y_pred):
    gini = calculate_norm_gini(y_true, y_pred)
    error_rate = calculate_error_rate(y_true, y_pred)
    spearman = calculate_spearman(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    gini_score = np.clip((gini - 0.70) / (0.90 - 0.70), 0.0, 1.0)
    error_score = np.clip((0.35 - error_rate) / (0.35 - 0.01), 0.0, 1.0)
    spearman_score = np.clip((spearman - 0.50) / (0.80 - 0.50), 0.0, 1.0)
    rmse_score = np.clip((260 - rmse) / (260 - 200), 0.0, 1.0)
    
    base_score = 0.35 * gini_score + 0.25 * spearman_score + 0.20 * rmse_score + 0.20 * error_score
    
    pareto_bonus = 0.0
    excellence_count = 0
    
    if gini_score > 0.8:
        pareto_bonus += 0.035
        excellence_count += 1
    if spearman_score > 0.8:
        pareto_bonus += 0.025
        excellence_count += 1
    if rmse_score > 0.8:
        pareto_bonus += 0.020
        excellence_count += 1
    if error_score > 0.8:
        pareto_bonus += 0.020
        excellence_count += 1
    
    scores = [gini_score, error_score, spearman_score, rmse_score]
    score_std = np.std(scores)
    
    diversity_bonus = min(0.02, score_std * 0.1) if score_std > 0.2 and base_score > 0.5 else 0.0
    
    if excellence_count >= 2:
        pareto_bonus += 0.02
    if excellence_count >= 3:
        pareto_bonus += 0.03
    if excellence_count == 4:
        pareto_bonus += 0.05
    
    final_score = base_score + pareto_bonus + diversity_bonus
    
    print(f"📊 Metrics: Gini={gini:.4f}, ErrRate={error_rate:.4f}, Spearman={spearman:.4f}, RMSE={rmse:.2f}")
    print(f"📊 Individual Scores: Gini={gini_score:.3f}, Err={error_score:.3f}, Spear={spearman_score:.3f}, RMSE={rmse_score:.3f}")
    print(f"📊 Base={base_score:.4f}, Pareto={pareto_bonus:.2f}, Diversity={diversity_bonus:.3f}, FINAL={final_score:.4f}")
    print(f"📊 Excellence count: {excellence_count}/4 metrics")
    
    return final_score, {
        'gini': gini, 'error_rate': error_rate, 'spearman': spearman, 'rmse': rmse,
        'gini_score': gini_score, 'error_score': error_score, 
        'spearman_score': spearman_score, 'rmse_score': rmse_score,
        'base_score': base_score, 'pareto_bonus': pareto_bonus,
        'diversity_bonus': diversity_bonus, 'excellence_count': excellence_count,
    }

# Optional wrapper to match task wording without altering existing metric logic
def compute_pareto_multi_objective(y_true, y_pred):
    s, _ = compute_composite_score(y_true, y_pred)
    return s

# =============================================================================
# DATA LOADING
# =============================================================================
def load_all_data():
    print("📂 Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"   Train: {train_df.shape} ({train_df[ID_COL].nunique()} users)")
    print(f"   Val:   {val_df.shape} ({val_df[ID_COL].nunique()} users)")
    print(f"   Test:  {test_df.shape} ({test_df[ID_COL].nunique()} users)")
    return train_df, val_df, test_df

def infer_columns(train_df: pd.DataFrame):
    # Categorical: configured + any object columns (excluding ID)
    cat_cols = [c for c in CATEGORICAL_COLS if c in train_df.columns]
    auto_obj = [c for c in train_df.columns if (train_df[c].dtype == 'object' or str(train_df[c].dtype).startswith('string'))]
    for c in auto_obj:
        if c not in cat_cols and c not in (ID_COL,):
            cat_cols.append(c)

    # Numeric: use all numeric columns except target/id/time
    exclude = {ID_COL, TARGET_COL, TEMPORAL_COL}
    num_cols = [c for c in train_df.select_dtypes(include=[np.number, 'bool']).columns if c not in exclude]

    # Fallback to provided list if something odd happens
    if len(num_cols) == 0:
        num_cols = [c for c in NUMERICAL_COLS if c in train_df.columns]

    return num_cols, cat_cols

def fit_category_maps(train_df: pd.DataFrame, cat_cols):
    cat_maps = {}
    cat_sizes = {}
    for c in cat_cols:
        vals = train_df[c].fillna("__NA__").astype(str).values
        uniq = pd.unique(vals)
        # reserve 0 for UNK
        mapping = {v: i + 1 for i, v in enumerate(uniq)}
        cat_maps[c] = mapping
        cat_sizes[c] = len(mapping) + 1
    return cat_maps, cat_sizes

def encode_categories(df: pd.DataFrame, cat_cols, cat_maps):
    if len(cat_cols) == 0:
        return np.zeros((len(df), 0), dtype=np.int64)
    out = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        mapping = cat_maps[c]
        vals = df[c].fillna("__NA__").astype(str).values
        # unknown -> 0
        out[:, j] = np.array([mapping.get(v, 0) for v in vals], dtype=np.int64)
    return out

def build_user_sequences(
    df: pd.DataFrame,
    num_cols,
    cat_cols,
    cat_maps=None,
    scaler=None,
    fit_scaler=False,
    expect_days=7,
):
    """
    Returns:
      X_seq_num: (n_users, 7, n_num_aug) float32
      X_seq_cat: (n_users, 7, n_cat) int64
      X_static:  (n_users, static_dim) float32
      y:         (n_users,) float32 or None
      ids:       (n_users,)
    """
    # Ensure sorting so reshape is valid if every user has exactly 7 rows
    df = df.sort_values([ID_COL, TEMPORAL_COL]).reset_index(drop=True)

    # Basic integrity: each user should have 7 rows in this task
    counts = df.groupby(ID_COL)[TEMPORAL_COL].count().values
    if not np.all(counts == expect_days):
        # Pad/truncate robustly if needed (rare); keep deterministic ordering by TEMPORAL_COL
        # We'll build per-user via loop (safe, still small enough at ~55k users * 7 rows).
        ids = df[ID_COL].unique()
        n_users = len(ids)
        n_num = len(num_cols)
        n_cat = len(cat_cols)

        X_num = np.zeros((n_users, expect_days, n_num), dtype=np.float32)
        X_cat = np.zeros((n_users, expect_days, n_cat), dtype=np.int64)
        y = np.zeros((n_users,), dtype=np.float32) if TARGET_COL in df.columns else None

        # pre-encode per-row
        num_rows = df[num_cols].astype(np.float32).fillna(0.0).values
        cat_rows = encode_categories(df, cat_cols, cat_maps) if cat_maps is not None else np.zeros((len(df), n_cat), dtype=np.int64)

        # build index per user
        grp = df.groupby(ID_COL).indices
        for i, uid in enumerate(ids):
            idxs = grp[uid]
            # sorted by TEMPORAL_COL already
            take = idxs[:expect_days]
            k = len(take)
            X_num[i, :k, :] = num_rows[take]
            X_cat[i, :k, :] = cat_rows[take]
            if y is not None:
                y[i] = float(df.loc[take[0], TARGET_COL])
    else:
        ids = df[ID_COL].drop_duplicates().values
        n_users = len(ids)
        n_num = len(num_cols)
        n_cat = len(cat_cols)

        num_rows = df[num_cols].astype(np.float32).fillna(0.0).values
        X_num = num_rows.reshape(n_users, expect_days, n_num)

        if n_cat > 0:
            cat_rows = encode_categories(df, cat_cols, cat_maps)
            X_cat = cat_rows.reshape(n_users, expect_days, n_cat)
        else:
            X_cat = np.zeros((n_users, expect_days, 0), dtype=np.int64)

        y = None
        if TARGET_COL in df.columns:
            # target is constant within user; take first row per user
            y = df.groupby(ID_COL)[TARGET_COL].first().astype(np.float32).values

    # Fit/Apply numeric scaler on flattened (train only)
    X_num_flat = X_num.reshape(-1, X_num.shape[-1]).astype(np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        X_num_flat = scaler.fit_transform(X_num_flat).astype(np.float32)
    else:
        X_num_flat = scaler.transform(X_num_flat).astype(np.float32)
    X_num_scaled = X_num_flat.reshape(X_num.shape)

    # Temporal augmentation: time index + first-differences
    # time feature in [-1, 1]
    t = np.linspace(-1.0, 1.0, expect_days, dtype=np.float32)[None, :, None]
    t = np.repeat(t, X_num_scaled.shape[0], axis=0)

    # diff features (velocity-like)
    diffs = np.diff(X_num_scaled, axis=1, prepend=X_num_scaled[:, 0:1, :]).astype(np.float32)

    # Concatenate numeric aug: [scaled, diffs, time]
    X_seq_num = np.concatenate([X_num_scaled, diffs, t], axis=2).astype(np.float32)

    # Static (user-level) summary features (helps head / ranking):
    # mean, std, min, max, last, slope (last-first)
    mean = X_num_scaled.mean(axis=1)
    std = X_num_scaled.std(axis=1)
    mn = X_num_scaled.min(axis=1)
    mx = X_num_scaled.max(axis=1)
    last = X_num_scaled[:, -1, :]
    slope = (X_num_scaled[:, -1, :] - X_num_scaled[:, 0, :])
    X_static = np.concatenate([mean, std, mn, mx, last, slope], axis=1).astype(np.float32)

    return X_seq_num, X_seq_cat, X_static, y, ids, scaler

# =============================================================================
# MODEL: Physics-inspired Euler time-stepping encoder + multi-head outputs
# =============================================================================
def choose_emb_dim(cardinality: int) -> int:
    # Common heuristic: O(sqrt(cardinality)), capped.
    if cardinality <= 2:
        return 2
    return int(min(16, max(3, round(1.6 * np.sqrt(cardinality)))))

class DynamicsMLP(nn.Module):
    def __init__(self, state_dim: int, input_dim: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(state_dim + input_dim)
        self.fc1 = nn.Linear(state_dim + input_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, state_dim)

    def forward(self, h, x):
        z = torch.cat([h, x], dim=-1)
        z = self.norm(z)
        z = self.fc1(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.fc2(z)
        return z  # dh/dt

class EulerResNetEncoder(nn.Module):
    """
    Forward Euler time stepping:
      h_{t+1} = (1-γ) h_t + Δt * f(h_t, x_t)
    """
    def __init__(self, seq_num_dim: int, cat_sizes: dict, cat_cols: list, state_dim: int = 128):
        super().__init__()
        self.cat_cols = cat_cols
        self.state_dim = state_dim

        # Embeddings
        self.embeddings = nn.ModuleDict()
        self.emb_dims = {}
        emb_total = 0
        for c in cat_cols:
            card = int(cat_sizes[c])
            dim = choose_emb_dim(card)
            self.embeddings[c] = nn.Embedding(card, dim)
            self.emb_dims[c] = dim
            emb_total += dim

        in_dim = seq_num_dim + emb_total
        self.in_proj = nn.Linear(in_dim, state_dim)

        self.dynamics = DynamicsMLP(state_dim=state_dim, input_dim=state_dim, hidden=256, dropout=0.15)

        # Learnable stable step-size and damping (bounded)
        self._dt = nn.Parameter(torch.tensor(0.0))      # dt = sigmoid(_dt) in (0,1)
        self._gamma = nn.Parameter(torch.tensor(-2.0))  # gamma = sigmoid(_gamma) ~ small

        self.h_norm = nn.LayerNorm(state_dim)

    def forward(self, x_num, x_cat):
        """
        x_num: (B, T, seq_num_dim)
        x_cat: (B, T, n_cat)
        returns: h_last (B, state_dim), h_mean (B, state_dim), energy (scalar tensor)
        """
        B, T, _ = x_num.shape
        dt = torch.sigmoid(self._dt)          # (0,1)
        gamma = torch.sigmoid(self._gamma)    # (0,1)

        # build per-step inputs
        if x_cat.shape[-1] > 0:
            emb_list = []
            for j, c in enumerate(self.cat_cols):
                emb = self.embeddings[c](x_cat[:, :, j])
                emb_list.append(emb)
            x = torch.cat([x_num] + emb_list, dim=-1)
        else:
            x = x_num

        x = self.in_proj(x)  # (B, T, state_dim)

        h = torch.zeros((B, self.state_dim), device=x.device, dtype=x.dtype)
        hs = []
        energy_accum = 0.0

        for t in range(T):
            xt = x[:, t, :]
            # physics-inspired stable residual step
            dh = self.dynamics(h, xt)
            h = (1.0 - gamma) * h + dt * dh
            h = self.h_norm(h)
            hs.append(h.unsqueeze(1))
            energy_accum = energy_accum + (dh ** 2).mean()

        h_seq = torch.cat(hs, dim=1)  # (B, T, D)
        h_last = h_seq[:, -1, :]
        h_mean = h_seq.mean(dim=1)
        energy = energy_accum / float(T)
        return h_last, h_mean, energy

class MultiHeadPredictor(nn.Module):
    def __init__(self, seq_num_dim: int, static_dim: int, cat_sizes: dict, cat_cols: list, state_dim: int = 128):
        super().__init__()
        self.encoder = EulerResNetEncoder(seq_num_dim=seq_num_dim, cat_sizes=cat_sizes, cat_cols=cat_cols, state_dim=state_dim)

        head_in = state_dim * 2 + static_dim  # h_last + h_mean + static
        self.shared = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.mean_head = nn.Linear(128, 1)
        self.q20_head = nn.Linear(128, 1)
        self.sign_head = nn.Linear(128, 1)  # logit for P(y<0)

    def forward(self, x_num, x_cat, x_static):
        h_last, h_mean, energy = self.encoder(x_num, x_cat)
        z = torch.cat([h_last, h_mean, x_static], dim=-1)
        z = self.shared(z)
        y_mean = self.mean_head(z).squeeze(-1)
        y_q20 = self.q20_head(z).squeeze(-1)
        sign_logit = self.sign_head(z).squeeze(-1)
        return y_mean, y_q20, sign_logit, energy

# =============================================================================
# LOSSES: Asymmetric Weighted MSE + Pinball + Weighted BCE
# =============================================================================
def awmse_loss(y_true, y_pred):
    """
    Piecewise AWMSE similar to the research guidance.
    NOTE: weight depends on sign mismatch; kept simple but effective.
    """
    # y_true, y_pred are torch tensors (B,)
    err = (y_pred - y_true)
    # false positive: predict >0 but true <0
    fp = (y_pred > 0) & (y_true < 0)
    # false negative: predict <0 but true >0
    fn = (y_pred < 0) & (y_true > 0)

    w = torch.ones_like(y_true)
    w_fp = 2.5 + 0.02 * torch.abs(y_true)
    w_fn = 1.5 + 0.01 * torch.clamp(y_true, min=0.0)

    w = torch.where(fp, w_fp, w)
    w = torch.where(fn, w_fn, w)
    return (w * (err ** 2)).mean()

def pinball_loss(y_true, y_pred, tau: float = 0.20):
    # y_true, y_pred: (B,)
    diff = y_true - y_pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))

def weighted_bce_with_logits(logits, targets, pos_weight: float = 1.67):
    # targets in {0,1}, where 1 denotes y<0 (negative users)
    # Penalize missing negatives (i.e., predicting non-negative for a truly negative) -> higher pos_weight.
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

# =============================================================================
# CALIBRATION (VAL ONLY): conservative blending & shift
# =============================================================================
def false_positive_rate(y_true, y_pred):
    # predicted profitable (pred>0) but actually costly (true<0)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos_pred = (y_pred > 0)
    if pos_pred.sum() == 0:
        return 0.0
    return float(((y_true < 0) & pos_pred).sum() / pos_pred.sum())

def apply_conservative_calibration(mean_pred, q20_pred, p_neg, alpha=0.30, beta=0.50, delta=0.0):
    """
    mean_pred: conditional mean estimate
    q20_pred:  τ=0.20 estimate (conservative lower bound)
    p_neg:     probability user is negative (risk)
    alpha: blend weight on q20 (more alpha -> more conservative)
    beta: risk shrink intensity
    delta: global downward shift (subtract delta)
    """
    mean_pred = np.asarray(mean_pred, dtype=np.float32)
    q20_pred = np.asarray(q20_pred, dtype=np.float32)
    p_neg = np.asarray(p_neg, dtype=np.float32)

    blended = (1.0 - alpha) * mean_pred + alpha * q20_pred
    # shrink positives when risk is high
    shrink = 1.0 - beta * p_neg
    calibrated = blended * shrink
    calibrated = calibrated - float(delta)
    return calibrated

def tune_calibration_on_val(y_val, mean_val, q20_val, pneg_val):
    """
    Simple grid search on VAL only.
    Objective: maximize composite score while encouraging FPR <= 0.40 via delta.
    """
    best = None
    # Conservative grid (small)
    alphas = [0.0, 0.15, 0.30, 0.45, 0.60]
    betas  = [0.0, 0.25, 0.50, 0.75, 1.00]
    deltas = [0, 5, 10, 20, 40, 80, 120, 160, 200]

    for a in alphas:
        for b in betas:
            # choose smallest delta that satisfies FPR<=0.40, then maximize score
            best_for_ab = None
            for d in deltas:
                pred = apply_conservative_calibration(mean_val, q20_val, pneg_val, alpha=a, beta=b, delta=d)
                fpr = false_positive_rate(y_val, pred)
                sc, _ = compute_composite_score(y_val, pred)
                ok = (fpr <= 0.40)
                # prioritize ok solutions; among ok pick highest score; if none ok, pick highest score anyway
                key = (1 if ok else 0, sc)
                if best_for_ab is None or key > best_for_ab["key"]:
                    best_for_ab = {"key": key, "alpha": a, "beta": b, "delta": d, "score": sc, "fpr": fpr}

            if best is None or best_for_ab["key"] > best["key"]:
                best = best_for_ab

    print(f"🛡️ Calibration (VAL) best: alpha={best['alpha']}, beta={best['beta']}, delta={best['delta']} | "
          f"val_score={best['score']:.4f} | val_fpr={best['fpr']:.3f}")
    return best

# =============================================================================
# TRAINING / INFERENCE
# =============================================================================
@torch.no_grad()
def predict_model(model, X_seq_num, X_seq_cat, X_static, batch_size=2048):
    model.eval()
    preds_mean = []
    preds_q20 = []
    preds_pneg = []

    ds = TensorDataset(
        torch.from_numpy(X_seq_num),
        torch.from_numpy(X_seq_cat),
        torch.from_numpy(X_static),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda'))

    for xb_num, xb_cat, xb_static in dl:
        xb_num = xb_num.to(device)
        xb_cat = xb_cat.to(device)
        xb_static = xb_static.to(device)

        y_mean, y_q20, sign_logit, _ = model(xb_num, xb_cat, xb_static)
        p_neg = torch.sigmoid(sign_logit)

        preds_mean.append(y_mean.cpu().numpy())
        preds_q20.append(y_q20.cpu().numpy())
        preds_pneg.append(p_neg.cpu().numpy())

    return (
        np.concatenate(preds_mean, axis=0),
        np.concatenate(preds_q20, axis=0),
        np.concatenate(preds_pneg, axis=0),
    )

def train_physics_inspired_model(
    Xtr_num, Xtr_cat, Xtr_static, ytr,
    Xva_num, Xva_cat, Xva_static, yva,
    cat_sizes, cat_cols,
    epochs=40,
    batch_size=1024,
):
    model = MultiHeadPredictor(
        seq_num_dim=Xtr_num.shape[-1],
        static_dim=Xtr_static.shape[-1],
        cat_sizes=cat_sizes,
        cat_cols=cat_cols,
        state_dim=128,
    ).to(device)

    opt = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    train_ds = TensorDataset(
        torch.from_numpy(Xtr_num),
        torch.from_numpy(Xtr_cat),
        torch.from_numpy(Xtr_static),
        ytr_t,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))

    best_state = None
    best_val_score = -1e18
    best_epoch = -1

    # loss weights
    lam_q = 0.35
    lam_bce = 0.20
    lam_energy = 0.02  # physics-like regularization

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb_num, xb_cat, xb_static, yb in train_dl:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            xb_static = xb_static.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            y_mean, y_q20, sign_logit, energy = model(xb_num, xb_cat, xb_static)

            loss_mean = awmse_loss(yb, y_mean)
            loss_q20 = pinball_loss(yb, y_q20, tau=0.20)
            yneg = (yb < 0).float()
            loss_sign = weighted_bce_with_logits(sign_logit, yneg, pos_weight=1.67)

            loss = loss_mean + lam_q * loss_q20 + lam_bce * loss_sign + lam_energy * energy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            running += float(loss.cpu().item())

        # evaluate every epoch (VAL is small)
        mean_va, q20_va, pneg_va = predict_model(model, Xva_num, Xva_cat, Xva_static, batch_size=2048)
        # quick default calibration for tracking (no grid each epoch)
        pred_va = apply_conservative_calibration(mean_va, q20_va, pneg_va, alpha=0.30, beta=0.50, delta=0.0)
        val_score, _ = compute_composite_score(yva, pred_va)

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | train_loss={running/len(train_dl):.4f} | val_score={val_score:.4f} | best={best_val_score:.4f}@{best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"✅ Best epoch: {best_epoch} with val_score={best_val_score:.4f}")
    return model

# =============================================================================
# MAIN
# =============================================================================
def main():
    try:
        print("=" * 80)
        print("Method 25 (Rewritten): Physics-Inspired Euler/ResNet Time-Stepping Network")
        print("=" * 80)

        train_df, val_df, test_df = load_all_data()

        # Column inference (train-only for maps/scaler fitting)
        num_cols, cat_cols = infer_columns(train_df)
        print(f"🔧 Using {len(num_cols)} numeric cols, {len(cat_cols)} categorical cols")
        # Fit cat mappings on TRAIN only (no leakage)
        cat_maps, cat_sizes = fit_category_maps(train_df, cat_cols) if len(cat_cols) > 0 else ({}, {})

        # Build sequences (fit scaler on train only)
        Xtr_num, Xtr_cat, Xtr_static, ytr, tr_ids, scaler = build_user_sequences(
            train_df, num_cols, cat_cols, cat_maps=cat_maps, scaler=None, fit_scaler=True, expect_days=7
        )
        Xva_num, Xva_cat, Xva_static, yva, va_ids, _ = build_user_sequences(
            val_df, num_cols, cat_cols, cat_maps=cat_maps, scaler=scaler, fit_scaler=False, expect_days=7
        )
        Xte_num, Xte_cat, Xte_static, yte, te_ids, _ = build_user_sequences(
            test_df, num_cols, cat_cols, cat_maps=cat_maps, scaler=scaler, fit_scaler=False, expect_days=7
        )

        print(f"   Train users: {len(tr_ids)}, Val users: {len(va_ids)}, Test users: {len(te_ids)}")
        print(f"   X_seq_num: {Xtr_num.shape}, X_seq_cat: {Xtr_cat.shape}, X_static: {Xtr_static.shape}")

        # Handle empty categorical case
        if len(cat_cols) == 0:
            cat_sizes = {}
        
        # Train model
        model = train_physics_inspired_model(
            Xtr_num, Xtr_cat, Xtr_static, ytr,
            Xva_num, Xva_cat, Xva_static, yva,
            cat_sizes=cat_sizes,
            cat_cols=cat_cols,
            epochs=35,
            batch_size=1024,
        )

        # Tune calibration on VAL only
        mean_va, q20_va, pneg_va = predict_model(model, Xva_num, Xva_cat, Xva_static)
        calib = tune_calibration_on_val(yva, mean_va, q20_va, pneg_va)

        # Predict TEST
        print("📊 Making predictions on TEST...")
        mean_te, q20_te, pneg_te = predict_model(model, Xte_num, Xte_cat, Xte_static)
        final_predictions = apply_conservative_calibration(
            mean_te, q20_te, pneg_te,
            alpha=calib["alpha"], beta=calib["beta"], delta=calib["delta"]
        )

        # Ground truth on test (user-level)
        y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
        pred_series = pd.Series(final_predictions, index=te_ids)
        pred_aligned = pred_series.reindex(y_test.index).values
        y_test_values = y_test.values

        # Score (TEST) - assign to `score`
        print("\n" + "=" * 60)
        score = compute_pareto_multi_objective(y_test_values, pred_aligned)
        # also keep detailed metrics for logging via compute_composite_score (same logic)
        score2, metrics = compute_composite_score(y_test_values, pred_aligned)
        print("=" * 60)

        print(f"\n🎯 Final Score: {score:.4f}")

        # Save result
        result = {
            "method": "Method 25 (Rewritten): Physics-Inspired Euler/ResNet Time-Stepping Network",
            "score": float(score),
            "score_check_same_logic": float(score2),
            "metrics": metrics,
            "calibration": {k: float(v) for k, v in calib.items() if k in ["alpha", "beta", "delta", "score", "fpr"]},
            "status": "success"
        }

        output_dir = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch"
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "method_25_results.json")

        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        with open(OUTPUT_JSON, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n💾 Results saved to: {result_path}")
        print(f"score = {score}")

    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()

        result = {"method": "Method 25 (Rewritten)", "score": None, "error": str(e), "status": "failed"}
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(result, f, indent=2)
        raise

if __name__ == "__main__":
    main()
