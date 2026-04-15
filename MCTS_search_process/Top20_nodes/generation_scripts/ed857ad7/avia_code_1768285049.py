# Suppress warnings to prevent false failures
import warnings
warnings.filterwarnings('ignore')

#!/usr/bin/env python3
"""
Physics-inspired Deep Residual Time-Stepping Network (Euler/ResNet view)

Implements a sequence model motivated by the strong connection between:
- Residual Networks (ResNets) and
- Explicit Euler discretization of an ODE:  h_{t+1} = h_t + dt * f(h_t, x_t)

Adapted to BT_IOS_2503_Pareto format:
- Each user has 7 daily rows (TDATE_RN=1..7)
- We build a per-user 7-step sequence and predict one D60 signed LTV per user.

CRITICAL:
- Scoring functions are preserved EXACTLY as provided (calc_gini, compute_score).
- Final score computed on TEST, assigned to variable named `score`,
  and printed with: print(f"score = {score}")
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TRAIN_PATH = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"
DAY_COL = "TDATE_RN"

# Kept from template (not required by this method, but we preserve the variable)
NUMERICAL_COLS = ['DEPOSIT_AMOUNT', 'REC_USD', 'REC_USD_CUM', 'REC_USD_D6', 'CPI',
    'RANK1_PLAY_CNT_ALL', 'PLAY_CNT_ALL', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_REWARD_CASH', 'PLAY_CNT_CASH', 'HIGHFEE_PLAY_CNT_CASH',
    'CASH_RATIO', 'ACTIVE_DAYS_ALL_CUM', 'PLAY_CNT_ALL_CUM', 'SESSION_CNT_ALL']


# =========================
# SCORING FUNCTIONS (DO NOT MODIFY)  ✅
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


# =========================
# Utilities
# =========================
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_day_col(df: pd.DataFrame) -> pd.DataFrame:
    if DAY_COL not in df.columns:
        df = df.copy()
        df[DAY_COL] = df.groupby(ID_COL).cumcount() + 1
    return df

def infer_feature_columns(df: pd.DataFrame):
    """Infer numeric + categorical columns from df (no target leakage)."""
    df = ensure_day_col(df)
    exclude = {ID_COL, TARGET_COL, DAY_COL}

    # numeric candidates
    numeric_cols = []
    cat_cols = []

    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    # Optionally treat low-cardinality integer columns as categorical:
    # (Conservative: only if dtype is integer-like and low unique)
    for c in list(numeric_cols):
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= 64:
            numeric_cols.remove(c)
            cat_cols.append(c)

    numeric_cols = sorted(numeric_cols)
    cat_cols = sorted(cat_cols)
    return numeric_cols, cat_cols

def fit_numeric_imputer_scaler(train_df: pd.DataFrame, numeric_cols):
    """Compute mean/std on training rows; impute NaN with mean; standardize."""
    means = {}
    stds = {}
    for c in numeric_cols:
        x = pd.to_numeric(train_df[c], errors="coerce")
        m = float(np.nanmean(x.values))
        s = float(np.nanstd(x.values))
        if not np.isfinite(m):
            m = 0.0
        if not np.isfinite(s) or s < 1e-6:
            s = 1.0
        means[c] = m
        stds[c] = s
    return means, stds

def build_cat_maps(train_df: pd.DataFrame, cat_cols):
    """Map categories to integers with 0 reserved for UNK/MISSING."""
    maps = {}
    for c in cat_cols:
        s = train_df[c].astype("object").fillna("__MISSING__")
        uniq = pd.unique(s)
        # 0 => unknown
        mapping = {k: i+1 for i, k in enumerate(uniq)}
        maps[c] = mapping
    return maps

def standardize_numeric(df: pd.DataFrame, numeric_cols, means, stds) -> np.ndarray:
    X = np.zeros((len(df), len(numeric_cols)), dtype=np.float32)
    for j, c in enumerate(numeric_cols):
        x = pd.to_numeric(df[c], errors="coerce").astype(np.float32).to_numpy()
        m, s = means[c], stds[c]
        x = np.where(np.isfinite(x), x, m).astype(np.float32)
        X[:, j] = (x - m) / s
    return X

def encode_cats(df: pd.DataFrame, cat_cols, cat_maps) -> np.ndarray:
    if len(cat_cols) == 0:
        return np.zeros((len(df), 0), dtype=np.int64)
    X = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        mapping = cat_maps[c]
        s = df[c].astype("object").fillna("__MISSING__")
        X[:, j] = s.map(mapping).fillna(0).astype(np.int64).to_numpy()
    return X

def make_complete_user_day_grid(df: pd.DataFrame, user_ids: np.ndarray) -> pd.DataFrame:
    """Reindex to ensure exactly 7 rows per user (days 1..7)."""
    idx = pd.MultiIndex.from_product([user_ids, np.arange(1, 8)], names=[ID_COL, DAY_COL])
    out = df.set_index([ID_COL, DAY_COL]).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.reindex(idx).reset_index()
    return out

def build_user_tensors(
    df: pd.DataFrame,
    numeric_cols,
    cat_cols,
    means, stds,
    cat_maps,
    user_order: np.ndarray = None
):
    """
    Returns:
      users: (U,)
      X_num: (U,7,N)
      X_cat: (U,7,C)
      X_static: (U,S)
      y_user: (U,) or None if target not present
    """
    df = ensure_day_col(df).copy()

    if user_order is None:
        users = df[ID_COL].drop_duplicates().to_numpy()
    else:
        users = np.asarray(user_order)

    # Make sure we have 7 days per user in consistent ordering
    df_grid = make_complete_user_day_grid(df, users)
    df_grid = df_grid.sort_values([ID_COL, DAY_COL], kind="mergesort").reset_index(drop=True)

    # Build y per user from original df (not grid), to avoid NaN if grid created missing rows
    y_user = None
    if TARGET_COL in df.columns:
        y_user = df.groupby(ID_COL)[TARGET_COL].first().reindex(users).to_numpy(dtype=np.float32)

    # Numeric + categorical per row (standardized/encoded)
    X_num_2d = standardize_numeric(df_grid, numeric_cols, means, stds)  # (U*7, N)
    X_cat_2d = encode_cats(df_grid, cat_cols, cat_maps)                # (U*7, C)

    U = len(users)
    N = len(numeric_cols)
    C = len(cat_cols)

    X_num = X_num_2d.reshape(U, 7, N)
    X_cat = X_cat_2d.reshape(U, 7, C)

    # Static aggregations (computed on standardized numerics)
    if N > 0:
        x_last = X_num[:, -1, :]
        x_mean = X_num.mean(axis=1)
        x_std  = X_num.std(axis=1)
        x_trend = X_num[:, -1, :] - X_num[:, 0, :]
        X_static = np.concatenate([x_last, x_mean, x_std, x_trend], axis=1).astype(np.float32)
    else:
        X_static = np.zeros((U, 0), dtype=np.float32)

    return users, X_num.astype(np.float32), X_cat.astype(np.int64), X_static, y_user


# =========================
# Dataset
# =========================
class UserSeqDataset(Dataset):
    def __init__(self, X_num, X_cat, X_static, y=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.X_static = X_static
        self.y = y

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        x_num = self.X_num[idx]
        x_cat = self.X_cat[idx]
        x_static = self.X_static[idx]
        if self.y is None:
            return x_num, x_cat, x_static
        return x_num, x_cat, x_static, self.y[idx]


# =========================
# Physics-inspired Euler/ResNet Sequence Model
# =========================
class EulerBlock(nn.Module):
    def __init__(self, d_state: int, d_in: int, d_hidden: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_state + d_in)
        self.fc1 = nn.Linear(d_state + d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, x):
        z = torch.cat([h, x], dim=-1)
        z = self.ln(z)
        z = self.fc1(z)
        z = F.gelu(z)
        z = self.drop(z)
        z = self.fc2(z)
        return z

class EulerResNetSeqModel(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities,
        d_cat: int = 16,
        d_state: int = 96,
        d_hidden: int = 192,
        dropout: float = 0.10,
        dt: float = 1.0
    ):
        super().__init__()
        self.n_num = n_num
        self.cat_cardinalities = list(cat_cardinalities)
        self.n_cat = len(self.cat_cardinalities)

        # Categorical embeddings
        self.cat_embs = nn.ModuleList([
            nn.Embedding(num_embeddings=card + 1, embedding_dim=d_cat)  # +1 for 0=UNK
            for card in self.cat_cardinalities
        ])

        d_in = n_num + self.n_cat * d_cat + 8  # + day embedding
        self.day_emb = nn.Embedding(8, 8)  # days 1..7 (0 unused)

        self.x_proj = nn.Linear(d_in, d_state)
        self.h0 = nn.Parameter(torch.zeros(d_state))

        # Shared Euler residual block across time (like time-invariant dynamics)
        self.block = EulerBlock(d_state=d_state, d_in=d_state, d_hidden=d_hidden, dropout=dropout)
        self.dt = dt

        # Pooling & heads
        self.static_proj = None
        self.base_head = None

        # static skip/linear baseline (residual view: prediction = baseline + deep residual)
        self._static_in = None  # set at runtime via build_heads

        self.head_ln = nn.LayerNorm(d_state * 3 + 64)
        self.head_mean = nn.Sequential(
            nn.Linear(d_state * 3 + 64, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1)
        )
        self.head_q20 = nn.Sequential(
            nn.Linear(d_state * 3 + 64, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1)
        )
        self.head_neg = nn.Sequential(
            nn.Linear(d_state * 3 + 64, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1)
        )

    def build_heads(self, static_dim: int):
        self._static_in = static_dim
        # compress static features into 64-d for fusion
        self.static_proj = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
        )
        # baseline (linear) skip connection
        self.base_head = nn.Linear(static_dim, 1)

    def forward(self, x_num, x_cat, x_static):
        """
        x_num: (B,7,n_num)
        x_cat: (B,7,n_cat)
        x_static: (B,S)
        """
        B, T, _ = x_num.shape
        assert T == 7

        # Build per-day input tokens
        parts = [x_num]
        if self.n_cat > 0:
            embs = []
            for j, emb in enumerate(self.cat_embs):
                embs.append(emb(x_cat[:, :, j]))
            parts.append(torch.cat(embs, dim=-1))

        day_ids = torch.arange(1, 8, device=x_num.device).view(1, 7).expand(B, 7)
        day_e = self.day_emb(day_ids)  # (B,7,8)
        parts.append(day_e)

        x = torch.cat(parts, dim=-1)          # (B,7,d_in)
        x = self.x_proj(x)                   # (B,7,d_state)

        # Euler time stepping / ResNet interpretation:
        # h_{t+1} = h_t + dt * f(h_t, x_t)
        h = self.h0.unsqueeze(0).expand(B, -1)
        hs = []
        for t in range(7):
            xt = x[:, t, :]
            dh = self.block(h, xt)           # uses concat(h, xt) inside
            h = h + self.dt * dh
            hs.append(h)

        H = torch.stack(hs, dim=1)  # (B,7,d_state)
        h_last = H[:, -1, :]
        h_mean = H.mean(dim=1)
        h_max = H.max(dim=1).values

        s64 = self.static_proj(x_static)
        z = torch.cat([h_last, h_mean, h_max, s64], dim=-1)
        z = self.head_ln(z)

        base = self.base_head(x_static).squeeze(-1)
        mean = base + self.head_mean(z).squeeze(-1)
        q20  = base + self.head_q20(z).squeeze(-1)
        neg_logit = self.head_neg(z).squeeze(-1)
        return mean, q20, neg_logit


# =========================
# Losses (training loss is flexible)
# =========================
def awmse_smooth(y, yhat, k=0.08):
    """
    Differentiable approximation of asymmetric weighting based on sign mismatch.
    Uses smooth probability of predicting positive: sigmoid(k*yhat).

    Weight rules (from task hints):
    - False Positive (y<0, yhat>0): 2.5 + 0.02*|y|
    - False Negative (y>0, yhat<0): 1.5 + 0.01*y
    - Otherwise: 1.0
    """
    y = y.view(-1)
    yhat = yhat.view(-1)

    y_neg = (y < 0).float()
    y_pos = (y > 0).float()

    p_pos = torch.sigmoid(k * yhat)      # prob yhat > 0 (smooth)
    p_neg = 1.0 - p_pos

    w_fp = 2.5 + 0.02 * torch.abs(y)
    w_fn = 1.5 + 0.01 * torch.clamp(y, min=0.0)

    w = torch.ones_like(y)
    w = w + (w_fp - 1.0) * p_pos * y_neg
    w = w + (w_fn - 1.0) * p_neg * y_pos
    w = torch.clamp(w, 0.1, 10.0)

    return torch.mean(w * (yhat - y) ** 2)

def pinball_loss(y, qhat, tau=0.20):
    e = y - qhat
    return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))

def fpr(pred, y_true):
    pred_pos = pred > 0
    denom = np.sum(pred_pos)
    if denom == 0:
        return 0.0
    return float(np.sum(pred_pos & (y_true < 0)) / denom)


# =========================
# Train / Predict
# =========================
@torch.no_grad()
def predict_model(model, loader, device, postprocess=True):
    model.eval()
    preds = []
    q20s = []
    pnegs = []
    for batch in loader:
        if len(batch) == 3:
            x_num, x_cat, x_static = batch
        else:
            x_num, x_cat, x_static, _y = batch
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        x_static = x_static.to(device)

        mean, q20, neg_logit = model(x_num, x_cat, x_static)
        mean = mean.detach().cpu().numpy()
        q20 = q20.detach().cpu().numpy()
        pneg = torch.sigmoid(neg_logit).detach().cpu().numpy()

        preds.append(mean)
        q20s.append(q20)
        pnegs.append(pneg)

    mean = np.concatenate(preds, axis=0)
    q20 = np.concatenate(q20s, axis=0)
    pneg = np.concatenate(pnegs, axis=0)

    if not postprocess:
        return mean, q20, pneg

    # Conservative combination near the zero boundary:
    # if negative risk is high OR conservative quantile is <0, lean conservative.
    final = mean.copy()
    mask = (pneg > 0.55) | (q20 < 0.0)
    final[mask] = np.minimum(mean[mask], q20[mask])
    return final, q20, pneg

def train_model(
    model,
    train_loader,
    val_loader,
    y_val,
    device,
    max_epochs=30,
    lr=2e-3,
    wd=2e-4,
    patience=6
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_score = -1e18
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for x_num, x_cat, x_static, y in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            x_static = x_static.to(device)
            y = y.to(device)

            mean, q20, neg_logit = model(x_num, x_cat, x_static)

            # Multi-task training:
            # - regression with smooth AWMSE
            # - conservative quantile head (tau=0.20)
            # - sign (negative) classifier head
            loss_reg = awmse_smooth(y, mean)
            loss_q = pinball_loss(y, q20, tau=0.20)

            y_neg = (y < 0).float()
            # emphasize catching negatives (avoid predicting profitable when costly)
            pos_weight = torch.tensor(2.0, device=device)
            loss_bce = F.binary_cross_entropy_with_logits(neg_logit, y_neg, pos_weight=pos_weight)

            loss = loss_reg + 0.35 * loss_q + 0.10 * loss_bce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.detach().cpu().item()) * len(y)
            n += len(y)

        # Validate using provided compute_score (no modification)
        val_pred, _q, _p = predict_model(model, val_loader, device, postprocess=True)
        val_score, _ = compute_score(y_val, val_pred)

        avg_loss = total / max(1, n)
        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.6f} | val_score={val_score:.6f}")

        if val_score > best_score + 1e-6:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch} (best val_score={best_score:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def select_delta_on_val(y_val, val_pred):
    """
    Post-hoc conservative calibration: choose a global shift delta using validation only.
    Applies: pred_shifted = pred - delta
    Optionally enforce FPR < 0.40 first, then maximize score.
    """
    candidates = np.linspace(-200, 200, 161)  # coarse but robust
    best = (-1e18, 0.0, None)  # (score, delta, fpr)

    feasible = []
    for d in candidates:
        p = val_pred - d
        cur_fpr = fpr(p, y_val)
        sc, _ = compute_score(y_val, p)
        if cur_fpr <= 0.40:
            feasible.append((sc, d, cur_fpr))

    if len(feasible) > 0:
        feasible.sort(key=lambda x: x[0], reverse=True)
        best_sc, best_d, best_fpr = feasible[0]
        print(f"[Calibration] Selected delta={best_d:.2f} with FPR={best_fpr:.3f} (val_score={best_sc:.6f})")
        return float(best_d)

    # fallback: maximize score even if FPR constraint not met
    for d in candidates:
        p = val_pred - d
        cur_fpr = fpr(p, y_val)
        sc, _ = compute_score(y_val, p)
        if sc > best[0]:
            best = (sc, d, cur_fpr)
    print(f"[Calibration] No FPR-feasible delta found. Using delta={best[1]:.2f} (FPR={best[2]:.3f}, val_score={best[0]:.6f})")
    return float(best[1])


# =========================
# Main
# =========================
def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Physics-inspired Euler/ResNet Sequence Model (Computational Physics view)")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    train_df = ensure_day_col(train_df)
    val_df   = ensure_day_col(val_df)
    test_df  = ensure_day_col(test_df)

    # Infer columns only from training schema (no test usage)
    numeric_cols, cat_cols = infer_feature_columns(train_df)
    print(f"Detected features: {len(numeric_cols)} numeric, {len(cat_cols)} categorical")

    # Fit preprocessing on TRAIN only
    means, stds = fit_numeric_imputer_scaler(train_df, numeric_cols)
    cat_maps = build_cat_maps(train_df, cat_cols)
    cat_cardinalities = [len(cat_maps[c]) + 1 for c in cat_cols]  # +1 for UNK already in mapping? (we used i+1 so max id == len(map))

    # Build tensors per split (user-level sequences)
    tr_users, Xtr_num, Xtr_cat, Xtr_static, ytr = build_user_tensors(
        train_df, numeric_cols, cat_cols, means, stds, cat_maps
    )
    va_users, Xva_num, Xva_cat, Xva_static, yva = build_user_tensors(
        val_df, numeric_cols, cat_cols, means, stds, cat_maps
    )
    te_users, Xte_num, Xte_cat, Xte_static, yte = build_user_tensors(
        test_df, numeric_cols, cat_cols, means, stds, cat_maps
    )

    print(f"Users: train={len(tr_users)}, val={len(va_users)}, test={len(te_users)}")
    print(f"Tensor shapes: X_num={Xtr_num.shape}, X_cat={Xtr_cat.shape}, X_static={Xtr_static.shape}")

    # Dataloaders
    batch_size = 512
    train_ds = UserSeqDataset(Xtr_num, Xtr_cat, Xtr_static, ytr)
    val_ds   = UserSeqDataset(Xva_num, Xva_cat, Xva_static, yva)
    test_ds  = UserSeqDataset(Xte_num, Xte_cat, Xte_static, yte)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = EulerResNetSeqModel(
        n_num=len(numeric_cols),
        cat_cardinalities=[len(cat_maps[c]) + 1 for c in cat_cols],
        d_cat=16,
        d_state=96,
        d_hidden=192,
        dropout=0.10,
        dt=1.0
    )
    model.build_heads(static_dim=Xtr_static.shape[1])
    model = model.to(device)

    # Train with early stopping (val score)
    model, best_val_score = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_val=yva,
        device=device,
        max_epochs=30,
        lr=2e-3,
        wd=2e-4,
        patience=6
    )

    # Validation-time conservative delta calibration (NO test usage)
    val_pred, _, _ = predict_model(model, val_loader, device, postprocess=True)
    delta = select_delta_on_val(yva, val_pred)

    # Predict test
    test_pred, _, _ = predict_model(model, test_loader, device, postprocess=True)
    test_pred = test_pred - delta

    # Align to test ground truth order (user-level)
    # yte is already in te_users order, but ensure alignment by reindex for safety.
    y_test_series = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_series = pd.Series(test_pred, index=te_users)
    pred_aligned = pred_series.reindex(y_test_series.index).to_numpy()

    print("\n" + "=" * 60)
    score, metrics = compute_score(y_test_series.values, pred_aligned)
    print("=" * 60)
    print(f"\n🎯 Final Score: {score:.4f}")

    result = {
        "method": "Physics-inspired Euler/ResNet Sequence Model",
        "device": str(device),
        "delta": float(delta),
        "score": float(score),
        "metrics": metrics,
        "best_val_score": float(best_val_score),
        "n_numeric": int(len(numeric_cols)),
        "n_categorical": int(len(cat_cols)),
        "static_dim": int(Xtr_static.shape[1]),
    }
    out_path = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/method_11_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"score = {score}")


if __name__ == "__main__":
    main()
