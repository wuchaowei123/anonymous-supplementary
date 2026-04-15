#!/usr/bin/env python3
"""
AviaAgentMonty - Execution Node: 80466eed
Type: original
Generated: 2026-01-12T22:01:48.401413
Mutation: replicate
Parent: None

DO NOT DELETE - This file is preserved for reproducibility.
"""
#!/usr/bin/env python3
"""
Method 18 (rewritten): Physics-inspired Deep Learning (Euler/ResNet time-stepping)

Key technique adapted from "Deep Learning and Computational Physics (Lecture Notes)":
- Treat a residual network as an explicit Euler discretization of a dynamical system.
  h_{t+1} = h_t + dt * f_theta(h_t, u_t)
- Use stable training practices analogous to computational physics: normalization,
  regularization (weight decay), and conservative calibration.

This implementation:
- Builds 7-day user sequences (no naive aggregation-only modeling)
- Uses categorical embeddings + numeric temporal features (raw, diffs, signed-log)
- Multi-task heads: sign (P(y<0)), mean regression, quantile regression (tau=0.20)
- Post-hoc conservative delta calibration on VAL to manage false positives near 0
- Scores on TEST using the provided scoring functions (UNMODIFIED)
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRAIN_PATH = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"
TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"

NUMERICAL_COLS = ['DEPOSIT_AMOUNT', 'REC_USD', 'REC_USD_CUM', 'REC_USD_D6', 'CPI',
    'RANK1_PLAY_CNT_ALL', 'PLAY_CNT_ALL', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_REWARD_CASH', 'PLAY_CNT_CASH', 'HIGHFEE_PLAY_CNT_CASH',
    'CASH_RATIO', 'ACTIVE_DAYS_ALL_CUM', 'PLAY_CNT_ALL_CUM']

# =============================================================================
# ⚠️ SCORING FUNCTIONS - DO NOT MODIFY (copied EXACTLY)
# =============================================================================
def calc_gini(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    if len(y_true) == 0 or np.sum(y_true) <= 0: return 0.0
    order = np.argsort(y_pred)[::-1]
    cumsum = np.cumsum(y_true[order])
    total = cumsum[-1]
    if total == 0: return 0.0
    lorenz = cumsum / total
    gini_actual = 2 * np.sum(lorenz) / len(y_true) - 1
    lorenz_perfect = np.cumsum(np.sort(y_true)[::-1]) / total
    gini_perfect = 2 * np.sum(lorenz_perfect) / len(y_true) - 1
    return gini_actual / gini_perfect if gini_perfect != 0 else 1.0

def compute_score(y_true, y_pred):
    gini = calc_gini(y_true, y_pred)
    error_rate = abs(np.sum(y_true) - np.sum(y_pred)) / abs(np.sum(y_true))
    spearman = spearmanr(y_true, y_pred)[0]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    gini_score = np.clip((gini - 0.70) / 0.20, 0, 1)
    error_score = np.clip((0.35 - error_rate) / 0.34, 0, 1)
    spearman_score = np.clip((spearman - 0.50) / 0.30, 0, 1)
    rmse_score = np.clip((260 - rmse) / 60, 0, 1)
    
    base = 0.35*gini_score + 0.25*spearman_score + 0.20*rmse_score + 0.20*error_score
    pareto = sum([0.035*(gini_score>0.8), 0.025*(spearman_score>0.8), 0.02*(rmse_score>0.8), 0.02*(error_score>0.8)])
    exc = sum([gini_score>0.8, spearman_score>0.8, rmse_score>0.8, error_score>0.8])
    if exc >= 2: pareto += 0.02
    if exc >= 3: pareto += 0.03
    if exc == 4: pareto += 0.05
    
    final = base + pareto
    print(f"📊 Gini={gini:.4f}, Err={error_rate:.4f}, Spear={spearman:.4f}, RMSE={rmse:.2f}, Score={final:.4f}")
    return final, {'gini': gini, 'error_rate': error_rate, 'spearman': spearman, 'rmse': rmse}

# =============================================================================
# Reproducibility
# =============================================================================
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# =============================================================================
# Data utilities (sequence building)
# =============================================================================
def ensure_day_column(df: pd.DataFrame) -> pd.DataFrame:
    if "TDATE_RN" not in df.columns:
        df = df.copy()
        df["TDATE_RN"] = df.groupby(ID_COL).cumcount() + 1
    return df

def infer_feature_columns(train_df: pd.DataFrame, val_df: pd.DataFrame):
    # Categorical: object/category columns (excluding ID_COL)
    cat_cols = []
    for c in train_df.columns:
        if c in [ID_COL, TARGET_COL]:
            continue
        if str(train_df[c].dtype) in ("object", "category"):
            cat_cols.append(c)

    # Numeric: numeric dtypes excluding ID/TARGET and excluding day index (used for ordering + day embedding)
    numeric_cols = []
    for c in train_df.columns:
        if c in [ID_COL, TARGET_COL, "TDATE_RN"]:
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]):
            numeric_cols.append(c)

    # Ensure stable column order
    cat_cols = sorted(cat_cols)
    numeric_cols = sorted(numeric_cols)
    return numeric_cols, cat_cols

def build_cat_maps(train_df: pd.DataFrame, val_df: pd.DataFrame, cat_cols):
    # Map categories using TRAIN+VAL only (labels-only; not target leakage)
    cat_maps = {}
    for c in cat_cols:
        s = pd.concat([train_df[c], val_df[c]], axis=0, ignore_index=True)
        s = s.fillna("UNK").astype(str)
        uniq = pd.Index(s.unique())
        # reserve 0 = UNK
        mapping = {k: i + 1 for i, k in enumerate(uniq.tolist())}
        cat_maps[c] = mapping
    return cat_maps

def encode_cats(df: pd.DataFrame, cat_cols, cat_maps):
    if not cat_cols:
        return df
    df = df.copy()
    for c in cat_cols:
        m = cat_maps[c]
        df[c] = df[c].fillna("UNK").astype(str).map(m).fillna(0).astype(np.int64)
    return df

def compute_user_aggregates(df: pd.DataFrame, numeric_cols_for_agg):
    # Aggregates across the 7-day window; uses a limited, meaningful subset (NUMERICAL_COLS)
    cols = [c for c in numeric_cols_for_agg if c in df.columns]
    if not cols:
        # fallback empty
        out = df[[ID_COL]].drop_duplicates().copy()
        out["agg_dummy"] = 0.0
        return out

    agg = {col: ["sum", "mean", "max", "min", "std"] for col in cols}
    uf = df.groupby(ID_COL).agg(agg)
    uf.columns = ["_".join(c) for c in uf.columns]
    uf = uf.reset_index().fillna(0.0)
    return uf

def build_user_sequences(df: pd.DataFrame,
                         numeric_cols,
                         cat_cols,
                         with_target: bool,
                         max_days: int = 7):
    """
    Returns:
      user_ids: (N,)
      X_num: (N, T, Dn) float32
      X_cat: (N, T, Dc) int64 (or None if no cats)
      X_agg: (N, Da) float32
      y: (N,) float32 (or None)
    """
    df = ensure_day_column(df)
    df = df.sort_values([ID_COL, "TDATE_RN"])
    user_ids = df[ID_COL].drop_duplicates().values
    n_users = len(user_ids)

    # Prepare aggregates
    agg_df = compute_user_aggregates(df, NUMERICAL_COLS)
    agg_df = agg_df.set_index(ID_COL).reindex(user_ids).fillna(0.0)
    X_agg = agg_df.drop(columns=[ID_COL], errors="ignore").values.astype(np.float32)
    if X_agg.ndim == 1:
        X_agg = X_agg.reshape(-1, 1)

    # Prepare numeric sequences
    Dn = len(numeric_cols)
    X_num_raw = np.zeros((n_users, max_days, Dn), dtype=np.float32)

    # Prepare categorical sequences
    Dc = len(cat_cols)
    X_cat = None
    if Dc > 0:
        X_cat = np.zeros((n_users, max_days, Dc), dtype=np.int64)

    # Targets at user-level
    y = None
    if with_target:
        y = df.groupby(ID_COL)[TARGET_COL].first().reindex(user_ids).values.astype(np.float32)

    # Fill sequences
    g = df.groupby(ID_COL, sort=False)
    for i, uid in enumerate(user_ids):
        part = g.get_group(uid)
        part = part.sort_values("TDATE_RN")
        part = part.head(max_days)

        if Dn > 0:
            X_num_raw[i, :len(part), :] = part[numeric_cols].fillna(0.0).values.astype(np.float32)

        if Dc > 0:
            X_cat[i, :len(part), :] = part[cat_cols].fillna(0).values.astype(np.int64)

    # Temporal engineering: diffs + signed log transform
    X_diff = np.zeros_like(X_num_raw)
    X_diff[:, 1:, :] = X_num_raw[:, 1:, :] - X_num_raw[:, :-1, :]
    X_slog = np.sign(X_num_raw) * np.log1p(np.abs(X_num_raw))

    X_num = np.concatenate([X_num_raw, X_diff, X_slog], axis=-1).astype(np.float32)
    return user_ids, X_num, X_cat, X_agg, y

def fit_standardizer(X_num_train: np.ndarray):
    # X_num_train: (N, T, D)
    flat = X_num_train.reshape(-1, X_num_train.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X_num: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X_num - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

# =============================================================================
# Dataset
# =============================================================================
class UserSeqDataset(Dataset):
    def __init__(self, X_num, X_cat, X_agg, y=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.X_agg = X_agg
        self.y = y

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        xnum = torch.from_numpy(self.X_num[idx])          # (T, Dn)
        xagg = torch.from_numpy(self.X_agg[idx])          # (Da,)
        if self.X_cat is None:
            xcat = None
        else:
            xcat = torch.from_numpy(self.X_cat[idx])      # (T, Dc)

        if self.y is None:
            return xnum, xcat, xagg
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return xnum, xcat, xagg, y

# =============================================================================
# Physics-inspired model: Euler time-stepping (ResNet/Neural ODE discretization)
# =============================================================================
class EulerResNetEncoder(nn.Module):
    def __init__(self,
                 num_dim: int,
                 cat_cardinalities,
                 cat_emb_dims,
                 agg_dim: int,
                 hidden_dim: int = 192,
                 day_emb_dim: int = 8,
                 dropout: float = 0.10,
                 dt: float = 1.0,
                 max_days: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.max_days = max_days

        # Categorical embeddings
        self.cat_cols = len(cat_cardinalities) if cat_cardinalities else 0
        self.cat_embs = nn.ModuleList()
        total_cat_emb = 0
        for card, ed in zip(cat_cardinalities, cat_emb_dims):
            self.cat_embs.append(nn.Embedding(card + 1, ed))  # +1 for index 0 (UNK)
            total_cat_emb += ed

        # Day embedding (computational-physics analog: time coordinate)
        self.day_emb = nn.Embedding(max_days + 1, day_emb_dim)  # day: 1..7, 0 unused

        in_day = num_dim + total_cat_emb + day_emb_dim

        # Initial condition mapping
        self.init = nn.Sequential(
            nn.Linear(in_day, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Dynamics f_theta(h, u)
        self.f = nn.Sequential(
            nn.Linear(hidden_dim + in_day, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.f_norm = nn.LayerNorm(hidden_dim)

        # Heads (multi-task)
        # Concatenate hidden state with user aggregates (like global constraints/boundary data)
        head_in = hidden_dim + agg_dim
        self.mean_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.q20_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.neg_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num, x_cat, x_agg):
        """
        x_num: (B, T, Dnum)
        x_cat: (B, T, Dcat) or None
        x_agg: (B, Dagg)
        """
        B, T, _ = x_num.shape
        device = x_num.device

        # Prepare day indices 1..T
        day_idx = torch.arange(1, T + 1, device=device).unsqueeze(0).expand(B, T)  # (B,T)
        day_e = self.day_emb(day_idx)  # (B,T,day_emb_dim)

        # Cat embeddings per day
        if x_cat is None or self.cat_cols == 0:
            cat_e = None
            total_in = torch.cat([x_num, day_e], dim=-1)
        else:
            embs = []
            for j, emb in enumerate(self.cat_embs):
                embs.append(emb(x_cat[:, :, j]))
            cat_e = torch.cat(embs, dim=-1)  # (B,T,sum_emb)
            total_in = torch.cat([x_num, cat_e, day_e], dim=-1)

        # Euler/ResNet time stepping
        h = self.init(total_in[:, 0, :])  # h0 from day 1

        for t in range(T):
            u = total_in[:, t, :]  # (B, in_day)
            dh = self.f(torch.cat([h, u], dim=-1))
            dh = self.f_norm(dh)
            h = h + self.dt * dh

        z = torch.cat([h, x_agg], dim=-1)

        mean = self.mean_head(z).squeeze(-1)
        q20 = self.q20_head(z).squeeze(-1)
        neg_logit = self.neg_head(z).squeeze(-1)
        return mean, q20, neg_logit

# =============================================================================
# Losses (asymmetric + quantile + sign)
# =============================================================================
def awmse_loss(y_true, y_pred):
    """
    Asymmetric Weighted MSE (piecewise weights):
      - False Positive (pred>0, y<0): 2.5 + 0.02|y|
      - False Negative (pred<0, y>0): 1.5 + 0.01 y
      - Otherwise: 1.0
    """
    y = y_true
    yp = y_pred
    w = torch.ones_like(y)

    fp = (yp > 0) & (y < 0)
    fn = (yp < 0) & (y > 0)

    w = torch.where(fp, 2.5 + 0.02 * torch.abs(y), w)
    w = torch.where(fn, 1.5 + 0.01 * y, w)

    return torch.mean(w * (yp - y) ** 2)

def pinball_loss(y_true, y_q, tau: float = 0.20):
    e = y_true - y_q
    return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))

def blend_predictions(mean, q20, neg_logit, risk_scale: float = 0.60, thr: float = 0.60):
    """
    Conservative blending using:
    - q20 as lower-bound estimate
    - p_neg to dampen optimistic predictions near zero
    """
    p_neg = torch.sigmoid(neg_logit)

    base = 0.70 * mean + 0.30 * q20
    # uncertainty proxy: |mean - q20|
    adj = base - risk_scale * p_neg * torch.abs(mean - q20)

    # hard conservative guards
    adj = torch.where(q20 < 0, torch.minimum(adj, q20), adj)
    adj = torch.where(p_neg > thr, torch.minimum(adj, q20), adj)
    return adj

def fpr(y_true_np, y_pred_np):
    y_true_np = np.asarray(y_true_np).reshape(-1)
    y_pred_np = np.asarray(y_pred_np).reshape(-1)
    pos_pred = y_pred_np > 0
    if pos_pred.sum() == 0:
        return 0.0
    return float((y_true_np[pos_pred] < 0).mean())

# =============================================================================
# Train / Eval
# =============================================================================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        if len(batch) == 3:
            xnum, xcat, xagg = batch
        else:
            xnum, xcat, xagg, _y = batch
        xnum = xnum.to(device)
        xagg = xagg.to(device)
        xcat = None if xcat is None else xcat.to(device)

        mean, q20, neg_logit = model(xnum, xcat, xagg)
        yhat = blend_predictions(mean, q20, neg_logit)
        preds.append(yhat.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)

def train_one_epoch(model, loader, optimizer, device, bce_pos_weight, tau=0.20,
                    lam_q=0.30, lam_bce=0.20, grad_clip=1.0, use_amp=True):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight], device=device))

    total = 0.0
    n = 0
    for xnum, xcat, xagg, y in loader:
        xnum = xnum.to(device)
        xagg = xagg.to(device)
        y = y.to(device)
        xcat = None if xcat is None else xcat.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            mean, q20, neg_logit = model(xnum, xcat, xagg)
            loss_reg = awmse_loss(y, mean)
            loss_q = pinball_loss(y, q20, tau=tau)
            y_neg = (y < 0).float()
            loss_s = bce(neg_logit, y_neg)
            loss = loss_reg + lam_q * loss_q + lam_bce * loss_s

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = y.shape[0]
        total += float(loss.detach().cpu().item()) * bs
        n += bs

    return total / max(n, 1)

def calibrate_delta_on_val(y_val, pred_val, deltas=None, require_fpr_le=0.40):
    if deltas is None:
        deltas = np.arange(-200.0, 201.0, 5.0)

    best = None
    best_score = -1e18
    best_metrics = None

    # Prefer deltas meeting FPR constraint; if none meet, pick best score anyway.
    feasible_found = False
    for d in deltas:
        shifted = pred_val - d
        cur_fpr = fpr(y_val, shifted)
        cur_score, cur_metrics = compute_score(y_val, shifted)

        feasible = (cur_fpr <= require_fpr_le)
        if feasible and not feasible_found:
            feasible_found = True
            best_score = -1e18  # reset so first feasible can win

        if feasible_found and not feasible:
            continue

        if cur_score > best_score:
            best_score = cur_score
            best = d
            best_metrics = dict(cur_metrics)
            best_metrics["fpr"] = cur_fpr

    return float(best if best is not None else 0.0), float(best_score), best_metrics

# =============================================================================
# Main
# =============================================================================
def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("=" * 60)
    print("Method 18: Physics-inspired Euler/ResNet (Neural ODE discretization)")
    print("=" * 60)
    print(f"Device: {device}")

    # Load
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = ensure_day_column(train_df)
    val_df = ensure_day_column(val_df)
    test_df = ensure_day_column(test_df)

    # Infer columns + encode categories
    numeric_cols, cat_cols = infer_feature_columns(train_df, val_df)
    cat_maps = build_cat_maps(train_df, val_df, cat_cols)
    train_df = encode_cats(train_df, cat_cols, cat_maps)
    val_df = encode_cats(val_df, cat_cols, cat_maps)
    test_df = encode_cats(test_df, cat_cols, cat_maps)

    # Ensure numeric types
    for c in numeric_cols:
        train_df[c] = train_df[c].fillna(0.0).astype(np.float32)
        val_df[c] = val_df[c].fillna(0.0).astype(np.float32)
        test_df[c] = test_df[c].fillna(0.0).astype(np.float32)

    # Build sequences
    tr_ids, Xtr_num, Xtr_cat, Xtr_agg, ytr = build_user_sequences(train_df, numeric_cols, cat_cols, with_target=True)
    va_ids, Xva_num, Xva_cat, Xva_agg, yva = build_user_sequences(val_df, numeric_cols, cat_cols, with_target=True)
    te_ids, Xte_num, Xte_cat, Xte_agg, _ = build_user_sequences(test_df, numeric_cols, cat_cols, with_target=False)

    # Standardize numeric seq features (fit on TRAIN only)
    mean, std = fit_standardizer(Xtr_num)
    Xtr_num = apply_standardizer(Xtr_num, mean, std)
    Xva_num = apply_standardizer(Xva_num, mean, std)
    Xte_num = apply_standardizer(Xte_num, mean, std)

    # Datasets / Loaders
    batch_size = 256
    train_ds = UserSeqDataset(Xtr_num, Xtr_cat, Xtr_agg, ytr)
    val_ds   = UserSeqDataset(Xva_num, Xva_cat, Xva_agg, yva)
    test_ds  = UserSeqDataset(Xte_num, Xte_cat, Xte_agg, None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # Categorical cardinalities for embeddings
    cat_cardinalities = []
    cat_emb_dims = []
    for c in cat_cols:
        # cardinality from maps (UNK=0 handled)
        card = max(cat_maps[c].values()) if len(cat_maps[c]) else 0
        cat_cardinalities.append(card)
        # heuristic embedding size
        ed = int(min(32, max(4, round(np.sqrt(card + 1)))))
        cat_emb_dims.append(ed)

    agg_dim = int(Xtr_agg.shape[1])
    num_dim = int(Xtr_num.shape[2])

    # Build model
    model = EulerResNetEncoder(
        num_dim=num_dim,
        cat_cardinalities=cat_cardinalities,
        cat_emb_dims=cat_emb_dims,
        agg_dim=agg_dim,
        hidden_dim=192,
        day_emb_dim=8,
        dropout=0.10,
        dt=1.0,
        max_days=7
    ).to(device)

    # Class weight for sign head: emphasize negatives (reduce FN for negative class)
    neg_rate = float((ytr < 0).mean())
    pos_weight = (1.0 - neg_rate) / max(neg_rate, 1e-6)
    # slightly amplify (business: false positive costly)
    bce_pos_weight = float(pos_weight * 1.25)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Training
    print(f"Users: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Numeric day-dim: {num_dim} | Categorical cols: {len(cat_cols)} | Agg dim: {agg_dim}")
    print(f"BCE pos_weight (neg-class): {bce_pos_weight:.3f}")

    best_val_score = -1e18
    best_state = None
    patience = 5
    bad = 0
    max_epochs = 30

    for epoch in range(1, max_epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            bce_pos_weight=bce_pos_weight,
            tau=0.20, lam_q=0.30, lam_bce=0.20,
            grad_clip=1.0, use_amp=True
        )

        pred_val = predict(model, val_loader, device)
        val_score, val_metrics = compute_score(yva, pred_val)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_score={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Conservative delta calibration on VAL (no test leakage)
    pred_val = predict(model, val_loader, device)
    best_delta, best_cal_score, best_cal_metrics = calibrate_delta_on_val(
        yva, pred_val,
        deltas=np.arange(-200.0, 201.0, 5.0),
        require_fpr_le=0.40
    )
    print(f"Best delta (VAL calibrated): {best_delta:.1f} | cal_val_score={best_cal_score:.4f} | details={best_cal_metrics}")

    # Predict TEST and score (TEST contains target in this environment)
    pred_test = predict(model, test_loader, device)
    pred_test = pred_test - best_delta

    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(pred_test, index=te_ids).reindex(y_test.index).values

    score, metrics = compute_score(y_test.values, pred_aligned)
    print(f"🎯 Final Score: {score:.4f}")

    result = {
        "method": "Method 18: Physics-inspired Euler/ResNet (Neural ODE) MTL",
        "device": str(device),
        "best_val_score": float(best_val_score),
        "best_delta": float(best_delta),
        "score": float(score),
        "metrics": metrics,
        "val_calibration_metrics": best_cal_metrics,
        "n_train_users": int(len(tr_ids)),
        "n_val_users": int(len(va_ids)),
        "n_test_users": int(len(te_ids)),
        "n_numeric_cols": int(len(numeric_cols)),
        "n_cat_cols": int(len(cat_cols)),
    }
    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/method_18_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
