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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    torch.manual_seed(seed)

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

def masked_mean_np(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cnt = np.clip(mask.sum(axis=1), 1.0, None)  # (U,1)
    return (X * mask).sum(axis=1) / cnt

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
    # keep padded positions at 0 to stabilize
    Xn = Xn * mask
    return Xn.astype(np.float32), mu.astype(np.float32), std.astype(np.float32)


# ============================================================
# Deep Learning (Computational Physics-inspired) model
#   - Finite-difference "velocity" features (x_t - x_{t-1})
#   - Residual temporal blocks (ResNet/ODE discretization viewpoint)
#   - Multi-task heads: sign (neg risk), mean regression, q20 (conservative)
# ============================================================
class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.10):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

        # "small step" residual scaling (stability like explicit time stepping)
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x_btC: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.ln1(x_btC)
        h = h.transpose(1, 2)  # (B, C, T)
        h = self.conv1(h)
        h = F.gelu(h)
        h = self.drop(h)

        h = h.transpose(1, 2)  # (B, T, C)
        h = self.ln2(h)
        h = h.transpose(1, 2)  # (B, C, T)
        h = self.conv2(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = h.transpose(1, 2)  # (B, T, C)

        return x_btC + self.res_scale * h

class PhysicsInspiredMTLNet(nn.Module):
    def __init__(
        self,
        num_in: int,
        static_in: int,
        cat_cardinalities: list[int],
        d_model: int = 160,
        depth: int = 6,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.num_in = num_in
        self.static_in = static_in

        # Categorical entity embeddings
        self.cat_embs = nn.ModuleList()
        emb_out = 0
        for card in cat_cardinalities:
            dim = int(min(32, max(4, round(1.6 * math.sqrt(max(card, 2))))))
            self.cat_embs.append(nn.Embedding(card, dim))
            emb_out += dim
        self.cat_out = emb_out

        # Sequence preprocessing: add finite difference ("velocity") features
        self.seq_proj = nn.Linear(num_in * 2, d_model)

        self.blocks = nn.ModuleList([
            ResidualTCNBlock(d_model, kernel_size=3, dilation=2 ** (i % 3), dropout=dropout)
            for i in range(depth)
        ])

        # Static branch
        self.static_ln = nn.LayerNorm(static_in) if static_in > 0 else None
        self.static_mlp = nn.Sequential(
            nn.Linear(static_in + emb_out, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        ) if (static_in + emb_out) > 0 else None

        # Fusion + heads
        fused_dim = (d_model * 3) + (d_model if self.static_mlp is not None else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head_mean = nn.Linear(d_model, 1)
        self.head_q20 = nn.Linear(d_model, 1)
        self.head_neg = nn.Linear(d_model, 1)  # logit for P(y<0)

        # Init (He/Xavier-like)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_seq: torch.Tensor,      # (B,T,D)
        mask: torch.Tensor,       # (B,T,1)
        x_static: torch.Tensor,   # (B,S)
        x_cat: torch.Tensor,      # (B,Ccat)
    ):
        # finite differences (velocity)
        dx = x_seq[:, 1:, :] - x_seq[:, :-1, :]
        dx = torch.cat([torch.zeros_like(dx[:, :1, :]), dx], dim=1)
        seq = torch.cat([x_seq, dx], dim=-1)  # (B,T,2D)
        h = self.seq_proj(seq)
        for blk in self.blocks:
            h = blk(h)  # (B,T,C)

        # masked pooling (physics: integral-like averages)
        m = mask
        cnt = torch.clamp(m.sum(dim=1), min=1.0)  # (B,1)
        h_mean = (h * m).sum(dim=1) / cnt  # (B,C)
        h_max = torch.where(m.bool(), h, torch.full_like(h, -1e9)).max(dim=1).values  # (B,C)
        h_last = h[:, -1, :]  # fixed horizon (day 7)

        # static cats
        cat_vecs = []
        for j, emb in enumerate(self.cat_embs):
            cat_vecs.append(emb(x_cat[:, j]))
        cat_vec = torch.cat(cat_vecs, dim=1) if cat_vecs else None

        if self.static_mlp is not None:
            if self.static_ln is not None:
                x_static = self.static_ln(x_static)
            if cat_vec is not None:
                s_in = torch.cat([x_static, cat_vec], dim=1)
            else:
                s_in = x_static
            s = self.static_mlp(s_in)
            fused = torch.cat([h_mean, h_max, h_last, s], dim=1)
        else:
            fused = torch.cat([h_mean, h_max, h_last], dim=1)

        z = self.fuse(fused)
        y_mean = self.head_mean(z).squeeze(1)
        y_q20  = self.head_q20(z).squeeze(1)
        neg_logit = self.head_neg(z).squeeze(1)
        return y_mean, y_q20, neg_logit


# ============================================================
# Losses (asymmetric + pinball)
# ============================================================
def awmse_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Asymmetric Weighted MSE consistent with the task guidance.
    """
    fp = (y_pred > 0) & (y_true < 0)
    fn = (y_pred < 0) & (y_true > 0)

    w = torch.ones_like(y_true, dtype=torch.float32)
    if fp.any():
        w = torch.where(fp, (2.5 + 0.02 * torch.abs(y_true)).to(torch.float32), w)
    if fn.any():
        w = torch.where(fn, (1.5 + 0.01 * torch.clamp(y_true, min=0.0)).to(torch.float32), w)

    return torch.mean(w * (y_pred - y_true) ** 2)

def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: float = 0.20) -> torch.Tensor:
    diff = y_true - y_pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))


# ============================================================
# Dataset / Prediction helpers
# ============================================================
class UserSeqDataset(Dataset):
    def __init__(self, X_seq, mask, X_static, X_cat, y=None):
        self.X_seq = X_seq.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.X_static = X_static.astype(np.float32) if X_static is not None else np.zeros((len(X_seq), 0), np.float32)
        self.X_cat = X_cat.astype(np.int64) if X_cat is not None else np.zeros((len(X_seq), 0), np.int64)
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X_seq[i])
        m = torch.from_numpy(self.mask[i])
        s = torch.from_numpy(self.X_static[i])
        c = torch.from_numpy(self.X_cat[i])
        if self.y is None:
            return x, m, s, c
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, m, s, c, y

@torch.no_grad()
def predict_deep(model, loader):
    model.eval()
    mean_list, q20_list, pneg_list = [], [], []
    for batch in loader:
        if len(batch) == 5:
            x, m, s, c, _ = batch
        else:
            x, m, s, c = batch
        x = x.to(device)
        m = m.to(device)
        s = s.to(device)
        c = c.to(device)
        y_mean, y_q20, neg_logit = model(x, m, s, c)
        pneg = torch.sigmoid(neg_logit)
        mean_list.append(y_mean.detach().cpu().numpy().astype(np.float32))
        q20_list.append(y_q20.detach().cpu().numpy().astype(np.float32))
        pneg_list.append(pneg.detach().cpu().numpy().astype(np.float32))
    return (
        np.concatenate(mean_list, axis=0),
        np.concatenate(q20_list, axis=0),
        np.concatenate(pneg_list, axis=0),
    )


# ============================================================
# Conservative policy (post-hoc calibration)
# ============================================================
def apply_policy_deep(pred_mean, pred_q20, pneg, w_q=0.20, use_min=True, thr_gate=0.65, snap_eps=1.0, risk_scale=0.25):
    pred_mean = pred_mean.astype(np.float32)
    pred_q20 = pred_q20.astype(np.float32)
    pneg = pneg.astype(np.float32)

    y = ((1.0 - w_q) * pred_mean + w_q * pred_q20).astype(np.float32)
    if use_min:
        y = np.minimum(y, pred_q20)

    # risk shaping: damp positives above q20 when pneg high
    if risk_scale > 0:
        gap = np.maximum(y - pred_q20, 0.0).astype(np.float32)
        y = y - (np.float32(risk_scale) * pneg * gap)

    # hard gate on negative-risk
    if thr_gate is not None:
        y = np.where(pneg > np.float32(thr_gate), np.minimum(y, 0.0), y).astype(np.float32)

    # snap tiny values to zero for zero inflation
    if snap_eps is not None and snap_eps > 0:
        y = np.where(np.abs(y) < np.float32(snap_eps), 0.0, y).astype(np.float32)

    return y.astype(np.float32)

def find_delta_for_fpr(y_true, y_pred, fpr_target=0.40, grid=None):
    """
    Smallest downward shift delta such that FPR <= fpr_target.
    """
    if grid is None:
        # coarse-to-fine grid (downshift)
        grid = np.concatenate([
            np.linspace(0, 30, 16),
            np.linspace(35, 120, 18),
            np.linspace(130, 250, 13),
        ]).astype(np.float32)

    best_delta = float(grid[-1])
    for d in grid:
        fpr = fpr_predicted_positive(y_true, y_pred - d)
        if fpr <= fpr_target:
            best_delta = float(d)
            break
    return best_delta

def tune_on_val_deep(yva, pred_mean_va, pred_q20_va, pneg_va, anchor_mean: float):
    w_q_grid = [0.00, 0.10, 0.20, 0.30, 0.40]
    use_min_grid = [True, False]
    thr_gate_grid = [0.55, 0.65, 0.75]
    snap_eps_grid = [0.0, 0.5, 1.0, 2.0]
    risk_scale_grid = [0.0, 0.20, 0.35]

    best = (-1e9, None)
    for w_q in w_q_grid:
        for use_min in use_min_grid:
            for thr_gate in thr_gate_grid:
                for snap_eps in snap_eps_grid:
                    for risk_scale in risk_scale_grid:
                        y_raw = apply_policy_deep(
                            pred_mean_va, pred_q20_va, pneg_va,
                            w_q=w_q, use_min=use_min, thr_gate=thr_gate,
                            snap_eps=snap_eps, risk_scale=risk_scale
                        )

                        a, b = affine_fit(yva, y_raw)
                        y_cal = (a * y_raw + b).astype(np.float32)
                        y_cal = mean_correct(y_cal, anchor_mean=anchor_mean)

                        # enforce conservative FPR constraint via dynamic shift (method-like calibration)
                        delta = find_delta_for_fpr(yva, y_cal, fpr_target=0.40)
                        y_final = (y_cal - np.float32(delta)).astype(np.float32)

                        with contextlib.redirect_stdout(io.StringIO()):
                            s = compute_pareto_multi_objective(yva, y_final)

                        # light penalty if still violates (shouldn't) or if over-shifted too much
                        fpr = fpr_predicted_positive(yva, y_final)
                        if fpr > 0.40:
                            s -= 0.25  # strong penalty for violation

                        if s > best[0]:
                            best = (s, dict(
                                w_q=float(w_q),
                                use_min=bool(use_min),
                                thr_gate=float(thr_gate),
                                snap_eps=float(snap_eps),
                                risk_scale=float(risk_scale),
                                a=float(a),
                                b=float(b),
                                delta=float(delta),
                            ))
    return best


# ============================================================
# Main pipeline (Deep Learning + physics-inspired discretization features)
# ============================================================
def main():
    set_seed(42)

    print("=" * 70)
    print("Physics-Inspired Deep MTL: finite-difference features + residual TCN + sign/mean/q20 heads")
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

    # Categories (entity embeddings + TE + counts)
    cat_maps = build_cat_maps(train_df, cat_cols)
    Xc_tr, cat_names = encode_user_cats(train_df, tr_users, cat_cols, cat_maps)
    Xc_va, _ = encode_user_cats(val_df,   va_users, cat_cols, cat_maps)
    Xc_te, _ = encode_user_cats(test_df,  te_users, cat_cols, cat_maps)

    # Sequence numeric (7-day)
    Xn_tr_3d, m_tr = build_user_numeric_tensor(train_df, tr_users, num_cols, n_days=N_DAYS)
    Xn_va_3d, m_va = build_user_numeric_tensor(val_df,   va_users, num_cols, n_days=N_DAYS)
    Xn_te_3d, m_te = build_user_numeric_tensor(test_df,  te_users, num_cols, n_days=N_DAYS)

    # Standardize using train (masked)
    Xn_tr_3d, mu_num, std_num = standardize_3d(Xn_tr_3d, m_tr, Xn_tr_3d, m_tr)
    Xn_va_3d = ((Xn_va_3d - mu_num[None, None, :]) / std_num[None, None, :]).astype(np.float32) * m_va
    Xn_te_3d = ((Xn_te_3d - mu_num[None, None, :]) / std_num[None, None, :]).astype(np.float32) * m_te

    # Static numeric: mean + last + day_count
    mean_tr = masked_mean_np(Xn_tr_3d, m_tr).astype(np.float32)
    mean_va = masked_mean_np(Xn_va_3d, m_va).astype(np.float32)
    mean_te = masked_mean_np(Xn_te_3d, m_te).astype(np.float32)

    last_tr = Xn_tr_3d[:, -1, :].astype(np.float32)
    last_va = Xn_va_3d[:, -1, :].astype(np.float32)
    last_te = Xn_te_3d[:, -1, :].astype(np.float32)

    daycnt_tr = m_tr.sum(axis=1).reshape(-1, 1).astype(np.float32)
    daycnt_va = m_va.sum(axis=1).reshape(-1, 1).astype(np.float32)
    daycnt_te = m_te.sum(axis=1).reshape(-1, 1).astype(np.float32)

    te_tr, te_va, te_te, te_names = build_target_encoding_features_v2(
        Xc_tr, ytr, Xc_va, Xc_te, cat_names, seed=42
    )
    (cnt_tr, cnt_names), (cnt_va, _), (cnt_te, _) = build_cat_count_features(train_df, [train_df, val_df, test_df], cat_cols)

    Xs_tr = np.concatenate([mean_tr, last_tr, daycnt_tr, te_tr, cnt_tr], axis=1).astype(np.float32)
    Xs_va = np.concatenate([mean_va, last_va, daycnt_va, te_va, cnt_va], axis=1).astype(np.float32)
    Xs_te = np.concatenate([mean_te, last_te, daycnt_te, te_te, cnt_te], axis=1).astype(np.float32)

    # Clip range (train-derived)
    lo, hi = np.quantile(ytr, [0.001, 0.999])
    lo, hi = float(lo), float(hi)

    # Model setup
    cat_cardinalities = []
    for j in range(Xc_tr.shape[1]):
        card = int(max(Xc_tr[:, j].max(), Xc_va[:, j].max(), Xc_te[:, j].max()) + 1)
        cat_cardinalities.append(max(card, 2))

    model = PhysicsInspiredMTLNet(
        num_in=Xn_tr_3d.shape[2],
        static_in=Xs_tr.shape[1],
        cat_cardinalities=cat_cardinalities,
        d_model=160,
        depth=6,
        dropout=0.10,
    ).to(device)

    # Dataloaders
    bs = 512
    tr_ds = UserSeqDataset(Xn_tr_3d, m_tr, Xs_tr, Xc_tr, y=ytr)
    va_ds = UserSeqDataset(Xn_va_3d, m_va, Xs_va, Xc_va, y=yva)
    te_ds = UserSeqDataset(Xn_te_3d, m_te, Xs_te, Xc_te, y=None)

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)
    te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)

    # Loss weights / optimizer (AdamW: lecture-note standard)
    ytr_neg = (ytr < 0).astype(np.float32)
    pos = float(ytr_neg.sum())
    neg = float(len(ytr_neg) - pos)
    # label 1 = negative class
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    best_state = None
    best_val = -1e9
    patience = 8
    bad = 0

    print("\n[Training] Deep MTL model...")
    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        nobs = 0

        for x, m, s, c, y in tr_loader:
            x = x.to(device)
            m = m.to(device)
            s = s.to(device)
            c = c.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            pred_mean, pred_q20, neg_logit = model(x, m, s, c)

            loss_reg = awmse_torch(pred_mean, y)
            loss_q = pinball_loss(pred_q20, y, tau=0.20)
            y_neg = (y < 0).to(torch.float32)
            loss_cls = bce(neg_logit, y_neg)

            loss = (1.00 * loss_reg) + (0.35 * loss_q) + (0.35 * loss_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            total_loss += float(loss.detach().cpu().item()) * len(y)
            nobs += len(y)

        scheduler.step()
        train_loss = total_loss / max(nobs, 1)

        # fast validation scoring with a fixed conservative policy (for early stop)
        pred_mean_va, pred_q20_va, pneg_va = predict_deep(model, va_loader)
        pred_mean_va = np.clip(pred_mean_va, lo, hi).astype(np.float32)
        pred_q20_va = np.clip(pred_q20_va, lo, hi).astype(np.float32)

        y_raw_va = apply_policy_deep(
            pred_mean_va, pred_q20_va, pneg_va,
            w_q=0.20, use_min=True, thr_gate=0.65, snap_eps=1.0, risk_scale=0.25
        )
        a, b = affine_fit(yva, y_raw_va)
        y_cal_va = mean_correct((a * y_raw_va + b).astype(np.float32), anchor_mean=anchor_mean)
        delta = find_delta_for_fpr(yva, y_cal_va, fpr_target=0.40)
        y_final_va = (y_cal_va - np.float32(delta)).astype(np.float32)

        with contextlib.redirect_stdout(io.StringIO()):
            s_val = compute_pareto_multi_objective(yva, y_final_va)

        if s_val > best_val:
            best_val = float(s_val)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 2 == 0 or epoch == 1:
            fpr = fpr_predicted_positive(yva, y_final_va)
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | val_score={s_val:.6f} | val_fpr={fpr:.3f}")

        if bad >= patience:
            print(f"Early stopping at epoch {epoch} (best_val={best_val:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n[Tuning] Searching conservative policy on VAL...")
    pred_mean_va, pred_q20_va, pneg_va = predict_deep(model, va_loader)
    pred_mean_va = np.clip(pred_mean_va, lo, hi).astype(np.float32)
    pred_q20_va = np.clip(pred_q20_va, lo, hi).astype(np.float32)

    best_val_score, best_params = tune_on_val_deep(
        yva=yva,
        pred_mean_va=pred_mean_va,
        pred_q20_va=pred_q20_va,
        pneg_va=pneg_va,
        anchor_mean=anchor_mean
    )
    print(f"Best VAL policy score = {best_val_score:.6f}")
    print("Best params:", best_params)

    print("\n[Inference] Predicting TEST...")
    pred_mean_te, pred_q20_te, pneg_te = predict_deep(model, te_loader)
    pred_mean_te = np.clip(pred_mean_te, lo, hi).astype(np.float32)
    pred_q20_te = np.clip(pred_q20_te, lo, hi).astype(np.float32)

    y_raw_te = apply_policy_deep(
        pred_mean_te, pred_q20_te, pneg_te,
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
    pred_aligned = pd.Series(final_pred, index=te_users).reindex(y_test.index).values

    print("\n" + "=" * 70)
    score_val, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 70)
    print(f"\n🎯 Final Score: {score_val:.6f}")

    out_path = "/home/jupyter/AviaAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/physics_inspired_deep_mtl_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "method": "Physics-Inspired Deep MTL: finite-difference + residual TCN + sign/mean/q20 heads",
                "score": float(score_val),
                "metrics": metrics,
                "best_val_policy_score": float(best_val_score),
                "policy": best_params,
                "num_cols": int(len(num_cols)),
                "cat_cols": int(len(cat_cols)),
                "seq_days": int(N_DAYS),
                "static_features": int(Xs_tr.shape[1]),
                "anchor_mean": float(anchor_mean),
                "clip_lo": float(lo),
                "clip_hi": float(hi),
            },
            f,
            indent=2,
        )

    return score_val, pred_aligned, y_test.values


if __name__ == "__main__":
    final_score, test_predictions, y_test = main()
    print(f"score = {final_score}")

score = compute_pareto_multi_objective(y_test, test_predictions)
print(f"score = {score}")