import warnings
warnings.filterwarnings('ignore')

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

TRAIN_PATH = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"

TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"

NUMERICAL_COLS = [
    "DEPOSIT_AMOUNT", "REC_USD", "REC_USD_CUM", "REC_USD_D6", "CPI",
    "RANK1_PLAY_CNT_ALL", "PLAY_CNT_ALL", "ACTUAL_ENTRY_FEE_CASH",
    "ACTUAL_REWARD_CASH", "PLAY_CNT_CASH", "HIGHFEE_PLAY_CNT_CASH",
    "CASH_RATIO", "ACTIVE_DAYS_ALL_CUM", "PLAY_CNT_ALL_CUM", "SESSION_CNT_ALL"
]

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

DAY_COL = None
SELECTED_NUM_COLS = None
SELECTED_CAT_COLS = None

def _infer_day_col(df: pd.DataFrame):
    for c in ["TDATE_RN", "DAY", "DAY_RN", "DAY_NUM", "DAYS_SINCE_INSTALL", "RN"]:
        if c in df.columns:
            return c
    return None

def _pick_numeric_cols(train_df: pd.DataFrame, max_cols: int = 80):
    day_col = _infer_day_col(train_df)
    numeric = [
        c for c in train_df.columns
        if c not in [ID_COL, TARGET_COL, day_col]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    forced = [c for c in NUMERICAL_COLS if c in numeric]
    rest = [c for c in numeric if c not in forced]

    if len(rest) > 0:
        v = train_df[rest].astype(np.float32).var(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_rest = v.sort_values(ascending=False).head(max(0, max_cols - len(forced))).index.tolist()
    else:
        top_rest = []

    cols = forced + top_rest
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _pick_cat_cols(train_df: pd.DataFrame):
    day_col = _infer_day_col(train_df)
    cat = [
        c for c in train_df.columns
        if c not in [ID_COL, TARGET_COL, day_col]
        and (train_df[c].dtype == "object" or str(train_df[c].dtype).startswith("category"))
    ]
    out = []
    for c in cat:
        if train_df[c].notna().mean() > 0.05:
            out.append(c)
    return out

def _safe_signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

def create_features(df, is_train=True):
    global DAY_COL, SELECTED_NUM_COLS, SELECTED_CAT_COLS

    if DAY_COL is None:
        DAY_COL = _infer_day_col(df)

    day_col = DAY_COL

    if SELECTED_NUM_COLS is None:
        num_all = [c for c in NUMERICAL_COLS if c in df.columns]
        if len(num_all) == 0:
            num_all = [
                c for c in df.columns
                if c not in [ID_COL, TARGET_COL, day_col]
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        SELECTED_NUM_COLS = num_all

    if SELECTED_CAT_COLS is None:
        SELECTED_CAT_COLS = _pick_cat_cols(df)

    num_cols = [c for c in SELECTED_NUM_COLS if c in df.columns]
    cat_cols = [c for c in SELECTED_CAT_COLS if c in df.columns]

    users = df[ID_COL].drop_duplicates().to_frame()

    if len(cat_cols) > 0:
        df_cat = df[[ID_COL, *(cat_cols + ([day_col] if day_col else []))]].copy()
        if day_col:
            df_cat = df_cat.sort_values([ID_COL, day_col])
        cat_agg = df_cat.groupby(ID_COL)[cat_cols].agg(["first", "last", "nunique"])
        cat_agg.columns = [f"{c0}_{c1}" for (c0, c1) in cat_agg.columns]
        cat_agg = cat_agg.reset_index()
        users = users.merge(cat_agg, on=ID_COL, how="left")

    if len(num_cols) > 0:
        df_num = df[[ID_COL, *(num_cols + ([day_col] if day_col else []))]].copy()
        if day_col:
            df_num = df_num.sort_values([ID_COL, day_col])

        base = df_num.groupby(ID_COL)[num_cols].agg(["sum", "mean", "max", "min", "std", "last"])
        base.columns = [f"{c0}_{c1}" for (c0, c1) in base.columns]
        base = base.reset_index()
        users = users.merge(base, on=ID_COL, how="left")

        if day_col and df_num[day_col].nunique() >= 2:
            df_num_dedup = df_num.drop_duplicates(subset=[ID_COL, day_col], keep='last')
            
            seq = df_num_dedup.set_index([ID_COL, day_col])[num_cols].unstack(day_col)

            days = sorted([d for d in seq.columns.levels[1].tolist() if pd.notna(d)])
            if all(d in days for d in [1, 2, 3, 4, 5, 6, 7]):
                days = [1, 2, 3, 4, 5, 6, 7]

            seq = seq.reindex(columns=pd.MultiIndex.from_product([num_cols, days]), copy=False)
            seq = seq.fillna(0.0)

            t = np.asarray(days, dtype=np.float32)
            t_center = t - t.mean()
            denom = float((t_center ** 2).sum()) if len(t_center) > 0 else 1.0
            eps = 1e-3

            feat_blocks = []
            idx = seq.index

            for col in num_cols:
                x = seq[col].to_numpy(dtype=np.float32, copy=False)
                n_days = x.shape[1]
                if n_days == 0:
                    continue

                x1 = x[:, 0]
                x_last = x[:, -1]

                first_k = min(3, n_days)
                last_k = min(3, n_days)

                mean_first = x[:, :first_k].mean(axis=1)
                mean_last = x[:, -last_k:].mean(axis=1)

                velocity = (mean_last - mean_first) / 4.0

                if n_days >= 6:
                    vel1 = (x[:, 2:4].mean(axis=1) - x[:, :2].mean(axis=1)) / 2.0
                    vel2 = (x[:, -2:].mean(axis=1) - x[:, -4:-2].mean(axis=1)) / 2.0
                    accel = vel2 - vel1
                elif n_days >= 4:
                    vel1 = (x[:, 1:3].mean(axis=1) - x[:, :1].mean(axis=1))
                    vel2 = (x[:, -1:].mean(axis=1) - x[:, -3:-2].mean(axis=1))
                    accel = vel2 - vel1
                else:
                    accel = np.zeros_like(velocity)

                if n_days >= 2:
                    dx = np.diff(x, axis=1)
                    vol = dx.std(axis=1) / (np.abs(x.mean(axis=1)) + eps)
                else:
                    vol = np.zeros_like(velocity)

                recency_ratio = x_last / (x1 + np.sign(x1) * eps + eps)

                peak_day_pos = np.argmax(x, axis=1)
                peak_last2 = (peak_day_pos >= max(0, n_days - 2)).astype(np.float32)

                last3_sum = x[:, -min(3, n_days):].sum(axis=1)
                nonzero_days = (x != 0).sum(axis=1).astype(np.float32)

                xm = x.mean(axis=1, keepdims=True)
                slope = ((x - xm) * t_center.reshape(1, -1)).sum(axis=1) / denom

                delta_1_last = x_last - x1

                is_moneyish = any(k in col.upper() for k in ["USD", "AMOUNT", "CASH", "FEE", "REWARD", "DEPOSIT", "CPI"])
                if is_moneyish:
                    mean_first_log = _safe_signed_log1p(mean_first)
                    mean_last_log = _safe_signed_log1p(mean_last)
                    last_log = _safe_signed_log1p(x_last)
                else:
                    mean_first_log = None

                block = pd.DataFrame({
                    ID_COL: idx,
                    f"{col}_day1": x1,
                    f"{col}_day_last": x_last,
                    f"{col}_mean_d1_3": mean_first,
                    f"{col}_mean_dlast3": mean_last,
                    f"{col}_velocity": velocity,
                    f"{col}_accel": accel,
                    f"{col}_volatility": vol,
                    f"{col}_recency_ratio": recency_ratio,
                    f"{col}_peak_last2": peak_last2,
                    f"{col}_last3_sum": last3_sum,
                    f"{col}_nonzero_days": nonzero_days,
                    f"{col}_slope": slope,
                    f"{col}_delta_1_last": delta_1_last,
                }).reset_index(drop=True)

                if mean_first_log is not None:
                    block[f"{col}_mean_d1_3_slog"] = mean_first_log
                    block[f"{col}_mean_dlast3_slog"] = mean_last_log
                    block[f"{col}_day_last_slog"] = last_log

                feat_blocks.append(block)

            if len(feat_blocks) > 0:
                temporal = feat_blocks[0]
                for b in feat_blocks[1:]:
                    temporal = temporal.merge(b, on=ID_COL, how="left")
                users = users.merge(temporal, on=ID_COL, how="left")

    if is_train and TARGET_COL in df.columns:
        y = df.groupby(ID_COL)[TARGET_COL].first().reset_index()
        users = users.merge(y, on=ID_COL, how="left")

    users = users.fillna(0)
    return users

def fit_apply_cat_encodings(train_u: pd.DataFrame, val_u: pd.DataFrame, test_u: pd.DataFrame):
    cat_cols = [
        c for c in train_u.columns
        if c not in [ID_COL, TARGET_COL]
        and (train_u[c].dtype == "object" or str(train_u[c].dtype).startswith("category"))
    ]
    if len(cat_cols) == 0:
        return train_u, val_u, test_u

    out_train, out_val, out_test = train_u.copy(), val_u.copy(), test_u.copy()

    y_train = out_train[TARGET_COL].values
    prior_neg = float((y_train < 0).mean())
    m = 30.0

    for c in cat_cols:
        vc = out_train[c].fillna("__NA__").astype(str).value_counts(dropna=False)
        freq_map = (vc / len(out_train)).to_dict()

        tmp = pd.DataFrame({c: out_train[c].fillna("__NA__").astype(str), "neg": (out_train[TARGET_COL] < 0).astype(np.float32)})
        grp = tmp.groupby(c)["neg"].agg(["sum", "count"])
        neg_rate = (grp["sum"] + m * prior_neg) / (grp["count"] + m)
        neg_map = neg_rate.to_dict()

        for df_ in (out_train, out_val, out_test):
            key = df_[c].fillna("__NA__").astype(str)
            df_[f"{c}__freq"] = key.map(freq_map).fillna(0.0).astype(np.float32)
            df_[f"{c}__neg_rate"] = key.map(neg_map).fillna(prior_neg).astype(np.float32)

        out_train.drop(columns=[c], inplace=True)
        out_val.drop(columns=[c], inplace=True)
        out_test.drop(columns=[c], inplace=True)

    return out_train, out_val, out_test

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

def train_value_network(X_train, y_train, X_val, y_val, epochs=80, batch_size=512):
    print("🚀 Training Value Network (RL-style)...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_s).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    main_net = ValueNetwork(X_train.shape[1]).to(device)

    optimizer = AdamW(main_net.parameters(), lr=0.0012, weight_decay=0.01)
    best_val_loss, best_state = float("inf"), None
    patience, bad = 12, 0

    for epoch in range(epochs):
        main_net.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = main_net(X_batch)

            w = torch.ones_like(y_batch)

            fp = (pred > 0) & (y_batch < 0)
            fn = (pred < 0) & (y_batch > 0)

            w[fp] = 2.5 + 0.02 * torch.abs(y_batch[fp])
            w[fn] = 1.5 + 0.01 * y_batch[fn]
            w = torch.clamp(w, 0.1, 10.0)

            loss = (w * (pred - y_batch) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        main_net.eval()
        with torch.no_grad():
            val_pred = main_net(X_val_t)
            val_loss = torch.mean((val_pred - y_val_t) ** 2).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in main_net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Train Loss={epoch_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")

        if bad >= patience:
            break

    if best_state is not None:
        main_net.load_state_dict(best_state)
    return main_net, scaler

def awmse_lgb_obj(preds: np.ndarray, dataset: lgb.Dataset):
    y = dataset.get_label()
    w = np.ones_like(y, dtype=np.float64)

    fp = (preds > 0) & (y < 0)
    fn = (preds < 0) & (y > 0)

    w[fp] = 2.5 + 0.02 * np.abs(y[fp])
    w[fn] = 1.5 + 0.01 * y[fn]
    w = np.clip(w, 0.1, 10.0)

    grad = 2.0 * w * (preds - y)
    hess = 2.0 * w
    return grad, hess

def rmse_feval(preds: np.ndarray, dataset: lgb.Dataset):
    y = dataset.get_label()
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    return "rmse", rmse, False

def compute_score_silent(y_true, y_pred):
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

    return float(base + pareto)

def main():
    print("=" * 60)
    print("Method 27 (Improved): Temporal Features + AWMSE LGB Stack + RL MLP")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    global DAY_COL, SELECTED_NUM_COLS, SELECTED_CAT_COLS
    DAY_COL = _infer_day_col(train_df)
    SELECTED_NUM_COLS = _pick_numeric_cols(train_df, max_cols=80)
    SELECTED_CAT_COLS = _pick_cat_cols(train_df)

    train_u = create_features(train_df, is_train=True)
    val_u   = create_features(val_df, is_train=True)
    test_u  = create_features(test_df, is_train=False)

    train_u, val_u, test_u = fit_apply_cat_encodings(train_u, val_u, test_u)

    feature_cols = [c for c in train_u.columns if c not in [ID_COL, TARGET_COL]]
    X_train = train_u[feature_cols].values.astype(np.float32)
    y_train = train_u[TARGET_COL].values.astype(np.float32)
    X_val   = val_u[feature_cols].values.astype(np.float32)
    y_val   = val_u[TARGET_COL].values.astype(np.float32)
    X_test  = test_u[feature_cols].values.astype(np.float32)
    test_ids = test_u[ID_COL].values

    rl_model, rl_scaler = train_value_network(X_train, y_train, X_val, y_val)

    rl_model.eval()
    with torch.no_grad():
        rl_val_pred = rl_model(torch.FloatTensor(rl_scaler.transform(X_val)).to(device)).detach().cpu().numpy()
        rl_test_pred = rl_model(torch.FloatTensor(rl_scaler.transform(X_test)).to(device)).detach().cpu().numpy()

    print("🚀 Training Random Forest...")
    rf_weights = np.ones_like(y_train, dtype=np.float32)
    rf_weights[y_train < 0] = 2.5 + 0.02 * np.abs(y_train[y_train < 0])
    rf_weights = np.clip(rf_weights, 0.1, 10.0)

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train, sample_weight=rf_weights)
    rf_val_pred = rf.predict(X_val).astype(np.float32)
    rf_test_pred = rf.predict(X_test).astype(np.float32)

    print("🚀 Training LightGBM (custom AWMSE objective)...")
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    lgb_params = dict(
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=1.0,
        max_depth=-1,
        verbosity=-1,
        seed=42,
    )

    lgb_aw = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        valid_names=["val"],
        fobj=awmse_lgb_obj,
        feval=rmse_feval,
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )
    aw_best_iter = int(lgb_aw.best_iteration or 1000)
    aw_val_pred = lgb_aw.predict(X_val, num_iteration=aw_best_iter).astype(np.float32)
    aw_test_pred = lgb_aw.predict(X_test, num_iteration=aw_best_iter).astype(np.float32)

    print("🚀 Training LightGBM Quantile (alpha=0.20)...")
    lgb_q = LGBMRegressor(
        objective="quantile",
        alpha=0.20,
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=96,
        min_child_samples=60,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_q.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    q_best_iter = int(getattr(lgb_q, "best_iteration_", 2000) or 2000)
    q_val_pred = lgb_q.predict(X_val, num_iteration=q_best_iter).astype(np.float32)
    q_test_pred = lgb_q.predict(X_test, num_iteration=q_best_iter).astype(np.float32)

    print("🚀 Training LightGBM Sign Classifier (P(LTV<0))...")
    y_train_neg = (y_train < 0).astype(int)
    y_val_neg = (y_val < 0).astype(int)

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(
        X_train, y_train_neg,
        eval_set=[(X_val, y_val_neg)],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    pneg_val = clf.predict_proba(X_val)[:, 1].astype(np.float32)
    pneg_test = clf.predict_proba(X_test)[:, 1].astype(np.float32)

    print("🧩 Training Ridge meta-learner on VAL (stacking holdout)...")
    meta_X_val = np.vstack([
        aw_val_pred,
        q_val_pred,
        rf_val_pred,
        rl_val_pred,
        pneg_val,
        (aw_val_pred - q_val_pred),
    ]).T.astype(np.float32)

    meta_X_test = np.vstack([
        aw_test_pred,
        q_test_pred,
        rf_test_pred,
        rl_test_pred,
        pneg_test,
        (aw_test_pred - q_test_pred),
    ]).T.astype(np.float32)

    meta_w = np.ones_like(y_val, dtype=np.float32)
    meta_w[y_val < 0] = 1.67
    meta = Ridge(alpha=1.0, random_state=42)
    meta.fit(meta_X_val, y_val, sample_weight=meta_w)

    meta_val_pred = meta.predict(meta_X_val).astype(np.float32)
    meta_test_pred = meta.predict(meta_X_test).astype(np.float32)

    print("🧪 Calibrating conservative shift on VAL...")
    deltas = np.linspace(-120, 160, 57).astype(np.float32)

    best_s = -1e9
    best_delta = 0.0
    best_bias = 0.0

    n_val = float(len(y_val))
    sum_y_val = float(np.sum(y_val))

    for d in deltas:
        pred = meta_val_pred - d
        bias = (sum_y_val - float(np.sum(pred))) / max(n_val, 1.0)
        pred2 = pred + bias
        s = compute_score_silent(y_val, pred2)
        if s > best_s:
            best_s = s
            best_delta = float(d)
            best_bias = float(bias)

    print(f"   Best VAL calibration: delta={best_delta:.4f}, bias={best_bias:.4f}, val_score={best_s:.6f}")

    final_pred = (meta_test_pred - best_delta + best_bias).astype(np.float32)

    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(final_pred, index=test_ids).reindex(y_test.index).values.astype(np.float32)

    print("\n" + "=" * 60)
    score, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 60)
    print(f"\n🎯 Final Score: {score:.4f}")

    result = {
        "method": "Method 27 Improved: Temporal + AWMSE LGB Stack + RL MLP",
        "score": float(score),
        "metrics": metrics,
        "calibration": {"delta": best_delta, "bias": best_bias},
        "aw_best_iter": aw_best_iter,
        "q_best_iter": q_best_iter,
        "n_features": int(len(feature_cols)),
    }
    out_path = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/method_27_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"score = {score}")
    return score

if __name__ == "__main__":
    score = main()
    print(f"score = {score}")
