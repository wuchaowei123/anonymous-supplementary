import warnings
warnings.filterwarnings('ignore')

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split, StratifiedKFold

import lightgbm as lgb

TRAIN_PATH = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/train.csv"
VAL_PATH   = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/val.csv"
TEST_PATH  = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/test.csv"
TARGET_COL, ID_COL = "REC_USD_D60", "DEVICE_ID"

NUMERICAL_COLS = ['DEPOSIT_AMOUNT', 'REC_USD', 'REC_USD_CUM', 'REC_USD_D6', 'CPI',
    'RANK1_PLAY_CNT_ALL', 'PLAY_CNT_ALL', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_REWARD_CASH', 'PLAY_CNT_CASH', 'HIGHFEE_PLAY_CNT_CASH',
    'CASH_RATIO', 'ACTIVE_DAYS_ALL_CUM', 'PLAY_CNT_ALL_CUM']

TCN_TEMPORAL_FEATURES = ['DEPOSIT_AMOUNT', 'REC_USD', 'PLAY_CNT_CASH', 'ACTUAL_ENTRY_FEE_CASH',
    'ACTUAL_REWARD_CASH', 'PLAY_CNT_ALL', 'SESSION_CNT_ALL', 'RANK1_PLAY_CNT_ALL',
    'AVG_SCORE_ALL', 'AD_PLAY_CNT_ALL', 'CLEAR_PLAY_CNT_ALL', 'HIGHFEE_PLAY_CNT_CASH']


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
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
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
    return final, {'gini': float(gini), 'error_rate': float(error_rate), 'spearman': float(spearman), 'rmse': float(rmse)}

def compute_pareto_multi_objective(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    s, _ = compute_score(y_true, y_pred)
    return s

def compute_score_silent(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
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

    final = float(base + pareto)
    return final, {'gini': float(gini), 'error_rate': float(error_rate), 'spearman': float(spearman), 'rmse': float(rmse)}


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_day_column(df: pd.DataFrame) -> pd.DataFrame:
    if "TDATE_RN" not in df.columns:
        df = df.copy()
        df["TDATE_RN"] = df.groupby(ID_COL).cumcount() + 1
    return df

def infer_feature_columns(train_df: pd.DataFrame):
    cat_cols = []
    for c in train_df.columns:
        if c in [ID_COL, TARGET_COL]:
            continue
        if str(train_df[c].dtype) in ("object", "category"):
            cat_cols.append(c)

    numeric_cols = []
    for c in train_df.columns:
        if c in [ID_COL, TARGET_COL, "TDATE_RN"]:
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]):
            numeric_cols.append(c)

    return sorted(numeric_cols), sorted(cat_cols)

def _safe_float32(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = df[c].astype(np.float32, copy=False)
    return df

def _make_user_level_base(df: pd.DataFrame, numeric_cols, cat_cols, with_target: bool, max_days: int = 7):
    df = ensure_day_column(df)
    df = df.sort_values([ID_COL, "TDATE_RN"])

    user_ids = df[ID_COL].drop_duplicates().to_numpy()
    n_users = len(user_ids)

    y = None
    if with_target:
        y = df.groupby(ID_COL, sort=False)[TARGET_COL].first().reindex(user_ids).to_numpy(np.float32)

    g = df.groupby(ID_COL, sort=False)

    agg_stats = ["sum", "mean", "std", "min", "max", "last", "first"]
    num_agg = g[numeric_cols].agg(agg_stats)
    num_agg.columns = [f"{c}__{s}" for (c, s) in num_agg.columns]
    num_agg = num_agg.reindex(user_ids).fillna(0.0)

    last_cols  = [f"{c}__last" for c in numeric_cols]
    first_cols = [f"{c}__first" for c in numeric_cols]
    num_diff = (num_agg[last_cols].to_numpy(np.float32) - num_agg[first_cols].to_numpy(np.float32))
    X_num_agg = num_agg.to_numpy(np.float32)
    X_num_agg = np.nan_to_num(X_num_agg, nan=0.0)
    X_num = np.concatenate([X_num_agg, num_diff], axis=1).astype(np.float32)

    cat_last = None
    cat_nuniq = None
    if cat_cols:
        cat_last = g[cat_cols].last().reindex(user_ids).fillna("UNK").astype(str)
        cat_nuniq = g[cat_cols].nunique().reindex(user_ids).fillna(0).astype(np.float32)
    else:
        cat_last = pd.DataFrame(index=user_ids)
        cat_nuniq = pd.DataFrame(index=user_ids)

    return user_ids, X_num, cat_last, cat_nuniq, y

def _make_seq_features(df: pd.DataFrame, user_ids: np.ndarray, seq_cols, max_days: int = 7):
    if not seq_cols:
        return np.zeros((len(user_ids), 0), dtype=np.float32)

    df = ensure_day_column(df)
    df = df.sort_values([ID_COL, "TDATE_RN"])
    
    df_grouped = df.groupby([ID_COL, "TDATE_RN"])[seq_cols].first().reset_index()
    base = df_grouped.set_index([ID_COL, "TDATE_RN"])[seq_cols]

    days = np.arange(1, max_days + 1, dtype=np.int64)
    mi = pd.MultiIndex.from_product([user_ids, days], names=[ID_COL, "TDATE_RN"])
    arr = base.reindex(mi).fillna(0.0).to_numpy(np.float32).reshape(len(user_ids), max_days, len(seq_cols))

    raw = arr
    dif = np.zeros_like(arr, dtype=np.float32)
    dif[:, 1:, :] = arr[:, 1:, :] - arr[:, :-1, :]
    slog = np.sign(arr) * np.log1p(np.abs(arr))

    x = np.arange(1, max_days + 1, dtype=np.float32)
    x = x - x.mean()
    denom = float((x ** 2).sum())
    y_center = arr - arr.mean(axis=1, keepdims=True)
    slope = (y_center * x[None, :, None]).sum(axis=1) / max(denom, 1e-6)

    raw_f  = raw.transpose(0, 2, 1).reshape(len(user_ids), -1)
    dif_f  = dif.transpose(0, 2, 1).reshape(len(user_ids), -1)
    slog_f = slog.transpose(0, 2, 1).reshape(len(user_ids), -1)

    X_seq = np.concatenate([raw_f, dif_f, slog_f, slope.astype(np.float32)], axis=1).astype(np.float32)
    return X_seq

def _stratify_3class(y: np.ndarray):
    y = np.asarray(y)
    return np.where(y < 0, 0, np.where(y == 0, 1, 2)).astype(np.int32)

def _build_oof_target_encoding(train_cat_last: pd.DataFrame, y_train: np.ndarray, cat_cols,
                               n_splits: int = 5, smooth: float = 30.0, seed: int = 42):
    N = len(y_train)
    if not cat_cols:
        return np.zeros((N, 0), np.float32), {}

    global_mean = float(np.mean(y_train))
    global_neg = float(np.mean(y_train < 0))
    global_zero = float(np.mean(y_train == 0))

    strat = _stratify_3class(y_train)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    te = np.zeros((N, 4 * len(cat_cols)), dtype=np.float32)

    def fit_maps(cat_df: pd.DataFrame, y: np.ndarray):
        maps = {}
        y_s = pd.Series(y, index=cat_df.index)
        neg_s = (y_s < 0).astype(np.float32)
        zero_s = (y_s == 0).astype(np.float32)

        for c in cat_cols:
            s = cat_df[c].fillna("UNK").astype(str)
            tmp = pd.DataFrame({"cat": s, "y": y_s.values, "neg": neg_s.values, "zero": zero_s.values})
            grp = tmp.groupby("cat", sort=False).agg(
                cnt=("y", "size"),
                mean=("y", "mean"),
                neg=("neg", "mean"),
                zero=("zero", "mean"),
            )
            cnt = grp["cnt"].astype(np.float32)
            mean = grp["mean"].astype(np.float32)
            neg = grp["neg"].astype(np.float32)
            zero = grp["zero"].astype(np.float32)

            enc_mean = (cnt * mean + smooth * global_mean) / (cnt + smooth)
            enc_neg = (cnt * neg + smooth * global_neg) / (cnt + smooth)
            enc_zero = (cnt * zero + smooth * global_zero) / (cnt + smooth)

            maps[c] = {
                "mean": enc_mean.to_dict(),
                "neg": enc_neg.to_dict(),
                "zero": enc_zero.to_dict(),
                "cnt": cnt.to_dict(),
            }
        return maps

    for tr_idx, va_idx in skf.split(np.zeros(N), strat):
        tr_cat = train_cat_last.iloc[tr_idx]
        va_cat = train_cat_last.iloc[va_idx]
        maps = fit_maps(tr_cat, y_train[tr_idx])

        cols_out = []
        for j, c in enumerate(cat_cols):
            s = va_cat[c].fillna("UNK").astype(str)
            m_mean = s.map(maps[c]["mean"]).fillna(global_mean).to_numpy(np.float32)
            m_neg = s.map(maps[c]["neg"]).fillna(global_neg).to_numpy(np.float32)
            m_zero = s.map(maps[c]["zero"]).fillna(global_zero).to_numpy(np.float32)
            m_cnt = np.log1p(s.map(maps[c]["cnt"]).fillna(0.0).to_numpy(np.float32))
            cols_out.append(m_mean)
            cols_out.append(m_neg)
            cols_out.append(m_zero)
            cols_out.append(m_cnt)

        te[va_idx, :] = np.stack(cols_out, axis=1).astype(np.float32)

    full_maps = fit_maps(train_cat_last, y_train)
    return te, {"global_mean": global_mean, "global_neg": global_neg, "global_zero": global_zero, "maps": full_maps}

def _apply_target_encoding(cat_last: pd.DataFrame, cat_cols, enc_bundle):
    if not cat_cols:
        return np.zeros((len(cat_last), 0), np.float32)

    global_mean = enc_bundle["global_mean"]
    global_neg = enc_bundle["global_neg"]
    global_zero = enc_bundle["global_zero"]
    maps = enc_bundle["maps"]

    cols_out = []
    for c in cat_cols:
        s = cat_last[c].fillna("UNK").astype(str)
        m_mean = s.map(maps[c]["mean"]).fillna(global_mean).to_numpy(np.float32)
        m_neg = s.map(maps[c]["neg"]).fillna(global_neg).to_numpy(np.float32)
        m_zero = s.map(maps[c]["zero"]).fillna(global_zero).to_numpy(np.float32)
        m_cnt = np.log1p(s.map(maps[c]["cnt"]).fillna(0.0).to_numpy(np.float32))
        cols_out.append(m_mean)
        cols_out.append(m_neg)
        cols_out.append(m_zero)
        cols_out.append(m_cnt)

    return np.stack(cols_out, axis=1).astype(np.float32)


def lgb_awmse_obj(preds, train_data):
    y = train_data.get_label().astype(np.float64)
    p = preds.astype(np.float64)
    w = np.ones_like(y, dtype=np.float64)

    fp = (p > 0) & (y < 0)
    fn = (p < 0) & (y > 0)

    if np.any(fp):
        w[fp] = 2.5 + 0.02 * np.abs(y[fp])
    if np.any(fn):
        w[fn] = 1.5 + 0.01 * y[fn]

    grad = 2.0 * w * (p - y)
    hess = 2.0 * w
    return grad, hess

def lgb_awmse_eval(preds, train_data):
    y = train_data.get_label().astype(np.float64)
    p = preds.astype(np.float64)
    w = np.ones_like(y, dtype=np.float64)

    fp = (p > 0) & (y < 0)
    fn = (p < 0) & (y > 0)

    if np.any(fp):
        w[fp] = 2.5 + 0.02 * np.abs(y[fp])
    if np.any(fn):
        w[fn] = 1.5 + 0.01 * y[fn]

    loss = float(np.mean(w * (p - y) ** 2))
    return ("awmse", loss, False)

def _train_lgb_reg_aw(X_tr, y_tr, X_va, y_va, seed: int):
    params = dict(
        boosting_type="gbdt",
        objective="regression",
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        seed=seed,
        verbose=-1,
        force_row_wise=True,
        metric="rmse",
    )
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=6000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return booster

def _train_lgb_reg_huber(X_tr, y_tr, X_va, y_va, seed: int):
    params = dict(
        boosting_type="gbdt",
        objective="huber",
        alpha=0.90,
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        seed=seed,
        verbose=-1,
        force_row_wise=True,
        metric="rmse",
    )
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=6000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return booster

def _train_lgb_quantile(X_tr, y_tr, X_va, y_va, seed: int, alpha: float = 0.20):
    params = dict(
        boosting_type="gbdt",
        objective="quantile",
        alpha=alpha,
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=60,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        seed=seed,
        verbose=-1,
        force_row_wise=True,
        metric="quantile",
    )
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=6000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    return booster

def _train_lgb_binary(X_tr, y_tr_bin, X_va, y_va_bin, seed: int):
    pos_rate = float(np.mean(y_tr_bin))
    pos_rate = min(max(pos_rate, 1e-6), 1 - 1e-6)
    scale_pos_weight = float((1.0 - pos_rate) / pos_rate)

    params = dict(
        boosting_type="gbdt",
        objective="binary",
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=80,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        seed=seed,
        verbose=-1,
        force_row_wise=True,
        metric="auc",
        scale_pos_weight=scale_pos_weight,
    )
    dtr = lgb.Dataset(X_tr, label=y_tr_bin.astype(np.float32))
    dva = lgb.Dataset(X_va, label=y_va_bin.astype(np.float32), reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return booster


def _linear_calibration(y_true: np.ndarray, pred: np.ndarray):
    y = y_true.astype(np.float64)
    p = pred.astype(np.float64)
    pm = p.mean()
    ym = y.mean()
    var = float(np.mean((p - pm) ** 2))
    if var < 1e-12:
        a = 1.0
        b = float(ym - a * pm)
        return float(a), float(b)
    cov = float(np.mean((p - pm) * (y - ym)))
    a = cov / var
    if a < 0:
        a = abs(a)
    b = float(ym - a * pm)
    return float(a), float(b)

def _blend_and_adjust(mean_pred, huber_pred, hurdle_pred, q20_pred, pneg,
                     m_mix: float, w_hurdle: float, w_q20: float,
                     risk_scale: float, thr_gate: float, thr_cap: float):
    mean_mix = m_mix * mean_pred + (1.0 - m_mix) * huber_pred
    w_mean = 1.0 - w_hurdle - w_q20
    base = w_mean * mean_mix + w_hurdle * hurdle_pred + w_q20 * q20_pred

    adj = base - risk_scale * pneg * np.abs(mean_mix - q20_pred)

    gate = (q20_pred < 0) | (pneg > thr_gate)
    adj = np.where(gate, np.minimum(adj, q20_pred), adj)

    adj = np.where(pneg > thr_cap, np.minimum(adj, -0.01), adj)

    return adj.astype(np.float32), mean_mix.astype(np.float32)

def main():
    seed_everything(42)

    print("=" * 80)
    print("Improved approach: LightGBM suite + OOF Target Encoding + Quantile/Risk heads + VAL affine calibration")
    print(" - Stronger user-level temporal features (aggregate + compact day-wise subset)")
    print(" - AWMSE custom objective for asymmetric cost")
    print(" - Quantile (τ=0.20) conservative bound")
    print(" - Negative-probability classifier used for risk adjustment")
    print(" - Post-hoc affine calibration on VAL to drastically improve budget/error_rate without hurting rank metrics")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    train_df = ensure_day_column(train_df)
    val_df   = ensure_day_column(val_df)
    test_df  = ensure_day_column(test_df)

    numeric_cols, cat_cols = infer_feature_columns(train_df)

    for df in (train_df, val_df, test_df):
        for c in numeric_cols:
            df[c] = df[c].fillna(0.0)
        df = _safe_float32(df, numeric_cols)

    fixed_seq = [c for c in sorted(set(NUMERICAL_COLS + TCN_TEMPORAL_FEATURES)) if c in numeric_cols]
    extra_k = 20
    var_rank = train_df[numeric_cols].astype(np.float32).std(axis=0).sort_values(ascending=False).index.tolist()
    extra = [c for c in var_rank if c not in fixed_seq][:extra_k]
    seq_cols = fixed_seq + extra
    seq_cols = [c for c in seq_cols if c in numeric_cols]
    seq_cols = seq_cols[:40]

    tr_ids, Xtr_num, tr_cat_last, tr_cat_nuniq, ytr = _make_user_level_base(train_df, numeric_cols, cat_cols, with_target=True)
    va_ids, Xva_num, va_cat_last, va_cat_nuniq, yva = _make_user_level_base(val_df, numeric_cols, cat_cols, with_target=True)
    te_ids, Xte_num, te_cat_last, te_cat_nuniq, _   = _make_user_level_base(test_df, numeric_cols, cat_cols, with_target=False)

    Xtr_seq = _make_seq_features(train_df, tr_ids, seq_cols)
    Xva_seq = _make_seq_features(val_df, va_ids, seq_cols)
    Xte_seq = _make_seq_features(test_df, te_ids, seq_cols)

    te_tr, te_bundle = _build_oof_target_encoding(tr_cat_last, ytr, cat_cols, n_splits=5, smooth=30.0, seed=42)
    te_va = _apply_target_encoding(va_cat_last, cat_cols, te_bundle)
    te_te = _apply_target_encoding(te_cat_last, cat_cols, te_bundle)

    tr_nuniq = tr_cat_nuniq.to_numpy(np.float32) if len(cat_cols) else np.zeros((len(tr_ids), 0), np.float32)
    va_nuniq = va_cat_nuniq.to_numpy(np.float32) if len(cat_cols) else np.zeros((len(va_ids), 0), np.float32)
    te_nuniq = te_cat_nuniq.to_numpy(np.float32) if len(cat_cols) else np.zeros((len(te_ids), 0), np.float32)

    Xtr = np.concatenate([Xtr_num, Xtr_seq, te_tr, tr_nuniq], axis=1).astype(np.float32)
    Xva = np.concatenate([Xva_num, Xva_seq, te_va, va_nuniq], axis=1).astype(np.float32)
    Xte = np.concatenate([Xte_num, Xte_seq, te_te, te_nuniq], axis=1).astype(np.float32)

    print(f"Users: train={len(tr_ids)} | val={len(va_ids)} | test={len(te_ids)}")
    print(f"Cols: numeric={len(numeric_cols)} | cat={len(cat_cols)} | seq_cols={len(seq_cols)}")
    print(f"Feature dim: Xtr={Xtr.shape[1]}")

    strat = _stratify_3class(ytr)
    tr_idx, es_idx = train_test_split(
        np.arange(len(ytr)),
        test_size=0.12,
        random_state=42,
        shuffle=True,
        stratify=strat
    )
    X_tr, y_tr = Xtr[tr_idx], ytr[tr_idx]
    X_es, y_es = Xtr[es_idx], ytr[es_idx]

    seeds = [42, 202]

    aw_models = []
    hu_models = []
    for sd in seeds:
        aw_models.append(_train_lgb_reg_aw(X_tr, y_tr, X_es, y_es, seed=sd))
        hu_models.append(_train_lgb_reg_huber(X_tr, y_tr, X_es, y_es, seed=sd))

    q20_model = _train_lgb_quantile(X_tr, y_tr, X_es, y_es, seed=42, alpha=0.20)

    neg_model = _train_lgb_binary(X_tr, (y_tr < 0).astype(np.int32), X_es, (y_es < 0).astype(np.int32), seed=42)
    zero_model = _train_lgb_binary(X_tr, (y_tr == 0).astype(np.int32), X_es, (y_es == 0).astype(np.int32), seed=202)

    pos_mask = y_tr > 0
    neg_mask = y_tr < 0

    if pos_mask.sum() < 200:
        pos_model = None
    else:
        pos_model = _train_lgb_reg_huber(X_tr[pos_mask], y_tr[pos_mask], X_es, y_es, seed=77)

    if neg_mask.sum() < 200:
        neg_reg_model = None
    else:
        neg_reg_model = _train_lgb_reg_huber(X_tr[neg_mask], y_tr[neg_mask], X_es, y_es, seed=88)

    def ens_predict(models, X):
        ps = []
        for m in models:
            ps.append(m.predict(X, num_iteration=m.best_iteration))
        return np.mean(np.stack(ps, axis=0), axis=0).astype(np.float32)

    aw_val = ens_predict(aw_models, Xva)
    hu_val = ens_predict(hu_models, Xva)
    q20_val = q20_model.predict(Xva, num_iteration=q20_model.best_iteration).astype(np.float32)

    pneg_val = neg_model.predict(Xva, num_iteration=neg_model.best_iteration).astype(np.float32)
    pzero_val = zero_model.predict(Xva, num_iteration=zero_model.best_iteration).astype(np.float32)

    if pos_model is None:
        pos_val = np.maximum(aw_val, 0.0)
    else:
        pos_val = np.maximum(pos_model.predict(Xva, num_iteration=pos_model.best_iteration).astype(np.float32), 0.0)

    if neg_reg_model is None:
        neg_val = np.minimum(aw_val, 0.0)
    else:
        neg_val = np.minimum(neg_reg_model.predict(Xva, num_iteration=neg_reg_model.best_iteration).astype(np.float32), 0.0)

    hurdle_val = (pneg_val * neg_val + (1.0 - pneg_val) * (1.0 - pzero_val) * pos_val).astype(np.float32)

    aw_test = ens_predict(aw_models, Xte)
    hu_test = ens_predict(hu_models, Xte)
    q20_test = q20_model.predict(Xte, num_iteration=q20_model.best_iteration).astype(np.float32)

    pneg_test = neg_model.predict(Xte, num_iteration=neg_model.best_iteration).astype(np.float32)
    pzero_test = zero_model.predict(Xte, num_iteration=zero_model.best_iteration).astype(np.float32)

    if pos_model is None:
        pos_test = np.maximum(aw_test, 0.0)
    else:
        pos_test = np.maximum(pos_model.predict(Xte, num_iteration=pos_model.best_iteration).astype(np.float32), 0.0)

    if neg_reg_model is None:
        neg_test = np.minimum(aw_test, 0.0)
    else:
        neg_test = np.minimum(neg_reg_model.predict(Xte, num_iteration=neg_reg_model.best_iteration).astype(np.float32), 0.0)

    hurdle_test = (pneg_test * neg_test + (1.0 - pneg_test) * (1.0 - pzero_test) * pos_test).astype(np.float32)

    grid_m_mix = [0.50, 0.70, 0.90]
    grid_w_hurdle = [0.00, 0.20, 0.40]
    grid_w_q20 = [0.10, 0.20, 0.30, 0.40]
    grid_risk = [0.00, 0.30, 0.60]
    grid_thr_gate = [0.55, 0.65, 0.75]
    grid_thr_cap = [0.80, 0.90]

    best = None
    best_score = -1e18
    best_meta = None

    for m_mix in grid_m_mix:
        for w_h in grid_w_hurdle:
            for w_q in grid_w_q20:
                if w_h + w_q >= 0.95:
                    continue
                for rs in grid_risk:
                    for tg in grid_thr_gate:
                        for tc in grid_thr_cap:
                            pred_raw, mean_mix_val = _blend_and_adjust(
                                aw_val, hu_val, hurdle_val, q20_val, pneg_val,
                                m_mix=m_mix, w_hurdle=w_h, w_q20=w_q,
                                risk_scale=rs, thr_gate=tg, thr_cap=tc
                            )
                            a, b = _linear_calibration(yva, pred_raw)
                            pred = (a * pred_raw + b).astype(np.float32)

                            pred = np.clip(pred, -1500.0, 3500.0)

                            sc, met = compute_score_silent(yva, pred)
                            if not np.isfinite(sc):
                                continue
                            if sc > best_score:
                                best_score = sc
                                best = (m_mix, w_h, w_q, rs, tg, tc, a, b)
                                best_meta = met

    print("\n--- VAL calibration search (best) ---")
    print(f"best_val_score={best_score:.6f} | metrics={best_meta}")
    print(f"params: m_mix={best[0]:.2f}, w_hurdle={best[1]:.2f}, w_q20={best[2]:.2f}, "
          f"risk_scale={best[3]:.2f}, thr_gate={best[4]:.2f}, thr_cap={best[5]:.2f}, "
          f"a={best[6]:.4f}, b={best[7]:.4f}")

    m_mix, w_h, w_q, rs, tg, tc, a, b = best
    pred_raw_test, _ = _blend_and_adjust(
        aw_test, hu_test, hurdle_test, q20_test, pneg_test,
        m_mix=m_mix, w_hurdle=w_h, w_q20=w_q,
        risk_scale=rs, thr_gate=tg, thr_cap=tc
    )
    pred_test = (a * pred_raw_test + b).astype(np.float32)
    pred_test = np.clip(pred_test, -1500.0, 3500.0)

    y_test = test_df.groupby(ID_COL)[TARGET_COL].first()
    pred_aligned = pd.Series(pred_test, index=te_ids).reindex(y_test.index).to_numpy(np.float32)

    print("\n" + "=" * 80)
    score0, metrics = compute_score(y_test.values, pred_aligned)
    print("=" * 80)

    out = {
        "method": "LightGBM suite (AWMSE+Huber+Quantile+Hurdle) + OOF TE + VAL affine calibration",
        "best_val_score": float(best_score),
        "best_val_metrics": best_meta,
        "best_params": {
            "m_mix": float(m_mix),
            "w_hurdle": float(w_h),
            "w_q20": float(w_q),
            "risk_scale": float(rs),
            "thr_gate": float(tg),
            "thr_cap": float(tc),
            "a": float(a),
            "b": float(b),
        },
        "score": float(score0),
        "metrics": metrics,
        "n_train_users": int(len(tr_ids)),
        "n_val_users": int(len(va_ids)),
        "n_test_users": int(len(te_ids)),
        "n_features": int(Xtr.shape[1]),
        "seq_cols": seq_cols,
    }
    out_path = "/home/jupyter/anonymous_institutionAgentMonty_1226/tasks/BT_IOS_2503_Pareto/run_deepresearch/lgbm_suite_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    return y_test.values, pred_aligned

if __name__ == "__main__":
    y_test, test_predictions = main()

    score = compute_pareto_multi_objective(y_test, test_predictions)
    print(f"score = {score}")
