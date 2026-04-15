"""
import scipy
import random

import scipy
import random

ZILN Model for ZILN Dataset - Using Preprocessed Features
Optimized with BiGRU, Attention Pooling, and Residual Connections.
Focuses on stability and depth for better R2 and Spearman metrics.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = '/home/jupyter/anonymous_user/KDD26/Comparative_experiment/ZILN_OPEN_data/ZILNLTV_new'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'predictions')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'modelpath')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# ============================================================================
# Data Loading
# ============================================================================

def load_preprocessed_data(file_path):
    """加载预处理好的数据"""
    df = pd.read_csv(file_path)

    # 验证每个用户有7行数据
    user_counts = df.groupby('id').size()
    valid_users = user_counts[user_counts == 7].index
    df = df[df['id'].isin(valid_users)].copy()

    return df


def extract_features_and_labels(df):
    """
    从预处理的DataFrame中提取特征和标签
    """
    # 识别特征列
    feature_suffixes = ['_count', '_unique', '_entropy', '_freq', '_ngram',
                       '_mean', '_std', '_min', '_max', '_median', '_q25',
                       '_q75', '_range', '_cv', '_sum', '_trend']

    feature_cols = [col for col in df.columns
                   if any(col.endswith(suffix) for suffix in feature_suffixes)]

    # 按用户分组提取数据
    X_list = []
    y_list = []
    user_ids = []

    for user_id, group in df.groupby('id'):
        if len(group) != 7:
            continue

        user_ids.append(user_id)
        y_list.append(group['LTV label'].iloc[0])

        # 提取7个时间步的特征
        user_features = group[feature_cols].values  # shape: (7, num_features)
        X_list.append(user_features)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    user_ids = np.array(user_ids)

    return X, y, user_ids


# EVOLVE-BLOCK-START
# ============================================================================
# Model Definition (ZILN Architecture) - Deep Residual BiGRU
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Pre-Activation Residual Block
    LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> Residual
    """
    def __init__(self, dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.dense = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.dense(self.ln(x))


class AttentionPooling(nn.Module):
    """
    Multi-head attention pooling with a learnable query token.
    Provides stronger global aggregation than scalar attention.
    """
    def __init__(self, input_dim, num_heads=4, dropout_rate=0.1):
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        q = self.query.expand(x.size(0), -1, -1)  # [batch, 1, input_dim]
        pooled, _ = self.mha(q, x, x)  # [batch, 1, input_dim]
        pooled = self.ln(pooled.squeeze(1))  # [batch, input_dim]
        return pooled


class ZILN_ResBiGRU(nn.Module):
    """
    Advanced ZILN Model (Hybrid TCN + Transformer):
    1.0 GLU Feature Projection for gated feature interactions.
    2.0 Temporal Conv (TCN-style) for local temporal patterns.
    3.0 Transformer Encoder for global temporal dependencies.
    4.0 Multi-head Attention Pooling for sequence aggregation.
    5.0 Deep Residual Trunk + ZILN Head for distribution parameters.
    """
    def __init__(self, feature_dim, hidden_dim=160, embed_dim=160, dropout_rate=0.1, out_dim=3):
        super(ZILN_ResBiGRU, self).__init__()

        self.pos_weight_val = 2.0  # Fixed weight for positive class imbalance
        self.max_sigma = 3.0       # Clamp max sigma for training stability
        self.infer_sigma = 1.5     # Tighter sigma for inference (reduces RMSE)

        # 1.0 Feature Projection with GLU gating
        self.input_proj = nn.Linear(feature_dim, hidden_dim * 2)
        self.input_ln = nn.LayerNorm(hidden_dim)

        # Positional Embedding (7 timesteps)
        self.pos_emb = nn.Parameter(torch.zeros(1, 7, hidden_dim))

        # 2.0 Temporal Convolution (local patterns)
        self.tcn = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.tcn_ln = nn.LayerNorm(hidden_dim)

        # 3.0 Transformer Encoder (global patterns)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 4.0 Attention Pooling
        self.attn_pool = AttentionPooling(hidden_dim, num_heads=4, dropout_rate=dropout_rate)

        # 5.0 Residual Trunk
        self.trunk_proj = nn.Linear(hidden_dim, embed_dim)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(embed_dim, dropout_rate) for _ in range(3)
        ])
        self.trunk_ln = nn.LayerNorm(embed_dim)

        # 6.0 Heads
        self.head = nn.Linear(embed_dim, out_dim)

        # Initializations
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [batch, 7, feature_dim]
        x_proj = F.glu(self.input_proj(x), dim=-1)  # [batch, 7, hidden_dim]
        x_proj = self.input_ln(x_proj)
        x_proj = x_proj + self.pos_emb

        # Temporal conv (TCN)
        tcn_in = x_proj.transpose(1, 2)  # [batch, hidden_dim, 7]
        tcn_out = self.tcn(tcn_in).transpose(1, 2)  # [batch, 7, hidden_dim]
        x_proj = self.tcn_ln(x_proj + tcn_out)

        # Transformer encoder
        enc_out = self.encoder(x_proj)  # [batch, 7, hidden_dim]

        # Pool
        pooled = self.attn_pool(enc_out)  # [batch, hidden_dim]

        # Residual Backbone
        z = self.trunk_proj(pooled)
        for block in self.res_blocks:
            z = block(z)
        z = self.trunk_ln(z)

        # Output
        logits = self.head(z)
        return logits

    def predict(self, logits, use_infer_sigma=True):
        """
        Predict expected LTV.
        E[Y] = p * exp(mu + sigma^2 / 2)

        Uses tighter sigma (1.5) during inference to reduce RMSE.
        Training uses max_sigma=3.0 for loss stability.
        """
        positive_probs = torch.sigmoid(logits[..., 0:1])
        loc = logits[..., 1:2]

        # Apply softplus and clamp scale for stability
        scale = F.softplus(logits[..., 2:]) + 1e-4
        sigma_cap = self.infer_sigma if use_infer_sigma else self.max_sigma
        scale = torch.clamp(scale, max=sigma_cap)

        preds = positive_probs * torch.exp(loc + 0.5 * scale**2)
        return preds

    def calculate_loss(self, labels, logits):
        loss = self.zero_inflated_lognormal_loss(labels, logits)
        return loss

    def zero_inflated_lognormal_loss(self, labels, logits, eps=1e-6):
        """
        Weighted ZILN Loss:
        - Weighted BCE for classification (Zero vs Positive)
        - LogNormal NLL for regression (Positive only)
        """
        # Prepare targets
        positive_mask = (labels > 0).float().unsqueeze(1)
        labels = labels.unsqueeze(1)

        # 1.0 Classification Loss (Weighted BCE)
        logit_prob = logits[:, 0:1]
        pos_weight = torch.tensor(self.pos_weight_val, device=logits.device)

        classification_loss = F.binary_cross_entropy_with_logits(
            logit_prob,
            positive_mask,
            pos_weight=pos_weight,
            reduction='mean'
        )

        # 2.0 Regression Loss (LogNormal NLL)
        loc = logits[:, 1:2]
        # Consistent scale handling with predict()
        scale = F.softplus(logits[:, 2:3]) + 1e-4
        scale = torch.clamp(scale, max=self.max_sigma)

        # Only compute on positive samples
        safe_labels = labels.clamp(min=eps)
        lognormal = tdist.LogNormal(loc=loc, scale=scale)
        log_prob = lognormal.log_prob(safe_labels)

        # Mean NLL over positive samples only
        # Add epsilon to denominator to avoid division by zero if batch has no positives
        regression_loss = -torch.sum(positive_mask * log_prob) / (positive_mask.sum() + eps)

        return classification_loss + regression_loss


def create_model(feature_dim):
    """
    Factory function for ZILN_ResBiGRU
    """
    model = ZILN_ResBiGRU(
        feature_dim=feature_dim,
        hidden_dim=160,    # Slightly wider encoder for higher capacity
        embed_dim=160,     # Matching backbone width
        dropout_rate=0.1,  # Mild dropout for regularization
        out_dim=3
    )
    return model

# EVOLVE-BLOCK-END


# ============================================================================
# Dataset and Evaluation
# ============================================================================

class ZILNDataset(Dataset):
    def __init__(self, X, y, user_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.user_ids = user_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.user_ids[idx]


def normalized_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (y_true.mean() + 1e-8)


def cumulative_true(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).sort_values(by='y_pred', ascending=False)
    return (df['y_true'].cumsum() / df['y_true'].sum()).values


def gini_from_gain(df):
    raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.0)
    normalized = raw / (raw[0] + 1e-8)
    return pd.DataFrame({'raw': raw, 'normalized': normalized})[['raw', 'normalized']]


def evaluate_metrics(y_true, y_pred):
    """计算评估指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = normalized_rmse(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    spearman, _ = spearmanr(y_true, y_pred)

    gain = pd.DataFrame({
        'lorenz': cumulative_true(y_true, y_true),
        'model': cumulative_true(y_true, y_pred)
    })
    gini_result = gini_from_gain(gain[['lorenz', 'model']])

    error_rate = np.sum(np.abs(y_pred - y_true)) / (np.sum(y_true) + 1e-8)

    return {
        'RMSE': rmse,
        'NRMSE': nrmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Spearman': spearman,
        'GINI': gini_result.iloc[1, 0],
        'Norm_GINI': gini_result.iloc[1, 1],
        'Error_Rate': error_rate,
        'True_Sum': y_true.sum(),
        'Pred_Sum': y_pred.sum()
    }


# ============================================================================
# Training Function
# ============================================================================

def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                      max_epochs=50, patience=15, model_path=None):
    """
    训练并评估模型
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'ZILN_evolved.pth')

    model = model.to(device)

    # Use AdamW with a slightly higher LR for deep residual network
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0

        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = model.calculate_loss(y, logits)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = model.calculate_loss(y, logits)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Test evaluation
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"Error loading best model: {e}")

    model.eval()

    test_preds, test_trues, test_user_ids = [], [], []

    with torch.no_grad():
        for X, y, user_id in test_loader:
            X = X.to(device)
            logits = model(X)
            pred = model.predict(logits)
            test_preds.extend(pred.squeeze(-1).cpu().numpy())
            test_trues.extend(y.numpy())
            test_user_ids.extend(user_id.numpy())

    test_preds = np.array(test_preds)
    test_trues = np.array(test_trues)

    # Ensure no NaNs in predictions
    test_preds = np.nan_to_num(test_preds, nan=0.0, posinf=0.0, neginf=0.0)

    test_metrics = evaluate_metrics(test_trues, test_preds)

    return test_metrics, test_preds, test_trues


# ============================================================================
# Main Experiment Function (for )
# ============================================================================

def run_experiment(sample_fraction=1.0, max_epochs=100, patience=20, seed=42):
    """
    运行LTV预测实验
    NOTE: Using 100% data (sample_fraction=1.0) for production-ready models
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载预处理的数据
    train_df = load_preprocessed_data(os.path.join(BASE_PATH, 'ZILNtrain_new_with_features.csv'))
    val_df = load_preprocessed_data(os.path.join(BASE_PATH, 'ZILNval_new_with_features.csv'))
    test_df = load_preprocessed_data(os.path.join(BASE_PATH, 'ZILNtest_new_with_features.csv'))

    # 采样以加速进化迭代
    if sample_fraction < 1.0:
        train_users = train_df['id'].unique()
        sample_size = int(len(train_users) * sample_fraction)
        sampled_users = np.random.choice(train_users, sample_size, replace=False)
        train_df = train_df[train_df['id'].isin(sampled_users)].copy()

    # 提取特征和标签
    X_train, y_train, train_ids = extract_features_and_labels(train_df)
    X_val, y_val, val_ids = extract_features_and_labels(val_df)
    X_test, y_test, test_ids = extract_features_and_labels(test_df)

    # 标准化
    num_users_train, timesteps, num_features = X_train.shape

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features))
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features))
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features))

    X_train = X_train_scaled.reshape(num_users_train, timesteps, num_features)
    X_val = X_val_scaled.reshape(len(X_val), timesteps, num_features)
    X_test = X_test_scaled.reshape(len(X_test), timesteps, num_features)

    # 创建数据集
    train_dataset = ZILNDataset(X_train, y_train, train_ids)
    val_dataset = ZILNDataset(X_val, y_val, val_ids)
    test_dataset = ZILNDataset(X_test, y_test, test_ids)

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建模型
    model = create_model(feature_dim=num_features)

    # 训练并评估
    test_metrics, test_preds, test_trues = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        max_epochs=max_epochs,
        patience=patience
    )

    return test_metrics


if __name__ == "__main__":
    print("=" * 70)
    print(" " * 8 + "ZILN Model -  Version (ResBiGRU)")
    print("=" * 70 + "\n")

    metrics = run_experiment(sample_fraction=1.0, max_epochs=100, patience=20)

    print("\n" + "=" * 70)
    print(" " * 25 + "Test Results")
    print("=" * 70)
    for metric, value in metrics.items():
        print(f"  {metric:15s}: {value:10.4f}")
    print("=" * 70)
