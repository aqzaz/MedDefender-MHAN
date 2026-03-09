#!/usr/bin/env python3
"""
MedDefender-MHAN: An Explainable Multi-Head Attention Network for Healthcare IoT Threat Detection

Complete implementation including:
- Data preprocessing and normalization
- Hierarchical Feature Extraction (CNN + Transformer streams)
- Multi-Head Attention Encoder
- Threat Classification Module
- Explainability Generation Module
- Training and evaluation pipelines
- Visualization utilities

Author: Implementation based on the MedDefender-MHAN paper
"""

import os
import math
import time
import random
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MHANConfig:
    """Configuration for MedDefender-MHAN model and training."""
    input_dim: int = 78
    temporal_window: int = 64
    model_dim: int = 256
    num_heads: int = 8
    attention_dim: int = 64
    cnn_layers: int = 4
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    transformer_layers: int = 3
    ffn_dim: int = 512
    num_classes: int = 15
    dropout: float = 0.3
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 100
    weight_decay: float = 1e-5
    focal_gamma: float = 2.0
    explainability_weight: float = 0.6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    model_save_path: str = "meddefender_mhan_best.pth"


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class NetworkTrafficDataset(Dataset):
    """Dataset class for network traffic data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, temporal_window: int = 64):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.temporal_window = temporal_window
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        x = self._reshape_to_temporal(x)
        return x, y
    
    def _reshape_to_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape features to temporal sequence format (Eq. 3)."""
        d = x.shape[0]
        T = self.temporal_window
        if d % T != 0:
            pad_size = T - (d % T)
            x = F.pad(x, (0, pad_size), mode='constant', value=0)
        return x.view(T, -1)


class DataPreprocessor:
    """Data preprocessing pipeline for MedDefender-MHAN."""
    
    def __init__(self, config: MHANConfig):
        self.config = config
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.heavy_tailed_cols: List[int] = []
        
    def fit_transform(self, df: pd.DataFrame, label_col: str = 'Label') -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data."""
        X = df.drop(columns=[label_col]).values
        y = df[label_col].values
        self._identify_heavy_tailed(X)
        X = self._apply_log_transform(X)
        X = self.scaler.fit_transform(X)
        y = self.label_encoder.fit_transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
    def transform(self, df: pd.DataFrame, label_col: str = 'Label') -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted preprocessor."""
        X = df.drop(columns=[label_col]).values
        y = df[label_col].values
        X = self._apply_log_transform(X)
        X = self.scaler.transform(X)
        y = self.label_encoder.transform(y)
        return X.astype(np.float32), y.astype(np.int64)
    
    def _identify_heavy_tailed(self, X: np.ndarray, skew_threshold: float = 3.0):
        """Identify columns with heavy-tailed distributions."""
        try:
            from scipy.stats import skew
            self.heavy_tailed_cols = []
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                col_data = col_data[~np.isnan(col_data)]
                if len(col_data) > 0 and skew(col_data) > skew_threshold:
                    self.heavy_tailed_cols.append(col_idx)
        except ImportError:
            self.heavy_tailed_cols = []
    
    def _apply_log_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply log transformation to heavy-tailed columns (Eq. 2)."""
        X = X.copy()
        for col_idx in self.heavy_tailed_cols:
            X[:, col_idx] = np.log1p(np.abs(X[:, col_idx]))
        return X


def create_dataloaders(X: np.ndarray, y: np.ndarray, config: MHANConfig,
                       test_size: float = 0.1, val_size: float = 0.1,
                       use_weighted_sampler: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train)
    
    train_dataset = NetworkTrafficDataset(X_train, y_train, config.temporal_window)
    val_dataset = NetworkTrafficDataset(X_val, y_val, config.temporal_window)
    test_dataset = NetworkTrafficDataset(X_test, y_test, config.temporal_window)
    
    sampler = None
    if use_weighted_sampler:
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer (Eq. 7-9)."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNStream(nn.Module):
    """CNN stream for local spatial feature extraction (Eq. 4-7)."""
    
    def __init__(self, input_channels: int, channels: List[int], kernel_sizes: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = input_channels
        for out_ch, k in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.output_channels = channels[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        return x


class TransformerStream(nn.Module):
    """Transformer stream for long-range temporal dependencies (Eq. 10-14)."""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return x


class MultiHeadAttentionEncoder(nn.Module):
    """Multi-Head Attention Encoder (Eq. 15-20)."""
    
    def __init__(self, d_model: int, num_heads: int, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, num_heads * d_k)
        self.W_k = nn.Linear(d_model, num_heads * d_k)
        self.W_v = nn.Linear(d_model, num_heads * d_k)
        self.W_o = nn.Linear(num_heads * d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention_weights: Optional[torch.Tensor] = None
        
    def forward(self, x: torch.Tensor, return_attention: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        self.attention_weights = attention_weights.detach()
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(context)
        output = self.layer_norm(x + self.dropout(output))
        
        if return_attention:
            return output, attention_weights
        return output


class FeatureFusion(nn.Module):
    """Feature fusion module (Eq. 15)."""
    
    def __init__(self, cnn_dim: int, trans_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(cnn_dim + trans_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, cnn_features: torch.Tensor, trans_features: torch.Tensor) -> torch.Tensor:
        if cnn_features.size(1) != trans_features.size(1):
            cnn_features = F.interpolate(cnn_features.transpose(1, 2), size=trans_features.size(1),
                                          mode='linear', align_corners=False).transpose(1, 2)
        fused = torch.cat([cnn_features, trans_features], dim=-1)
        fused = self.projection(fused)
        fused = self.layer_norm(fused)
        return fused


class ThreatClassifier(nn.Module):
    """Threat classification module (Eq. 21-25)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.layer_norm2(x)
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits


class ExplainabilityModule(nn.Module):
    """Explainability generation module (Eq. 26-32)."""
    
    def __init__(self, num_heads: int, lambda_weight: float = 0.6):
        super().__init__()
        self.num_heads = num_heads
        self.lambda_weight = lambda_weight
        
    def compute_feature_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute feature importance from attention weights (Eq. 27-28)."""
        avg_attention = attention_weights.mean(dim=1)
        importance = avg_attention.sum(dim=1)
        importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
        return importance
    
    def compute_gradient_weighted_attention(self, attention_weights: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """Compute gradient-weighted attention map (Eq. 29-30)."""
        alpha = gradients.mean(dim=(2, 3), keepdim=True)
        weighted_attention = (alpha * attention_weights).sum(dim=1)
        grad_cam = F.relu(weighted_attention)
        return grad_cam
    
    def identify_temporal_patterns(self, attention_weights: torch.Tensor, window_size: int = 8) -> torch.Tensor:
        """Identify temporal attack patterns (Eq. 31)."""
        avg_attention = attention_weights.mean(dim=1)
        seq_len = avg_attention.size(1)
        num_windows = max(1, seq_len // window_size)
        patterns = []
        for i in range(num_windows):
            start = i * window_size
            end = min((i + 1) * window_size, seq_len)
            window_attention = avg_attention[:, start:end, :].mean(dim=(1, 2))
            patterns.append(window_attention)
        return torch.stack(patterns, dim=1)
    
    def generate_explanation(self, attention_weights: torch.Tensor, gradients: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate comprehensive explanation (Eq. 32)."""
        explanation = {
            'feature_importance': self.compute_feature_importance(attention_weights),
            'temporal_patterns': self.identify_temporal_patterns(attention_weights)
        }
        if gradients is not None:
            explanation['grad_weighted_attention'] = self.compute_gradient_weighted_attention(attention_weights, gradients)
        explanation['combined_score'] = self.lambda_weight * explanation['feature_importance'] + (1 - self.lambda_weight) * explanation['feature_importance']
        return explanation


# ============================================================================
# MAIN MODEL
# ============================================================================

class MedDefenderMHAN(nn.Module):
    """MedDefender-MHAN: Explainable Multi-Head Attention Network for Healthcare IoT Threat Detection."""
    
    def __init__(self, config: MHANConfig):
        super().__init__()
        self.config = config
        
        features_per_step = config.input_dim // config.temporal_window
        if config.input_dim % config.temporal_window != 0:
            features_per_step += 1
        
        self.input_projection = nn.Linear(features_per_step, config.model_dim)
        
        self.cnn_stream = CNNStream(input_channels=config.model_dim, channels=config.cnn_channels,
                                     kernel_sizes=config.cnn_kernel_sizes, dropout=config.dropout)
        
        self.transformer_stream = TransformerStream(d_model=config.model_dim, nhead=config.num_heads,
                                                     num_layers=config.transformer_layers, dim_feedforward=config.ffn_dim,
                                                     dropout=config.dropout)
        
        self.feature_fusion = FeatureFusion(cnn_dim=config.cnn_channels[-1], trans_dim=config.model_dim, output_dim=config.model_dim)
        
        self.mha_encoder = MultiHeadAttentionEncoder(d_model=config.model_dim, num_heads=config.num_heads,
                                                      d_k=config.attention_dim, dropout=config.dropout)
        
        self.classifier = ThreatClassifier(input_dim=config.model_dim, hidden_dim=config.model_dim * 2,
                                            num_classes=config.num_classes, dropout=config.dropout)
        
        self.explainability = ExplainabilityModule(num_heads=config.num_heads, lambda_weight=config.explainability_weight)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False, return_explanation: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass through MedDefender-MHAN."""
        x = self.input_projection(x)
        cnn_features = self.cnn_stream(x)
        trans_features = self.transformer_stream(x)
        fused_features = self.feature_fusion(cnn_features, trans_features)
        attn_features, attention_weights = self.mha_encoder(fused_features, return_attention=True)
        logits = self.classifier(attn_features)
        
        if return_explanation or return_attention:
            explanation = {'attention_weights': attention_weights}
            if return_explanation:
                explanation.update(self.explainability.generate_explanation(attention_weights))
            return logits, explanation
        return logits
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self.mha_encoder.attention_weights


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Eq. 26)."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss with focal loss."""
    
    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=class_weights)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(logits, targets)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """Training pipeline for MedDefender-MHAN."""
    
    def __init__(self, model: MedDefenderMHAN, config: MHANConfig, train_loader: DataLoader,
                 val_loader: DataLoader, class_weights: Optional[torch.Tensor] = None):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = CombinedLoss(gamma=config.focal_gamma,
                                       class_weights=class_weights.to(config.device) if class_weights is not None else None)
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.scaler = GradScaler() if config.use_amp and config.device == 'cuda' else None
        self.train_losses, self.val_losses, self.train_accs, self.val_accs = [], [], [], []
        self.best_val_acc, self.best_val_f1 = 0.0, 0.0
        
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for data, targets in self.train_loader:
            data, targets = data.to(self.config.device), targets.to(self.config.device)
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    logits = self.model(data)
                    loss = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(data)
                loss = self.criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(self.train_loader), 100.0 * correct / total
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        for data, targets in self.val_loader:
            data, targets = data.to(self.config.device), targets.to(self.config.device)
            logits = self.model(data)
            loss = self.criterion(logits, targets)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * accuracy_score(all_targets, all_preds)
        f1 = 100.0 * f1_score(all_targets, all_preds, average='weighted')
        return avg_loss, accuracy, f1
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        epochs = epochs or self.config.epochs
        print(f"Training MedDefender-MHAN for {epochs} epochs on {self.config.device}...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.scheduler.step(val_loss)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc, self.best_val_f1 = val_acc, val_f1
                self.save_checkpoint(self.config.model_save_path)
            
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | Time: {time.time()-start_time:.1f}s")
        
        print(f"\nBest Val Acc: {self.best_val_acc:.2f}%, Best Val F1: {self.best_val_f1:.2f}%")
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses,
                'train_accs': self.train_accs, 'val_accs': self.val_accs}
    
    def save_checkpoint(self, path: str):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config, 'best_val_acc': self.best_val_acc}, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


class Evaluator:
    """Evaluation pipeline for MedDefender-MHAN."""
    
    def __init__(self, model: MedDefenderMHAN, config: MHANConfig, test_loader: DataLoader,
                 class_names: Optional[List[str]] = None):
        self.model = model.to(config.device)
        self.config = config
        self.test_loader = test_loader
        self.class_names = class_names or [f"Class_{i}" for i in range(config.num_classes)]
        
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Union[float, np.ndarray]]:
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []
        total_time, num_samples = 0.0, 0
        
        for data, targets in self.test_loader:
            data, targets = data.to(self.config.device), targets.to(self.config.device)
            start_time = time.time()
            logits, _ = self.model(data, return_attention=True)
            total_time += time.time() - start_time
            num_samples += data.size(0)
            
            probs = F.softmax(logits, dim=-1)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds, all_targets, all_probs = np.array(all_preds), np.array(all_targets), np.array(all_probs)
        
        results = {
            'accuracy': 100.0 * accuracy_score(all_targets, all_preds),
            'precision': 100.0 * precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': 100.0 * recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1_score': 100.0 * f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'classification_report': classification_report(all_targets, all_preds, target_names=self.class_names, digits=4, zero_division=0),
            'predictions': all_preds, 'targets': all_targets, 'probabilities': all_probs,
            'avg_inference_time_ms': 1000.0 * total_time / num_samples,
            'throughput_samples_per_sec': num_samples / total_time
        }
        
        # ROC curves
        results['roc_curves'] = {}
        for i, name in enumerate(self.class_names):
            binary_targets = (all_targets == i).astype(int)
            if binary_targets.sum() > 0:
                fpr, tpr, _ = roc_curve(binary_targets, all_probs[:, i])
                results['roc_curves'][name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
        
        return results
    
    def print_results(self, results: Dict):
        print("\n" + "=" * 60 + "\nEVALUATION RESULTS\n" + "=" * 60)
        print(f"\nOverall: Acc={results['accuracy']:.2f}%, Prec={results['precision']:.2f}%, "
              f"Rec={results['recall']:.2f}%, F1={results['f1_score']:.2f}%")
        print(f"Inference: {results['avg_inference_time_ms']:.2f}ms/sample, {results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"\n{results['classification_report']}")


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Visualization utilities."""
    
    def __init__(self, save_dir: str = "./figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_curves(self, history: Dict[str, List[float]], save_name: str = "training_curves.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history['train_losses'], label='Train'); axes[0].plot(history['val_losses'], label='Val')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].set_title('Loss Curves')
        axes[1].plot(history['train_accs'], label='Train'); axes[1].plot(history['val_accs'], label='Val')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(); axes[1].set_title('Accuracy Curves')
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, save_name), dpi=150); plt.close()
        
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_name: str = "confusion_matrix.png"):
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix')
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, save_name), dpi=150); plt.close()
        
    def plot_roc_curves(self, roc_curves: Dict[str, Dict], save_name: str = "roc_curves.png"):
        fig, ax = plt.subplots(figsize=(10, 8))
        for name, data in roc_curves.items():
            ax.plot(data['fpr'], data['tpr'], label=f"{name} (AUC={data['auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC Curves'); ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, save_name), dpi=150); plt.close()
        
    def plot_attention_heatmap(self, attention_weights: torch.Tensor, save_name: str = "attention_heatmap.png"):
        attn = attention_weights[0].mean(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attn, cmap='viridis', ax=ax)
        ax.set_xlabel('Key'); ax.set_ylabel('Query'); ax.set_title('Attention Weights')
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, save_name), dpi=150); plt.close()


# ============================================================================
# DATA UTILITIES
# ============================================================================

def create_synthetic_dataset(n_samples: int = 10000, n_features: int = 78, n_classes: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Create synthetic dataset for testing."""
    np.random.seed(random_state)
    features = []
    for i in range(n_features):
        if i % 3 == 0:
            col = np.random.randn(n_samples) * 10 + 50
        elif i % 3 == 1:
            col = np.random.exponential(scale=100, size=n_samples)
        else:
            col = np.random.uniform(0, 1000, size=n_samples)
        features.append(col)
    X = np.column_stack(features)
    
    class_probs = np.random.dirichlet(np.ones(n_classes) * 0.5)
    y = np.random.choice(n_classes, size=n_samples, p=class_probs)
    
    for cls in range(n_classes):
        mask = y == cls
        start_idx = cls * (n_features // n_classes)
        end_idx = min((cls + 1) * (n_features // n_classes), n_features)
        X[mask, start_idx:end_idx] += cls * 10
    
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])
    class_names = ['Benign', 'DoS', 'DDoS', 'PortScan', 'BruteForce', 'Ransomware', 'Backdoor', 'Injection', 'XSS', 'MITM'][:n_classes]
    df['Label'] = [class_names[i] for i in y]
    print(f"Created synthetic dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    return df


def load_cicids2017(data_path: str) -> pd.DataFrame:
    """Load CICIDS2017 dataset."""
    print("Loading CICIDS2017...")
    if os.path.isdir(data_path):
        dfs = [pd.read_csv(os.path.join(data_path, f), low_memory=False) for f in os.listdir(data_path) if f.endswith('.csv')]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=df.columns[df.nunique() <= 1], inplace=True)
    print(f"Loaded {len(df)} samples")
    return df


def load_toniot(data_path: str) -> pd.DataFrame:
    """Load TON_IoT dataset."""
    print("Loading TON_IoT...")
    if os.path.isdir(data_path):
        dfs = [pd.read_csv(os.path.join(data_path, f), low_memory=False) for f in os.listdir(data_path) if f.endswith('.csv')]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=df.columns[df.nunique() <= 1], inplace=True)
    print(f"Loaded {len(df)} samples")
    return df


# ============================================================================
# MAIN
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 60)
    print("MedDefender-MHAN: Healthcare IoT Threat Detection")
    print("=" * 60)
    
    set_seed(42)
    
    config = MHANConfig(input_dim=78, temporal_window=64, model_dim=256, num_heads=8, attention_dim=64,
                        cnn_layers=4, transformer_layers=3, num_classes=10, dropout=0.3, batch_size=256,
                        learning_rate=0.001, epochs=50, focal_gamma=2.0)
    
    print(f"\nConfig: device={config.device}, model_dim={config.model_dim}, heads={config.num_heads}")
    
    # Create synthetic data for demo
    df = create_synthetic_dataset(n_samples=50000, n_features=config.input_dim, n_classes=config.num_classes)
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    X, y = preprocessor.fit_transform(df, label_col='Label')
    config.num_classes = len(np.unique(y))
    
    # Class weights
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(len(y) / (len(class_counts) * class_counts))
    
    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(X, y, config)
    
    # Model
    model = MedDefenderMHAN(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
    
    # Train
    trainer = Trainer(model, config, train_loader, val_loader, class_weights)
    history = trainer.train(epochs=config.epochs)
    
    # Load best and evaluate
    trainer.load_checkpoint(config.model_save_path)
    class_names = list(preprocessor.label_encoder.classes_)
    evaluator = Evaluator(model, config, test_loader, class_names)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    
    # Visualize
    visualizer = Visualizer("./figures")
    visualizer.plot_training_curves(history)
    visualizer.plot_confusion_matrix(results['confusion_matrix'], class_names)
    if results['roc_curves']:
        visualizer.plot_roc_curves(results['roc_curves'])
    
    # Attention visualization
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data[:1].to(config.device)
    with torch.no_grad():
        _, explanation = model(sample_data, return_explanation=True)
    visualizer.plot_attention_heatmap(explanation['attention_weights'])
    
    print("\n" + "=" * 60)
    print("MedDefender-MHAN Complete!")
    print("=" * 60)
    
    return model, results, history


if __name__ == "__main__":
    model, results, history = main()
