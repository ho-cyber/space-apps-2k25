#!/usr/bin/env python3
"""
improved_train_model.py

Enhanced training with proper labeling and transit detection focus.

Usage:
    # Train with pre-downloaded local data (RECOMMENDED)
    python improved_train_model.py --use-local-curated --n-synthetic 450 --epochs 15
    
    # Train by fetching online
    python improved_train_model.py --fetch-online --n-online 60 --n-synthetic 400 --epochs 15
    
    # Train with synthetic only
    python improved_train_model.py --n-synthetic 500 --epochs 12
"""
import logging
import argparse
import random
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from scipy.signal import savgol_filter
except:
    savgol_filter = None

try:
    import lightkurve as lk
except:
    lk = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================
# CONFIG
# ==========================================
SEED = 42
SEQ_LEN = 2000
BATCH_SIZE = 32
EPOCHS = 15
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# CURATED STAR LISTS (only used by download script)
# ==========================================
# These lists are defined here for reference but not used in training
# Training simply loads whatever .npy files exist in curated_data/

# ==========================================
# PREPROCESSING
# ==========================================
def detrend_savgol(flux, window_length=101, polyorder=3):
    """Detrend using Savitzky-Golay filter."""
    if savgol_filter is None:
        return flux / (np.median(flux) + 1e-12)
    
    wl = min(int(window_length), len(flux) - 1)
    if wl < 5:
        return flux
    if wl % 2 == 0:
        wl -= 1
    
    try:
        trend = savgol_filter(flux, wl, min(polyorder, wl-1))
        trend[trend == 0] = np.median(trend[trend != 0])
        return flux / (trend + 1e-12)
    except:
        return flux / (np.median(flux) + 1e-12)

def preprocess_flux_array(flux, target_len=SEQ_LEN):
    """Enhanced preprocessing with better normalization."""
    try:
        flux = np.asarray(flux, dtype=np.float32)
    except:
        return None
    
    # Remove NaNs/Infs
    mask = np.isfinite(flux)
    if mask.sum() < max(20, len(flux) // 5):
        return None
    flux = flux[mask]
    
    # Sigma clipping for outliers
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad == 0:
        return None
    
    sigma_clip = 5.0
    mask = np.abs(flux - median) < sigma_clip * mad * 1.4826
    if mask.sum() < 20:
        return None
    flux = flux[mask]
    
    # Detrend
    flux_dt = detrend_savgol(flux, window_length=201, polyorder=2)
    
    # Remove remaining NaNs
    flux_dt = np.nan_to_num(flux_dt, nan=0.0)
    
    # Resample to target length
    if len(flux_dt) < 10:
        return None
    
    x_old = np.linspace(0, 1, len(flux_dt))
    x_new = np.linspace(0, 1, target_len)
    resampled = np.interp(x_new, x_old, flux_dt).astype(np.float32)
    
    # Robust standardization
    median = np.median(resampled)
    mad = np.median(np.abs(resampled - median))
    if mad == 0:
        std = np.std(resampled)
        if std == 0:
            return None
        resampled = (resampled - median) / (std + 1e-12)
    else:
        resampled = (resampled - median) / (1.4826 * mad + 1e-12)
    
    return resampled

# ==========================================
# SYNTHETIC DATA GENERATION
# ==========================================
def generate_realistic_transit(t, depth, duration, period, phase=0.0):
    """Generate a more realistic transit shape with ingress/egress."""
    transit_signal = np.zeros_like(t)
    
    # Calculate transit times
    t_normalized = (t - phase) % period
    
    ingress_duration = duration * 0.15  # 15% of duration for ingress
    egress_duration = duration * 0.15
    full_duration = duration * 0.7  # 70% at full depth
    
    transit_start = period / 2 - duration / 2
    transit_end = period / 2 + duration / 2
    
    for i, time in enumerate(t_normalized):
        if transit_start <= time <= transit_end:
            relative_time = time - transit_start
            
            # Ingress
            if relative_time < ingress_duration:
                fraction = relative_time / ingress_duration
                transit_signal[i] = -depth * fraction
            # Full depth
            elif relative_time < (ingress_duration + full_duration):
                transit_signal[i] = -depth
            # Egress
            elif relative_time < duration:
                fraction = (duration - relative_time) / egress_duration
                transit_signal[i] = -depth * fraction
    
    return transit_signal

def generate_synthetic_dataset(n_samples=200, seq_len=SEQ_LEN, positive_fraction=0.5):
    """Generate improved synthetic light curves with stellar variability including Tabby-like dips."""
    features = []
    labels = []
    
    for _ in range(n_samples):
        t = np.linspace(0, 10, seq_len)  # 10 arbitrary time units
        flux = np.zeros(seq_len)
        
        # Realistic stellar variability (multiple sinusoidal components)
        n_modes = np.random.randint(2, 5)
        for _ in range(n_modes):
            freq = np.random.uniform(0.5, 3.0)
            amp = np.random.uniform(0.0003, 0.0015)
            phase = np.random.uniform(0, 2*np.pi)
            flux += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Red noise (correlated)
        red_noise = np.cumsum(np.random.randn(seq_len)) * 0.0001
        red_noise = red_noise - np.mean(red_noise)
        flux += red_noise
        
        # White noise
        flux += np.random.normal(0, 0.0005, seq_len)
        
        # Rare outliers (cosmic rays)
        if np.random.rand() < 0.3:
            n_outliers = np.random.randint(1, 4)
            indices = np.random.choice(seq_len, n_outliers, replace=False)
            flux[indices] += np.random.uniform(-0.005, 0.005, n_outliers)

        # ----- Tabby-like variability -----
        if np.random.rand() < 0.2:  # 20% chance to add a major dip
            n_dips = np.random.randint(1, 3)
            for _ in range(n_dips):
                dip_center = np.random.uniform(1, 9)  # avoid edges
                dip_duration = np.random.uniform(0.1, 0.6)
                dip_depth = np.random.uniform(0.005, 0.03)
                
                ingress_ratio = np.random.uniform(0.2, 0.6)
                egress_ratio = 1.0 - ingress_ratio
                
                for i, time in enumerate(t):
                    dt = time - dip_center
                    if -dip_duration/2 <= dt <= dip_duration/2:
                        if dt < 0:
                            flux[i] -= dip_depth * (0.5 + 0.5*dt/(ingress_ratio*dip_duration/2))
                        else:
                            flux[i] -= dip_depth * (0.5 - 0.5*dt/(egress_ratio*dip_duration/2))
            
            # Optional slow long-term trend
            trend_amp = np.random.uniform(-0.002, 0.002)
            flux += trend_amp * (t - t.mean()) / (t.max() - t.min())

        # Add transit if positive class
        has_transit = np.random.rand() < positive_fraction
        if has_transit:
            depth = np.random.uniform(0.003, 0.025)
            duration = np.random.uniform(0.1, 0.4)
            period = np.random.uniform(2.0, 8.0)
            phase = np.random.uniform(0, period)
            transit = generate_realistic_transit(t, depth, duration, period, phase)
            flux += transit
            
            if np.random.rand() < 0.2:  # multi-planet system
                depth2 = np.random.uniform(0.002, 0.015)
                duration2 = np.random.uniform(0.08, 0.35)
                period2 = np.random.uniform(3.0, 9.0)
                phase2 = np.random.uniform(0, period2)
                transit2 = generate_realistic_transit(t, depth2, duration2, period2, phase2)
                flux += transit2
            
            labels.append(1)
        else:
            labels.append(0)
        
        # Normalize
        flux = (flux - np.median(flux)) / (np.std(flux) + 1e-12)
        features.append(flux.astype(np.float32))
    
    return features, labels

# ==========================================
# DATA LOADING
# ==========================================
# ==========================================
# DATA LOADING
# ==========================================
def load_curated_local_data(data_dir="../curated_data"):
    """Load pre-downloaded lightcurves from local directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logging.warning(f"Curated data directory not found: {data_dir}")
        return [], []
    
    features, labels = [], []
    
    npy_files = sorted(data_dir.glob("*.npy"))
    if len(npy_files) == 0:
        logging.warning(f"No .npy files found in {data_dir}")
        return [], []
    
    for npy_file in tqdm(npy_files, desc="Loading local curated data"):
        try:
            # Parse label from filename (format: targetname_label_X.npy)
            label = int(npy_file.stem.split("_label_")[-1])
            
            # Load flux (handle any array type)
            flux = np.load(npy_file, allow_pickle=False)
            flux = np.asarray(flux, dtype=np.float32)
            
            processed = preprocess_flux_array(flux, target_len=SEQ_LEN)
            
            if processed is not None:
                features.append(processed)
                labels.append(label)
                logging.info(f"Loaded {npy_file.name} (label={label})")
        
        except Exception as e:
            logging.warning(f"Failed to load {npy_file}: {e}")
            continue
    
    logging.info(f"Successfully loaded {len(features)} samples from {len(npy_files)} files")
    return features, labels

# ==========================================
# DATASET & MODEL
# ==========================================
class LightCurveTorchDataset(Dataset):
    def __init__(self, features, labels):
        self.X = np.array(features, dtype=np.float32)
        self.y = np.array(labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(int(self.y[idx]), dtype=torch.long)

class ImprovedTimeSeriesTransformer(nn.Module):
    """Enhanced transformer with better regularization."""
    def __init__(self, seq_len=SEQ_LEN, d_model=64, nhead=4, num_layers=3, num_classes=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
    
    def forward(self, x):
        b, l = x.shape
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_emb[:, :l, :].to(x.device)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.cls_head(x)

# ==========================================
# TRAINING
# ==========================================
def train_loop(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, checkpoint_path=None):
    model = model.to(DEVICE)
    
    # Calculate class weights for imbalanced dataset
    train_labels = train_loader.dataset.y
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(len(train_labels) / (len(class_counts) * class_counts), 
                                 dtype=torch.float32).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = 0.0
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds)
                y_prob.extend(probs[:,1])
        
        val_loss /= len(val_loader.dataset)
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        
        print(f"[Epoch {epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Save best model based on F1 score
        if checkpoint_path and f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_f1": f1,
                "val_auc": auc
            }, checkpoint_path)
            print(f"Saved checkpoint (F1={f1:.4f})")
    
    return model

# ==========================================
# MAIN
# ==========================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-local-curated", action="store_true", help="Use pre-downloaded curated data (recommended)")
    p.add_argument("--n-synthetic", type=int, default=400, help="Number of synthetic samples")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--checkpoint", type=str, default="checkpoints/improved_model.pt")
    args = p.parse_args()
    
    print(f"Device: {DEVICE}")
    
    all_features, all_labels = [], []
    
    # Load from pre-downloaded local data
    if args.use_local_curated:
        feats, labs = load_curated_local_data("curated_data")
        if len(feats) == 0:
            logging.warning("No local data found, will rely on synthetic data only")
        else:
            all_features.extend(feats)
            all_labels.extend(labs)
    
    # Add synthetic data
    if args.n_synthetic > 0:
        logging.info(f"Generating {args.n_synthetic} synthetic samples...")
        feats_syn, labs_syn = generate_synthetic_dataset(
            n_samples=args.n_synthetic,
            seq_len=SEQ_LEN,
            positive_fraction=0.5
        )
        all_features.extend(feats_syn)
        all_labels.extend(labs_syn)
    
    if len(all_features) == 0:
        raise ValueError("No training data available! Run download_curated_data.py first or use --n-synthetic")
    
    label_counts = Counter(all_labels)
    print(f"\nDataset composition: {label_counts}")
    print(f"Total samples: {len(all_features)}\n")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, all_labels, test_size=0.2, stratify=all_labels, random_state=SEED
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}\n")
    
    train_ds = LightCurveTorchDataset(X_train, y_train)
    val_ds = LightCurveTorchDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Train
    model = ImprovedTimeSeriesTransformer(seq_len=SEQ_LEN, num_layers=3, dropout=0.2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    trained = train_loop(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, 
                        checkpoint_path=args.checkpoint)
    
    # Final evaluation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in DataLoader(val_ds, batch_size=64):
            xb = xb.to(DEVICE)
            out = trained(xb)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(probs.argmax(axis=1))
            y_prob.extend(probs[:,1])
    
    print("\n" + "="*70)
    print("FINAL VALIDATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=["No Transit", "Transit"]))
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("="*70)

if __name__ == "__main__":
    main()