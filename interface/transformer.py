#!/usr/bin/env python3
"""
main.py

Contains:
- Transformer model definition
- Flux preprocessing for fixed-length input
- Prediction helper
- Constants like SEQ_LEN and DEVICE
"""

import torch
import torch.nn as nn
import numpy as np
from astropy.io import fits

# ------------------------
# Device / Constants
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 2000  # Must match model max_len

# ------------------------
# Transformer Model
# ------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_heads=4, num_layers=2,
                 dim_feedforward=256, num_classes=2, max_len=SEQ_LEN):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return self.cls_head(x)

# ------------------------
# Preprocessing Helpers
# ------------------------
def preprocess_flux_array(flux_array, target_len=SEQ_LEN):
    """
    Resample and normalize flux array to fixed length for transformer input.
    Handles np.nan safely.
    """
    try:
        flux_array = np.asarray(flux_array, dtype=np.float32)
    except Exception:
        return None

    flux_array = flux_array[np.isfinite(flux_array)]
    if len(flux_array) < 10:
        return None

    # normalize
    flux_array = flux_array - np.median(flux_array)
    flux_array = flux_array / (np.std(flux_array) + 1e-12)

    # interpolate to fixed length
    x_old = np.linspace(0, 1, len(flux_array))
    x_new = np.linspace(0, 1, target_len)
    flux_resampled = np.interp(x_new, x_old, flux_array).astype(np.float32)
    return flux_resampled

# ------------------------
# Prediction Helper
# ------------------------
def predict_flux(flux_array, model, target_len=SEQ_LEN):
    """
    Preprocess and predict using transformer model.
    Returns: (pred_class, probabilities)
    """
    processed = preprocess_flux_array(flux_array, target_len=target_len)
    if processed is None:
        return None, None

    x = torch.tensor(processed, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs

# ------------------------
# FITS Loader (optional)
# ------------------------
def load_flux_from_fits(fits_path):
    """
    Extract flux from FITS file (PDCSAP_FLUX > SAP_FLUX)
    """
    try:
        hdul = fits.open(fits_path)
        flux = hdul[1].data.get("PDCSAP_FLUX") or hdul[1].data.get("SAP_FLUX")
        if flux is None:
            return None
        return np.array(flux, dtype=np.float32)
    except Exception:
        return None
