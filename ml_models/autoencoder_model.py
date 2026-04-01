"""
AI Hospital Operating System
ML Models – Autoencoder for Anomaly Detection
===============================================
Deep autoencoder trained on normal ICU physiology.
High reconstruction error = anomaly (deterioration).
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# PyTorch-based implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("autoencoder_model")

VITAL_FEATURES = [
    "heart_rate", "sbp", "dbp", "map",
    "temperature", "spo2", "resp_rate", "gcs_total",
]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# PyTorch Autoencoder Architecture
# ─────────────────────────────────────────────
if TORCH_AVAILABLE:
    class VitalsEncoder(nn.Module):
        """
        Encoder: input_dim → latent_dim
        Uses skip connections for gradient stability.
        """
        def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.2):
            super().__init__()
            hidden1 = max(input_dim * 2, 32)
            hidden2 = max(input_dim,     16)

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(hidden2, latent_dim),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)


    class VitalsDecoder(nn.Module):
        """Decoder: latent_dim → input_dim (reconstruction)"""
        def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.2):
            super().__init__()
            hidden1 = max(input_dim,     16)
            hidden2 = max(input_dim * 2, 32)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(hidden2, input_dim),
                # No activation – raw reconstruction
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)


    class VitalsAutoencoder(nn.Module):
        """Full Autoencoder: encode then decode."""
        def __init__(self, input_dim: int, latent_dim: int = 4, dropout: float = 0.2):
            super().__init__()
            self.encoder = VitalsEncoder(input_dim, latent_dim, dropout)
            self.decoder = VitalsDecoder(input_dim, latent_dim, dropout)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            z    = self.encoder(x)
            recon = self.decoder(z)
            return recon, z


# ─────────────────────────────────────────────
# NumPy Fallback Autoencoder (no torch)
# ─────────────────────────────────────────────
class SimpleNumpyAutoencoder:
    """
    Lightweight autoencoder using SVD-based dimensionality reduction.
    Used as fallback when PyTorch is not installed.
    """
    def __init__(self, latent_dim: int = 4):
        self.latent_dim = latent_dim
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray) -> "SimpleNumpyAutoencoder":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_) / self.std_

        # PCA via SVD
        U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
        self.components_ = Vt[:self.latent_dim]

        # Compute reconstruction errors on training data
        X_recon = self._reconstruct(X_norm)
        errors  = np.mean((X_norm - X_recon) ** 2, axis=1)
        self.threshold_ = np.percentile(errors, 95)
        return self

    def _reconstruct(self, X_norm: np.ndarray) -> np.ndarray:
        Z     = X_norm @ self.components_.T
        X_hat = Z @ self.components_
        return X_hat

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self.mean_) / self.std_
        X_hat  = self._reconstruct(X_norm)
        return np.mean((X_norm - X_hat) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self.reconstruction_error(X)
        return (errors > self.threshold_).astype(int)


# ─────────────────────────────────────────────
# AutoencoderTrainer (PyTorch)
# ─────────────────────────────────────────────
class AutoencoderTrainer:
    """
    Trains and evaluates the deep Autoencoder for anomaly detection.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        device: Optional[str] = None,
        anomaly_percentile: float = 95,
    ):
        self.latent_dim         = latent_dim
        self.lr                 = learning_rate
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.anomaly_percentile = anomaly_percentile
        self.threshold_: float  = 0.0
        self.input_dim: int     = 0

        # Feature scaling params
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

        if device:
            self.device = torch.device(device)
        elif TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[VitalsAutoencoder] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_  = X.std(axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in VITAL_FEATURES if c in df.columns]
        return df[cols].fillna(df[cols].median()).values.astype(np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "AutoencoderTrainer":
        X = self._get_features(df)
        X = self._preprocess(X, fit=True)
        self.input_dim = X.shape[1]

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available – using NumPy fallback autoencoder")
            self._fallback = SimpleNumpyAutoencoder(latent_dim=self.latent_dim)
            self._fallback.fit(X)
            self.threshold_ = self._fallback.threshold_
            return self

        # Build model
        self.model = VitalsAutoencoder(
            input_dim=self.input_dim, latent_dim=self.latent_dim
        ).to(self.device)

        optimizer  = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion  = nn.MSELoss()

        # Train DataLoader
        X_tensor   = torch.tensor(X)
        train_ds   = TensorDataset(X_tensor)
        train_dl   = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Val DataLoader
        if val_df is not None:
            Xv     = self._preprocess(self._get_features(val_df))
            Xv_t   = torch.tensor(Xv)
            val_ds = TensorDataset(Xv_t)
            val_dl = DataLoader(val_ds, batch_size=self.batch_size)
        else:
            val_dl = None

        logger.info(
            "Training Autoencoder: %d samples, input_dim=%d, latent=%d, epochs=%d",
            len(X), self.input_dim, self.latent_dim, self.epochs,
        )

        for epoch in range(1, self.epochs + 1):
            # ── Training ──
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_dl:
                batch = batch.to(self.device)
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(train_ds)
            self.history["train_loss"].append(train_loss)

            # ── Validation ──
            if val_dl:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for (batch,) in val_dl:
                        batch  = batch.to(self.device)
                        recon, _ = self.model(batch)
                        val_loss += criterion(recon, batch).item() * len(batch)
                val_loss /= (len(Xv_t))  # type: ignore
                self.history["val_loss"].append(val_loss)
                scheduler.step(val_loss)
                if epoch % 10 == 0:
                    logger.info(
                        "Epoch %3d/%d | train=%.5f | val=%.5f",
                        epoch, self.epochs, train_loss, val_loss,
                    )
            else:
                if epoch % 10 == 0:
                    logger.info("Epoch %3d/%d | train=%.5f", epoch, self.epochs, train_loss)

        # Compute threshold from training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            X_t  = torch.tensor(X).to(self.device)
            recon, _ = self.model(X_t)
            errors = ((X_t - recon) ** 2).mean(dim=1).cpu().numpy()
        self.threshold_ = float(np.percentile(errors, self.anomaly_percentile))
        logger.info("Reconstruction error threshold (p%.0f): %.6f",
                    self.anomaly_percentile, self.threshold_)
        return self

    def reconstruction_error(self, df: pd.DataFrame) -> np.ndarray:
        """Compute per-row reconstruction error."""
        X = self._preprocess(self._get_features(df))
        if not TORCH_AVAILABLE:
            return self._fallback.reconstruction_error(X)

        self.model.eval()
        X_t = torch.tensor(X).to(self.device)
        with torch.no_grad():
            recon, _ = self.model(X_t)
        return ((X_t - recon) ** 2).mean(dim=1).cpu().numpy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns df with:
          recon_error  : float
          ae_anomaly   : 0/1
          ae_score     : normalised [0,1]
        """
        df = df.copy()
        errors = self.reconstruction_error(df)
        df["recon_error"] = errors
        df["ae_anomaly"]  = (errors > self.threshold_).astype(int)
        # Normalise to [0,1]
        e_max = max(errors.max(), self.threshold_, 1e-8)
        df["ae_score"] = np.clip(errors / e_max, 0, 1)
        return df

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "input_dim":    self.input_dim,
            "latent_dim":   self.latent_dim,
            "threshold_":   self.threshold_,
            "mean_":        self.mean_,
            "std_":         self.std_,
            "history":      self.history,
        }
        if TORCH_AVAILABLE and self.model:
            state["model_state"] = self.model.state_dict()
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Autoencoder saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderTrainer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        trainer = cls(
            latent_dim=state["latent_dim"],
        )
        trainer.input_dim  = state["input_dim"]
        trainer.threshold_ = state["threshold_"]
        trainer.mean_      = state["mean_"]
        trainer.std_       = state["std_"]
        trainer.history    = state.get("history", {})
        if TORCH_AVAILABLE and "model_state" in state:
            trainer.model = VitalsAutoencoder(
                input_dim=state["input_dim"], latent_dim=state["latent_dim"]
            ).to(trainer.device)
            trainer.model.load_state_dict(state["model_state"])
            trainer.model.eval()
        logger.info("Autoencoder loaded: %s", path)
        return trainer


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_engineering.load_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    df  = gen.generate_vitals(n_patients=50, hours=48, deterioration_prob=0.2)

    train = df[df["patient_id"] < "SYNTH-P0040"]
    test  = df[df["patient_id"] >= "SYNTH-P0040"]

    trainer = AutoencoderTrainer(latent_dim=4, epochs=20, batch_size=32)
    trainer.fit(train, val_df=test)

    results = trainer.predict(test)
    print("\n=== Autoencoder Anomaly Detection ===")
    print(results[["patient_id","recon_error","ae_anomaly","ae_score"]].describe())
    print(f"\nAnomalies flagged: {results['ae_anomaly'].sum()} / {len(results)}")
