"""
AI Hospital Operating System
ML Models – LSTM Time-Series Deterioration Predictor
======================================================
Bidirectional LSTM with attention for predicting
clinical deterioration 6-12 hours in advance.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("lstm_model")

VITAL_FEATURES = [
    "heart_rate", "sbp", "dbp", "map",
    "temperature", "spo2", "resp_rate", "gcs_total",
]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Attention Mechanism
# ─────────────────────────────────────────────
if TORCH_AVAILABLE:
    class TemporalAttention(nn.Module):
        """
        Scaled dot-product attention over LSTM hidden states.
        Allows the model to weight earlier time steps differently.
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args
                lstm_output: (batch, seq_len, hidden_dim)
            Returns
                context:    (batch, hidden_dim)   – attended representation
                weights:    (batch, seq_len)       – attention weights
            """
            scores  = self.attention(lstm_output).squeeze(-1)    # (batch, seq_len)
            weights = torch.softmax(scores, dim=-1)               # (batch, seq_len)
            context = torch.bmm(
                weights.unsqueeze(1), lstm_output
            ).squeeze(1)                                           # (batch, hidden_dim)
            return context, weights


    class ClinicalLSTM(nn.Module):
        """
        Bidirectional LSTM with attention for clinical deterioration prediction.

        Architecture:
          Input → BiLSTM layers → Temporal Attention → FC → Sigmoid
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            n_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = True,
        ):
            super().__init__()
            self.hidden_dim    = hidden_dim
            self.n_layers      = n_layers
            self.bidirectional = bidirectional
            self.n_directions  = 2 if bidirectional else 1

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

            lstm_out_dim = hidden_dim * self.n_directions
            self.attention  = TemporalAttention(lstm_out_dim)
            self.layer_norm = nn.LayerNorm(lstm_out_dim)
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(lstm_out_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args
                x: (batch, seq_len, input_dim)
            Returns
                output:  (batch, 1)  – deterioration probability
                weights: (batch, seq_len) – attention weights
            """
            lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden*dirs)
            lstm_out    = self.layer_norm(lstm_out)
            context, weights = self.attention(lstm_out)
            context     = self.dropout(context)
            output      = self.classifier(context)
            return output.squeeze(-1), weights


# ─────────────────────────────────────────────
# Numpy Fallback LSTM-equivalent (SimpleRNN-like)
# ─────────────────────────────────────────────
class SimpleRNNFallback:
    """
    Gradient Boosting-based sequence classifier used when PyTorch
    is not available. Treats flattened window as features.
    """
    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier
        self.clf = GradientBoostingClassifier(n_estimators=100, max_depth=3)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRNNFallback":
        # Flatten sequences: (n, seq, feat) → (n, seq*feat)
        n, s, f = X.shape
        self.clf.fit(X.reshape(n, s * f), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n, s, f = X.shape
        return self.clf.predict_proba(X.reshape(n, s * f))[:, 1]


# ─────────────────────────────────────────────
# LSTM Trainer
# ─────────────────────────────────────────────
class LSTMTrainer:
    """
    Trains the ClinicalLSTM for deterioration prediction.
    Handles class imbalance via pos_weight.
    """

    def __init__(
        self,
        sequence_length: int = 12,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 30,
        threshold: float = 0.5,
    ):
        self.sequence_length = sequence_length
        self.hidden_dim      = hidden_dim
        self.n_layers        = n_layers
        self.dropout         = dropout
        self.lr              = learning_rate
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.threshold       = threshold

        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None
        self.input_dim: int = 0
        self.model: Optional[object] = None
        self.history: Dict[str, List] = {"train_loss":[], "val_loss":[], "val_auc":[]}

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean_ = X.mean(axis=(0, 1), keepdims=True)
            self.std_  = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - self.mean_) / self.std_

    def _get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in VITAL_FEATURES if c in df.columns]
        return df[cols].fillna(df[cols].median())

    def _build_sequences(
        self, df: pd.DataFrame, label_col: str = "label_deteriorating"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build fixed-length sequences per patient."""
        feat_df = self._get_feature_matrix(df)
        X_list, y_list = [], []
        seq = self.sequence_length

        for pid, group in df.groupby("patient_id"):
            feats = feat_df.loc[group.index].values
            labels = (
                group[label_col].values
                if label_col in df.columns
                else np.zeros(len(group))
            )
            for i in range(len(feats) - seq):
                X_list.append(feats[i : i + seq])
                y_list.append(labels[i + seq])

        if not X_list:
            return np.array([]), np.array([])
        return np.stack(X_list).astype(np.float32), np.array(y_list).astype(np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = "label_deteriorating",
        val_df: Optional[pd.DataFrame] = None,
    ) -> "LSTMTrainer":
        X, y = self._build_sequences(df, label_col)
        if len(X) == 0:
            logger.error("No sequences built – check data length vs sequence_length")
            return self

        self.input_dim = X.shape[2]
        X = self._normalize(X, fit=True)

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable – using GBM fallback")
            self.model = SimpleRNNFallback()
            self.model.fit(X, y)
            return self

        # Validation split if no val_df given
        if val_df is None:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y.astype(int), random_state=42
            )
        else:
            X_val, y_val = self._build_sequences(val_df, label_col)
            X_val = self._normalize(X_val)
            X_tr, y_tr = X, y

        logger.info(
            "LSTM training: %d train / %d val sequences, input_dim=%d, seq_len=%d",
            len(X_tr), len(X_val), self.input_dim, self.sequence_length,
        )

        # Build model
        self.model = ClinicalLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Handle class imbalance
        pos_weight = torch.tensor(
            [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
            dtype=torch.float32,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr * 10,
            steps_per_epoch=max(len(X_tr) // self.batch_size, 1),
            epochs=self.epochs,
        )

        # DataLoaders
        tr_dl  = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=self.batch_size, shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=self.batch_size,
        )

        best_auc   = 0.0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # Train
            self.model.train()
            t_loss = 0.0
            for Xb, yb in tr_dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                pred, _ = self.model(Xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                t_loss += loss.item() * len(Xb)
            t_loss /= len(X_tr)

            # Validate
            self.model.eval()
            all_preds, all_labels = [], []
            v_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_dl:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    pred, _ = self.model(Xb)
                    v_loss += criterion(pred, yb).item() * len(Xb)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())
            v_loss /= len(X_val)
            auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["val_auc"].append(auc)

            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0:
                logger.info(
                    "Epoch %3d/%d | train=%.4f | val=%.4f | AUC=%.4f",
                    epoch, self.epochs, t_loss, v_loss, auc,
                )

        if best_state:
            self.model.load_state_dict(best_state)
        logger.info("Best validation AUC: %.4f", best_auc)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of deterioration for each sequence."""
        X, _ = self._build_sequences(df)
        if len(X) == 0:
            return np.array([])
        X = self._normalize(X)

        if not TORCH_AVAILABLE or isinstance(self.model, SimpleRNNFallback):
            return self.model.predict_proba(X)

        self.model.eval()
        X_t = torch.tensor(X).to(self.device)
        with torch.no_grad():
            probs, _ = self.model(X_t)
        return probs.cpu().numpy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with risk scores and binary predictions."""
        probs = self.predict_proba(df)
        results = pd.DataFrame({
            "lstm_risk_score": probs,
            "lstm_prediction": (probs >= self.threshold).astype(int),
        })
        return results

    def evaluate(
        self,
        df: pd.DataFrame,
        label_col: str = "label_deteriorating",
    ) -> Dict:
        X, y_true = self._build_sequences(df, label_col)
        if len(X) == 0 or len(set(y_true)) < 2:
            return {"error": "Insufficient data or single class"}
        X     = self._normalize(X)
        probs = self.predict_proba(df)
        preds = (probs >= self.threshold).astype(int)
        return {
            "roc_auc": roc_auc_score(y_true, probs),
            "f1":      f1_score(y_true, preds, zero_division=0),
            "report":  classification_report(y_true, preds),
        }

    def save(self, path: Path) -> None:
        state = {
            "config": {
                "sequence_length": self.sequence_length,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "threshold": self.threshold,
            },
            "mean_": self.mean_,
            "std_":  self.std_,
            "input_dim": self.input_dim,
            "history": self.history,
        }
        if TORCH_AVAILABLE and hasattr(self.model, "state_dict"):
            state["model_state"] = self.model.state_dict()
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("LSTM model saved: %s", path)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_engineering.load_data import SyntheticDataGenerator

    gen   = SyntheticDataGenerator(seed=0)
    df    = gen.generate_vitals(n_patients=40, hours=48, deterioration_prob=0.3)

    train = df[df["patient_id"] < "SYNTH-P0030"]
    test  = df[df["patient_id"] >= "SYNTH-P0030"]

    trainer = LSTMTrainer(
        sequence_length=6,
        hidden_dim=32,
        n_layers=1,
        epochs=10,
        batch_size=32,
    )
    trainer.fit(train, label_col="label_deteriorating", val_df=test)
    results = trainer.predict(test)
    print("\n=== LSTM Predictions ===")
    print(results.head(20))
    print(f"High-risk sequences: {results['lstm_prediction'].sum()}")
