"""Autoencoder anomaly detector: flags points that reconstruct poorly.

Trained only on (approximately) normal data. At inference, reconstruction
error is the anomaly signal — a pump operating outside the patterns seen
in training will reconstruct badly regardless of whether any single
sensor crosses a fixed threshold, which is what lets it catch subtle
multivariate drift that Isolation Forest's axis-aligned splits can miss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import AutoencoderConfig


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 16]

        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        # Drop the trailing ReLU on the bottleneck so it isn't clipped to >=0.
        self.encoder = nn.Sequential(*encoder_layers[:-1])

        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        for h in reversed_dims:
            decoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        # Drop the trailing ReLU on the output layer — reconstructed sensor
        # values (after standardization) can legitimately be negative.
        self.decoder = nn.Sequential(*decoder_layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderTrainer:
    """Handles the training loop, early stopping, and scoring for
    ``Autoencoder`` so ``Autoencoder`` itself stays a plain nn.Module."""

    def __init__(self, config: AutoencoderConfig, input_dim: int, device: str | None = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim, config.hidden_dims).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
    ) -> list[float]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_train_tensor, X_train_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        history: list[float] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for _epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if X_val is not None:
                val_loss = float(np.mean(self.reconstruction_error(X_val)))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        return errors.cpu().numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (raw reconstruction MSE)."""
        return self.reconstruction_error(X)

    def save(self, path: str | Path) -> None:
        torch.save(
            {"state_dict": self.model.state_dict(), "hidden_dims": self.config.hidden_dims},
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, config: AutoencoderConfig, input_dim: int, device: str | None = None
    ) -> AutoencoderTrainer:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(config, input_dim, device)
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.eval()
        return instance
