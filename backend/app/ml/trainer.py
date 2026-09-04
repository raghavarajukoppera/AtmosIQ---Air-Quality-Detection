from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from app.config import (
    FEATURE_NAMES,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from app.ml.model import GCNRegressor
from app.services.logging_utils import get_logger


@dataclass
class TrainingResult:
    train_samples: int
    val_samples: int
    test_samples: int
    test_mae: float
    test_rmse: float
    epochs_trained: int
    model_path: str


class GCNTrainer:
    def __init__(self) -> None:
        self.logger = get_logger("gcn_trainer")
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    def train(
        self,
        samples: list[dict[str, Any]],
        graphs: dict[str, Any],
        station_df,
        training_summary: dict[str, Any],
        max_epochs: int = 40,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> TrainingResult:
        splits = self._split_samples(samples=samples)
        train_samples = splits["train"]
        val_samples = splits["val"]
        test_samples = splits["test"]

        feature_mean, feature_std = self._fit_feature_scaler(train_samples)

        model = GCNRegressor(in_features=len(FEATURE_NAMES), hidden_features=64, dropout=0.2)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_state = copy.deepcopy(model.state_dict())
        best_val_mae = float("inf")
        best_epoch = 0
        patience = 8
        stale_epochs = 0

        for epoch in range(1, max_epochs + 1):
            random.shuffle(train_samples)
            train_losses = []

            model.train()
            for sample in train_samples:
                city = sample["city"]
                norm_adj = torch.tensor(graphs[city].normalized_adjacency, dtype=torch.float32)
                x = torch.tensor(self._transform_features(sample["X"], feature_mean, feature_std), dtype=torch.float32)
                y = torch.tensor(sample["y"], dtype=torch.float32)

                optimizer.zero_grad()
                pred = model(x, norm_adj)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            val_metrics = self._evaluate(
                model=model,
                samples=val_samples,
                graphs=graphs,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            self.logger.info(
                "Epoch %s | train_mse=%.4f | val_mae=%.4f | val_rmse=%.4f",
                epoch,
                train_loss,
                val_metrics["mae"],
                val_metrics["rmse"],
            )

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    self.logger.info("Early stopping at epoch %s", epoch)
                    break

        model.load_state_dict(best_state)
        test_metrics = self._evaluate(
            model=model,
            samples=test_samples,
            graphs=graphs,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        self.logger.info("Test metrics | MAE=%.4f | RMSE=%.4f", test_metrics["mae"], test_metrics["rmse"])

        artifact = self._build_artifact(
            model=model,
            graphs=graphs,
            station_df=station_df,
            feature_mean=feature_mean,
            feature_std=feature_std,
            metrics=test_metrics,
            training_summary=training_summary,
            best_epoch=best_epoch,
        )
        torch.save(artifact, MODEL_PATH)
        self.logger.info("Saved model artifact: %s", MODEL_PATH)

        return TrainingResult(
            train_samples=len(train_samples),
            val_samples=len(val_samples),
            test_samples=len(test_samples),
            test_mae=float(test_metrics["mae"]),
            test_rmse=float(test_metrics["rmse"]),
            epochs_trained=best_epoch,
            model_path=str(MODEL_PATH),
        )

    def _split_samples(self, samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        timestamps = sorted({sample["timestamp"] for sample in samples})
        n = len(timestamps)
        train_end = int(n * TRAIN_SPLIT)
        val_end = train_end + int(n * VAL_SPLIT)
        if val_end >= n:
            val_end = n - 1

        train_ts = set(timestamps[:train_end])
        val_ts = set(timestamps[train_end:val_end])
        test_ts = set(timestamps[val_end:])

        train_samples = [s for s in samples if s["timestamp"] in train_ts]
        val_samples = [s for s in samples if s["timestamp"] in val_ts]
        test_samples = [s for s in samples if s["timestamp"] in test_ts]

        return {"train": train_samples, "val": val_samples, "test": test_samples}

    @staticmethod
    def _fit_feature_scaler(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        stacked = np.concatenate([sample["X"] for sample in samples], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _transform_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    def _evaluate(
        self,
        model: GCNRegressor,
        samples: list[dict[str, Any]],
        graphs: dict[str, Any],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> dict[str, float]:
        if not samples:
            return {"mae": float("nan"), "rmse": float("nan")}

        model.eval()
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        with torch.no_grad():
            for sample in samples:
                city = sample["city"]
                norm_adj = torch.tensor(graphs[city].normalized_adjacency, dtype=torch.float32)
                x = torch.tensor(self._transform_features(sample["X"], feature_mean, feature_std), dtype=torch.float32)
                y = sample["y"].astype(np.float32)
                pred = model(x, norm_adj).cpu().numpy().astype(np.float32)
                preds.append(pred)
                targets.append(y)

        pred_all = np.concatenate(preds)
        target_all = np.concatenate(targets)
        mae = float(np.mean(np.abs(pred_all - target_all)))
        rmse = float(np.sqrt(np.mean((pred_all - target_all) ** 2)))
        return {"mae": mae, "rmse": rmse}

    def _build_artifact(
        self,
        model: GCNRegressor,
        graphs: dict[str, Any],
        station_df,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        metrics: dict[str, float],
        training_summary: dict[str, Any],
        best_epoch: int,
    ) -> dict[str, Any]:
        stations_by_city: dict[str, list[dict[str, Any]]] = {}
        graph_payload: dict[str, Any] = {}

        for city, graph in graphs.items():
            stations_by_city[city] = (
                station_df[station_df["city"] == city]
                .sort_values("location_id")[["location_id", "station_name", "lat", "lon"]]
                .to_dict(orient="records")
            )
            graph_payload[city] = {
                "adjacency": graph.adjacency.tolist(),
                "normalized_adjacency": graph.normalized_adjacency.tolist(),
            }

        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "model_state_dict": model.state_dict(),
            "graphs": graph_payload,
            "stations_by_city": stations_by_city,
            "metrics": metrics,
            "training_summary": training_summary,
            "best_epoch": best_epoch,
        }
