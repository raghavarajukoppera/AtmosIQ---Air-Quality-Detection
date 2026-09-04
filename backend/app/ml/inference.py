from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from app.config import FEATURE_NAMES, MODEL_PATH
from app.ml.model import GCNRegressor


class GCNInferenceEngine:
    def __init__(self, model_path: Path | str = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        artifact = torch.load(self.model_path, map_location="cpu")
        self.feature_names = artifact["feature_names"]
        self.feature_mean = np.array(artifact["feature_mean"], dtype=np.float32)
        self.feature_std = np.array(artifact["feature_std"], dtype=np.float32)
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std).astype(np.float32)
        self.graphs = artifact["graphs"]
        self.stations_by_city = artifact["stations_by_city"]

        self.model = GCNRegressor(in_features=len(self.feature_names), hidden_features=64, dropout=0.0)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def available_cities(self) -> list[str]:
        return sorted(self.stations_by_city.keys())

    def city_stations(self, city: str) -> list[dict[str, Any]]:
        return list(self.stations_by_city[city])

    def predict_city(self, city: str, feature_frame: pd.DataFrame) -> list[dict[str, Any]]:
        stations = self.stations_by_city[city]
        station_order = [int(station["location_id"]) for station in stations]
        aligned = self._align_city_feature_frame(feature_frame, station_order)

        norm_adj = np.array(self.graphs[city]["normalized_adjacency"], dtype=np.float32)
        pred = self._forward(aligned.to_numpy(dtype=np.float32), norm_adj)

        results: list[dict[str, Any]] = []
        for station, aqi_pred in zip(stations, pred.tolist()):
            clipped = float(np.clip(aqi_pred, 0.0, 500.0))
            results.append(
                {
                    "location_id": int(station["location_id"]),
                    "station_name": station["station_name"],
                    "lat": float(station["lat"]),
                    "lon": float(station["lon"]),
                    "predicted_aqi": clipped,
                }
            )
        return results

    def predict_custom_graph(
        self,
        feature_matrix: np.ndarray,
        normalized_adjacency: np.ndarray,
        stations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pred = self._forward(feature_matrix.astype(np.float32), normalized_adjacency.astype(np.float32))
        output: list[dict[str, Any]] = []
        for station, value in zip(stations, pred.tolist()):
            clipped = float(np.clip(value, 0.0, 500.0))
            output.append(
                {
                    "location_id": int(station["location_id"]),
                    "station_name": station["station_name"],
                    "lat": float(station["lat"]),
                    "lon": float(station["lon"]),
                    "predicted_aqi": clipped,
                }
            )
        return output

    def _forward(self, feature_matrix: np.ndarray, normalized_adjacency: np.ndarray) -> np.ndarray:
        scaled = (feature_matrix - self.feature_mean) / self.feature_std
        with torch.no_grad():
            x = torch.tensor(scaled, dtype=torch.float32)
            a = torch.tensor(normalized_adjacency, dtype=torch.float32)
            pred = self.model(x, a).cpu().numpy()
        return pred

    def _align_city_feature_frame(self, feature_frame: pd.DataFrame, station_order: list[int]) -> pd.DataFrame:
        frame = feature_frame.copy()
        required = ["location_id", *self.feature_names]
        missing_columns = [column for column in required if column not in frame.columns]
        for column in missing_columns:
            if column == "location_id":
                frame[column] = np.nan
            else:
                frame[column] = np.nan

        frame = frame[required].copy()
        frame["location_id"] = pd.to_numeric(frame["location_id"], errors="coerce")
        frame = frame.set_index("location_id").reindex(station_order)

        for idx, feature in enumerate(self.feature_names):
            fallback = float(self.feature_mean[idx])
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            city_median = frame[feature].median()
            fill_value = fallback if np.isnan(city_median) else float(city_median)
            frame[feature] = frame[feature].fillna(fill_value)

        frame = frame.reset_index(drop=True)
        return frame
