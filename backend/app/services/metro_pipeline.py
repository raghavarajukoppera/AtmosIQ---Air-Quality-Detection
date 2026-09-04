from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_NAMES,
    FEATURE_TABLE_PATH,
    GRAPH_K_NEIGHBORS,
    GRAPH_META_PATH,
    METRO_CITY_NAMES,
    RAW_MONTHLY_PATH,
    REQUIRED_POLLUTANTS,
    TRAINING_MONTH,
    TRAINING_SUMMARY_PATH,
    TRAINING_YEAR,
)
from app.data_sources.open_meteo import OpenMeteoClient
from app.data_sources.openaq_archive import OpenAQArchiveClient
from app.ml.aqi import compute_aqi_from_values, convert_to_standard
from app.ml.graph import GraphInfo, build_city_graph
from app.services.logging_utils import get_logger


@dataclass
class PreparedTrainingData:
    stations: list[dict]
    station_df: pd.DataFrame
    feature_table: pd.DataFrame
    graphs: dict[str, GraphInfo]
    samples: list[dict[str, Any]]
    summary: dict[str, Any]


class MetroPipeline:
    def __init__(
        self,
        archive_client: OpenAQArchiveClient | None = None,
        weather_client: OpenMeteoClient | None = None,
    ) -> None:
        self.archive = archive_client or OpenAQArchiveClient()
        self.weather = weather_client or OpenMeteoClient()
        self.logger = get_logger("metro_pipeline")

    def prepare_training_data(self, force_refresh: bool = False) -> PreparedTrainingData:
        self.logger.info("Preparing metro training data from OpenAQ archive")

        stations = asyncio.run(
            self.archive.discover_metro_stations(
                year=TRAINING_YEAR,
                month=TRAINING_MONTH,
                force_refresh=force_refresh,
            )
        )
        station_df = pd.DataFrame(stations).sort_values(["city", "location_id"]).reset_index(drop=True)

        if RAW_MONTHLY_PATH.exists() and not force_refresh:
            raw_df = pd.read_pickle(RAW_MONTHLY_PATH)
            self.logger.info("Loaded cached raw monthly rows: %s", len(raw_df))
        else:
            raw_df = asyncio.run(self.archive.download_monthly_rows(stations=stations, year=TRAINING_YEAR, month=TRAINING_MONTH))
            raw_df.to_pickle(RAW_MONTHLY_PATH)
            self.logger.info("Downloaded and cached raw monthly rows: %s", len(raw_df))

        if FEATURE_TABLE_PATH.exists() and not force_refresh:
            feature_table = pd.read_pickle(FEATURE_TABLE_PATH)
            self.logger.info("Loaded cached feature table: %s", len(feature_table))
        else:
            feature_table = self._build_feature_table(raw_df=raw_df, station_df=station_df)
            feature_table.to_pickle(FEATURE_TABLE_PATH)
            self.logger.info("Built and cached feature table: %s", len(feature_table))

        graphs = self._build_graphs(station_df=station_df)
        samples = self._build_samples(feature_table=feature_table, station_df=station_df)
        summary = self._build_summary(raw_df=raw_df, station_df=station_df, feature_table=feature_table, graphs=graphs, samples=samples)
        TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._save_graph_meta(graphs=graphs, station_df=station_df)

        return PreparedTrainingData(
            stations=stations,
            station_df=station_df,
            feature_table=feature_table,
            graphs=graphs,
            samples=samples,
            summary=summary,
        )

    def _build_feature_table(self, raw_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df["parameter"] = df["parameter"].str.lower()
        df = df[df["parameter"].isin(REQUIRED_POLLUTANTS)].copy()
        df["hour"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.floor("h")
        df = df.dropna(subset=["hour", "value"])

        df["standard_value"] = df.apply(
            lambda row: convert_to_standard(
                parameter=str(row["parameter"]),
                value=float(row["value"]),
                unit=str(row.get("units", "")),
            ),
            axis=1,
        )

        group_cols = ["city", "location_id", "station_name", "lat", "lon", "hour", "parameter"]
        hourly = df.groupby(group_cols, as_index=False)["standard_value"].mean()
        pivot = (
            hourly.pivot_table(
                index=["city", "location_id", "station_name", "lat", "lon", "hour"],
                columns="parameter",
                values="standard_value",
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        weather_df = self._build_weather_table(station_df=station_df, pivot=pivot)
        merged = pivot.merge(weather_df, on=["location_id", "hour"], how="left")

        complete = self._build_complete_hourly_grid(merged=merged, station_df=station_df)
        return complete

    def _build_weather_table(self, station_df: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
        start_date = pivot["hour"].min().date().isoformat()
        end_date = pivot["hour"].max().date().isoformat()
        weather_frames: list[pd.DataFrame] = []

        for station in station_df.itertuples(index=False):
            weather = self.weather.fetch_historical_hourly(
                latitude=float(station.lat),
                longitude=float(station.lon),
                start_date=start_date,
                end_date=end_date,
            )
            weather = weather.rename(columns={"timestamp": "hour"})
            weather["location_id"] = int(station.location_id)
            weather_frames.append(weather)

        weather_df = pd.concat(weather_frames, ignore_index=True)
        weather_df["hour"] = pd.to_datetime(weather_df["hour"], utc=True, errors="coerce")
        return weather_df

    def _build_complete_hourly_grid(self, merged: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
        city_frames: list[pd.DataFrame] = []

        for city in METRO_CITY_NAMES:
            city_stations = station_df[station_df["city"] == city].copy().sort_values("location_id")
            city_data = merged[merged["city"] == city].copy()
            if city_data.empty:
                continue

            start_hour = city_data["hour"].min().floor("d")
            end_hour = city_data["hour"].max().ceil("d") - pd.Timedelta(hours=1)
            hours = pd.date_range(start=start_hour, end=end_hour, freq="h", tz="UTC")

            grid = pd.MultiIndex.from_product(
                [city_stations["location_id"].tolist(), hours],
                names=["location_id", "hour"],
            ).to_frame(index=False)
            grid["city"] = city

            station_cols = city_stations[["location_id", "station_name", "lat", "lon"]]
            grid = grid.merge(station_cols, on="location_id", how="left")
            grid = grid.merge(
                city_data[
                    [
                        "city",
                        "location_id",
                        "hour",
                        "pm25",
                        "pm10",
                        "no2",
                        "so2",
                        "co",
                        "o3",
                        "temperature_2m",
                        "relative_humidity_2m",
                        "wind_speed_10m",
                    ]
                ],
                on=["city", "location_id", "hour"],
                how="left",
            )
            grid = grid.sort_values(["location_id", "hour"]).reset_index(drop=True)

            for column in FEATURE_NAMES:
                grid[column] = grid.groupby("location_id")[column].transform(lambda series: series.ffill().bfill())
                city_median = float(grid[column].median()) if not np.isnan(grid[column].median()) else 0.0
                grid[column] = grid[column].fillna(city_median)

            grid["aqi"] = grid.apply(
                lambda row: compute_aqi_from_values(
                    {
                        "pm25": row["pm25"],
                        "pm10": row["pm10"],
                        "no2": row["no2"],
                        "so2": row["so2"],
                        "co": row["co"],
                        "o3": row["o3"],
                    }
                ),
                axis=1,
            )
            grid["aqi"] = pd.to_numeric(grid["aqi"], errors="coerce").ffill().bfill().fillna(0.0)
            city_frames.append(grid)

        full = pd.concat(city_frames, ignore_index=True)
        full = full.sort_values(["city", "hour", "location_id"]).reset_index(drop=True)
        return full

    def _build_graphs(self, station_df: pd.DataFrame) -> dict[str, GraphInfo]:
        graphs: dict[str, GraphInfo] = {}
        for city in METRO_CITY_NAMES:
            city_stations = station_df[station_df["city"] == city].sort_values("location_id")
            coords = city_stations[["lat", "lon"]].to_numpy(dtype=np.float32)
            graphs[city] = build_city_graph(city=city, coords=coords, k_neighbors=GRAPH_K_NEIGHBORS)
        return graphs

    def _build_samples(self, feature_table: pd.DataFrame, station_df: pd.DataFrame) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        for city in METRO_CITY_NAMES:
            city_data = feature_table[feature_table["city"] == city].copy()
            city_data["hour"] = pd.to_datetime(city_data["hour"], utc=True, errors="coerce")
            station_order = (
                station_df[station_df["city"] == city]
                .sort_values("location_id")["location_id"]
                .astype(int)
                .tolist()
            )

            for hour, hour_df in city_data.groupby("hour"):
                hour_df = hour_df.set_index("location_id").reindex(station_order)
                x = hour_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
                y = hour_df["aqi"].to_numpy(dtype=np.float32)
                samples.append(
                    {
                        "city": city,
                        "timestamp": pd.Timestamp(hour).isoformat(),
                        "X": x,
                        "y": y,
                    }
                )

        samples.sort(key=lambda sample: sample["timestamp"])
        return samples

    def _build_summary(
        self,
        raw_df: pd.DataFrame,
        station_df: pd.DataFrame,
        feature_table: pd.DataFrame,
        graphs: dict[str, GraphInfo],
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        date_min = pd.to_datetime(raw_df["datetime"], utc=True, errors="coerce").min()
        date_max = pd.to_datetime(raw_df["datetime"], utc=True, errors="coerce").max()

        graph_stats = {
            city: {
                "nodes": info.node_count,
                "edges": info.edge_count,
                "components": info.connected_components,
            }
            for city, info in graphs.items()
        }

        summary = {
            "raw_rows": int(len(raw_df)),
            "feature_rows": int(len(feature_table)),
            "stations": int(station_df["location_id"].nunique()),
            "cities": sorted(station_df["city"].unique().tolist()),
            "date_range_utc": {
                "start": date_min.isoformat() if pd.notna(date_min) else None,
                "end": date_max.isoformat() if pd.notna(date_max) else None,
            },
            "graph_stats": graph_stats,
            "sample_count": len(samples),
        }
        return summary

    def _save_graph_meta(self, graphs: dict[str, GraphInfo], station_df: pd.DataFrame) -> None:
        payload: dict[str, Any] = {"graphs": {}, "stations_by_city": {}}
        for city, info in graphs.items():
            payload["graphs"][city] = {
                "adjacency": info.adjacency.tolist(),
                "normalized_adjacency": info.normalized_adjacency.tolist(),
                "node_count": info.node_count,
                "edge_count": info.edge_count,
                "components": info.connected_components,
            }
            payload["stations_by_city"][city] = (
                station_df[station_df["city"] == city]
                .sort_values("location_id")[["location_id", "station_name", "lat", "lon"]]
                .to_dict(orient="records")
            )
        GRAPH_META_PATH.write_text(json.dumps(payload), encoding="utf-8")
