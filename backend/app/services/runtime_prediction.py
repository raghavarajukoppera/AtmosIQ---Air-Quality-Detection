from __future__ import annotations

import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.config import FEATURE_NAMES, METRO_CITY_CENTERS, REQUIRED_POLLUTANTS, RUNTIME_CITY_CACHE_PATH
from app.data_sources.open_meteo import OpenMeteoClient
from app.data_sources.openaq_live import OpenAQLiveClient
from app.data_sources.openaq_archive import OpenAQArchiveClient
from app.ml.aqi import convert_to_standard
from app.ml.graph import build_city_graph
from app.ml.inference import GCNInferenceEngine
from app.services.logging_utils import get_logger


@dataclass
class RuntimeCityProfile:
    name: str
    lat: float
    lon: float
    stations: list[dict[str, Any]]
    normalized_adjacency: np.ndarray


class RuntimePredictionService:
    def __init__(
        self,
        inference_engine: GCNInferenceEngine,
        archive_client: OpenAQArchiveClient | None = None,
        weather_client: OpenMeteoClient | None = None,
    ) -> None:
        self.engine = inference_engine
        self.archive = archive_client or OpenAQArchiveClient()
        self.weather = weather_client or OpenMeteoClient()
        self.live_openaq = OpenAQLiveClient()
        self.logger = get_logger("runtime_prediction")
        self._current_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._forecast_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._weather_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._downloaded_cities: dict[str, RuntimeCityProfile] = {}
        self._load_runtime_city_cache()

    def available_cities(self) -> list[str]:
        known = set(self.engine.available_cities())
        known.update(self._downloaded_cities.keys())
        return sorted(known)

    def has_city(self, city: str) -> bool:
        return city in METRO_CITY_CENTERS or city in self._downloaded_cities

    def city_stations(self, city: str) -> list[dict[str, Any]]:
        if city in METRO_CITY_CENTERS:
            return self.engine.city_stations(city)
        profile = self._downloaded_cities.get(city)
        if profile is None:
            raise KeyError(f"Unknown city '{city}'")
        return list(profile.stations)

    def resolve_runtime_city(self, city_name: str) -> str | None:
        normalized = city_name.strip().lower()
        for city in self.available_cities():
            if city.lower() == normalized:
                return city
        return None

    def current_city_prediction(self, city: str, cache_ttl_seconds: int = 1800) -> dict[str, Any]:
        cached = self._current_cache.get(city)
        now = time.time()
        if cached and cached[0] > now:
            return cached[1]

        stations = self.city_stations(city)
        city_weather = self._city_weather_now(city)
        feature_rows = []
        station_snapshots = []
        station_ids = [int(station["location_id"]) for station in stations]

        # Use live OpenAQ when a key is configured; otherwise fetch only the newest
        # archive file concurrently. Both paths avoid the previous 3-day, sequential
        # downloads that made city changes feel very slow.
        live_rows = self.live_openaq.latest_for_stations(stations) if self.live_openaq.enabled else {}
        archive_rows = {} if live_rows else self.archive.fetch_latest_station_rows_many(station_ids)

        for station in stations:
            loc_id = int(station["location_id"])
            if self.live_openaq.enabled:
                rows = pd.DataFrame(live_rows.get(loc_id, []))
            else:
                rows = archive_rows.get(loc_id, pd.DataFrame())
            hourly = self._rows_to_hourly_pivot(rows)
            if hourly.empty:
                feature = {pollutant: np.nan for pollutant in REQUIRED_POLLUTANTS}
                timestamp = None
            else:
                latest = hourly.sort_values("hour").iloc[-1]
                feature = {pollutant: float(latest.get(pollutant, np.nan)) for pollutant in REQUIRED_POLLUTANTS}
                timestamp = pd.Timestamp(latest["hour"]).isoformat()

            feature_rows.append({"location_id": loc_id, **feature, **city_weather})
            station_snapshots.append({"location_id": loc_id, "timestamp": timestamp, **feature})

        feature_frame = pd.DataFrame(feature_rows)
        predictions = self._predict_city(city, feature_frame)
        latest_ts = max((snap["timestamp"] for snap in station_snapshots if snap["timestamp"]), default=None)

        payload = {
            "city": city,
            "as_of_utc": latest_ts or datetime.now(timezone.utc).isoformat(),
            "stations": [
                {
                    **prediction,
                    "features": next(
                        snapshot for snapshot in station_snapshots if snapshot["location_id"] == prediction["location_id"]
                    ),
                }
                for prediction in predictions
            ],
            "city_avg_aqi": float(np.mean([item["predicted_aqi"] for item in predictions])),
        }

        self._current_cache[city] = (now + cache_ttl_seconds, payload)
        return payload

    def city_forecast(self, city: str, hours: int = 24, cache_ttl_seconds: int = 1800) -> dict[str, Any]:
        cache_key = f"{city}:{hours}"
        cached = self._forecast_cache.get(cache_key)
        now = time.time()
        if cached and cached[0] > now:
            return cached[1]

        stations = self.city_stations(city)
        weather = self._city_weather_series(city, hours=hours)

        station_histories: dict[int, pd.DataFrame] = {}
        station_ids = [int(station["location_id"]) for station in stations]
        if self.live_openaq.enabled:
            live_rows = self.live_openaq.latest_for_stations(stations)
            for loc_id in station_ids:
                station_histories[loc_id] = self._rows_to_hourly_pivot(pd.DataFrame(live_rows.get(loc_id, [])))
        else:
            archive_rows = self.archive.fetch_latest_station_rows_many(station_ids)
            for loc_id in station_ids:
                station_histories[loc_id] = self._rows_to_hourly_pivot(archive_rows.get(loc_id, pd.DataFrame()))

        timeline = weather["timestamp"].tolist()[:hours]
        forecast_steps: list[dict[str, Any]] = []
        station_series: dict[int, list[dict[str, Any]]] = {int(s["location_id"]): [] for s in stations}

        for step, ts in enumerate(timeline, start=1):
            w = weather.iloc[min(step - 1, len(weather) - 1)]
            feature_rows = []
            for station in stations:
                loc_id = int(station["location_id"])
                history = station_histories[loc_id]
                pollutant_forecast = self._forecast_pollutants_from_history(history, horizon_step=step)
                feature_rows.append(
                    {
                        "location_id": loc_id,
                        **pollutant_forecast,
                        "temperature_2m": float(w["temperature_2m"]),
                        "relative_humidity_2m": float(w["relative_humidity_2m"]),
                        "wind_speed_10m": float(w["wind_speed_10m"]),
                    }
                )

            feature_frame = pd.DataFrame(feature_rows)
            preds = self._predict_city(city, feature_frame)
            city_mean = float(np.mean([p["predicted_aqi"] for p in preds]))

            forecast_steps.append(
                {
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "city_avg_aqi": city_mean,
                }
            )

            for prediction in preds:
                station_series[int(prediction["location_id"])].append(
                    {
                        "timestamp": pd.Timestamp(ts).isoformat(),
                        "predicted_aqi": float(prediction["predicted_aqi"]),
                    }
                )

        payload = {
            "city": city,
            "hours": hours,
            "forecast": forecast_steps,
            "station_series": [
                {
                    "location_id": int(station["location_id"]),
                    "station_name": station["station_name"],
                    "series": station_series[int(station["location_id"])],
                }
                for station in stations
            ],
        }
        self._forecast_cache[cache_key] = (now + cache_ttl_seconds, payload)
        return payload

    def _get_weather_frame(self, city: str, hours: int = 24) -> pd.DataFrame:
        cache_key = f"{city}:{hours}"
        now = time.time()
        cached = self._weather_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]
        lat, lon = self._city_center(city)
        weather = self.weather.fetch_forecast_hourly(latitude=lat, longitude=lon, hours=hours)
        if weather.empty:
            raise RuntimeError("Live weather data is temporarily unavailable")
        self._weather_cache[cache_key] = (now + 300, weather)
        return weather

    def current_weather(self, city: str) -> dict[str, Any]:
        weather = self._get_weather_frame(city, hours=24)
        row = weather.iloc[0]
        return {
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "temperature_c": float(row["temperature_2m"]),
            "humidity_pct": float(row["relative_humidity_2m"]),
            "wind_kmh": float(row["wind_speed_10m"]),
        }

    def weather_forecast(self, city: str, hours: int = 24) -> list[dict[str, Any]]:
        weather = self._get_weather_frame(city, hours=hours)
        return [
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "temperature_c": float(row["temperature_2m"]),
                "humidity_pct": float(row["relative_humidity_2m"]),
                "wind_kmh": float(row["wind_speed_10m"]),
            }
            for _, row in weather.iterrows()
        ]

    def download_and_run_other_city(self, city_name: str, target_stations: int = 4) -> dict[str, Any]:
        geocoded = self.weather.geocode_city(city_name, country_code="IN")
        if geocoded is None:
            raise ValueError(f"Could not geocode city '{city_name}'")
        lat, lon, resolved_name = geocoded

        if self.live_openaq.enabled:
            discovered = self.live_openaq.discover_city(
                city_name=resolved_name, lat=lat, lon=lon,
                target_stations=target_stations, radius_km=45.0,
            )
        else:
            discovered = self.archive.discover_city_by_center(
                city_name=resolved_name, center_lat=lat, center_lon=lon,
                target_stations=target_stations, radius_km=45.0,
            )
        if not discovered:
            raise RuntimeError(f"No OpenAQ stations found near {resolved_name}")

        city_weather = self._city_weather_now_for_center(lat=lat, lon=lon)
        feature_rows: list[dict[str, Any]] = []
        station_ids = [int(station["location_id"]) for station in discovered]
        live_rows = self.live_openaq.latest_for_stations(discovered) if self.live_openaq.enabled else {}
        archive_rows = {} if live_rows else self.archive.fetch_latest_station_rows_many(station_ids)

        for station in discovered:
            loc_id = int(station["location_id"])
            rows = pd.DataFrame(live_rows.get(loc_id, [])) if self.live_openaq.enabled else archive_rows.get(loc_id, pd.DataFrame())
            hourly = self._rows_to_hourly_pivot(rows)
            if hourly.empty:
                pollutants = {pollutant: np.nan for pollutant in REQUIRED_POLLUTANTS}
            else:
                latest = hourly.sort_values("hour").iloc[-1]
                pollutants = {pollutant: float(latest.get(pollutant, np.nan)) for pollutant in REQUIRED_POLLUTANTS}

            feature_rows.append({"location_id": loc_id, **pollutants, **city_weather})

        feature_frame = pd.DataFrame(feature_rows)
        station_order = sorted(discovered, key=lambda s: int(s["location_id"]))
        ordered_ids = [int(station["location_id"]) for station in station_order]
        aligned = feature_frame.set_index("location_id").reindex(ordered_ids).reset_index()

        for feature in FEATURE_NAMES:
            if feature not in aligned.columns:
                aligned[feature] = np.nan
            aligned[feature] = pd.to_numeric(aligned[feature], errors="coerce")
            aligned[feature] = aligned[feature].fillna(float(aligned[feature].median()) if not np.isnan(aligned[feature].median()) else 0.0)

        coords = np.array([[float(station["lat"]), float(station["lon"])] for station in station_order], dtype=np.float32)
        graph = build_city_graph(city=resolved_name, coords=coords, k_neighbors=3)
        predictions = self.engine.predict_custom_graph(
            feature_matrix=aligned[list(FEATURE_NAMES)].to_numpy(dtype=np.float32),
            normalized_adjacency=graph.normalized_adjacency,
            stations=station_order,
        )

        self._downloaded_cities[resolved_name] = RuntimeCityProfile(
            name=resolved_name,
            lat=float(lat),
            lon=float(lon),
            stations=station_order,
            normalized_adjacency=graph.normalized_adjacency.copy(),
        )
        self._save_runtime_city_cache()
        self._current_cache.pop(resolved_name, None)
        for key in [k for k in self._forecast_cache if k.startswith(f"{resolved_name}:")]:
            self._forecast_cache.pop(key, None)

        return {
            "city": resolved_name,
            "warning": "Live OpenAQ mode" if self.live_openaq.enabled else "Archive fallback mode",
            "station_count": len(predictions),
            "predictions": predictions,
            "graph": {
                "nodes": graph.node_count,
                "edges": graph.edge_count,
                "components": graph.connected_components,
            },
        }

    def _save_runtime_city_cache(self) -> None:
        payload = {}
        for name, profile in self._downloaded_cities.items():
            payload[name] = {
                "name": profile.name,
                "lat": profile.lat,
                "lon": profile.lon,
                "stations": profile.stations,
                "normalized_adjacency": profile.normalized_adjacency.tolist(),
            }
        try:
            RUNTIME_CITY_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("Could not save runtime city cache: %s", exc)

    def _load_runtime_city_cache(self) -> None:
        if not RUNTIME_CITY_CACHE_PATH.exists():
            return
        try:
            payload = json.loads(RUNTIME_CITY_CACHE_PATH.read_text(encoding="utf-8"))
            for name, item in payload.items():
                self._downloaded_cities[name] = RuntimeCityProfile(
                    name=str(item["name"]),
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    stations=list(item["stations"]),
                    normalized_adjacency=np.asarray(item["normalized_adjacency"], dtype=np.float32),
                )
            self.logger.info("Loaded %s cached expansion cities", len(self._downloaded_cities))
        except Exception as exc:
            self.logger.warning("Could not load runtime city cache: %s", exc)

    def _predict_city(self, city: str, feature_frame: pd.DataFrame) -> list[dict[str, Any]]:
        if city in METRO_CITY_CENTERS:
            return self.engine.predict_city(city, feature_frame)

        profile = self._downloaded_cities.get(city)
        if profile is None:
            raise KeyError(f"Unknown city '{city}'")

        station_order = [int(station["location_id"]) for station in profile.stations]
        aligned = self._align_feature_frame_for_station_order(feature_frame, station_order)
        return self.engine.predict_custom_graph(
            feature_matrix=aligned.to_numpy(dtype=np.float32),
            normalized_adjacency=profile.normalized_adjacency,
            stations=profile.stations,
        )

    def _align_feature_frame_for_station_order(self, feature_frame: pd.DataFrame, station_order: list[int]) -> pd.DataFrame:
        frame = feature_frame.copy()
        required = ["location_id", *FEATURE_NAMES]
        for column in required:
            if column not in frame.columns:
                frame[column] = np.nan

        frame = frame[required].copy()
        frame["location_id"] = pd.to_numeric(frame["location_id"], errors="coerce")
        frame = frame.set_index("location_id").reindex(station_order)

        defaults = {
            name: float(self.engine.feature_mean[idx]) for idx, name in enumerate(self.engine.feature_names)
        }
        for feature in FEATURE_NAMES:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            city_median = frame[feature].median()
            fallback = defaults.get(feature, 0.0)
            fill_value = fallback if np.isnan(city_median) else float(city_median)
            frame[feature] = frame[feature].fillna(fill_value)

        frame = frame.reset_index(drop=True)
        return frame[list(FEATURE_NAMES)]

    def _rows_to_hourly_pivot(self, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame()

        frame = rows.copy()
        frame["parameter"] = frame["parameter"].str.lower()
        frame = frame[frame["parameter"].isin(REQUIRED_POLLUTANTS)]
        if frame.empty:
            return pd.DataFrame()

        frame["hour"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dt.floor("h")
        frame = frame.dropna(subset=["hour", "value"])
        frame["standard_value"] = frame.apply(
            lambda row: convert_to_standard(
                parameter=str(row["parameter"]),
                value=float(row["value"]),
                unit=str(row.get("units", "")),
            ),
            axis=1,
        )
        hourly = frame.groupby(["hour", "parameter"], as_index=False)["standard_value"].mean()
        pivot = hourly.pivot_table(index=["hour"], columns="parameter", values="standard_value", aggfunc="mean").reset_index()
        return pivot

    def _forecast_pollutants_from_history(self, history: pd.DataFrame, horizon_step: int) -> dict[str, float]:
        if history.empty:
            return {pollutant: 0.0 for pollutant in REQUIRED_POLLUTANTS}

        hist = history.sort_values("hour").copy()
        out: dict[str, float] = {}

        for pollutant in REQUIRED_POLLUTANTS:
            if pollutant not in hist.columns:
                out[pollutant] = 0.0
                continue
            series = pd.to_numeric(hist[pollutant], errors="coerce").dropna()
            if series.empty:
                out[pollutant] = 0.0
                continue

            values = series.to_numpy(dtype=np.float32)
            last = float(values[-1])
            if len(values) >= 3:
                recent = values[-12:] if len(values) >= 12 else values
                x = np.arange(len(recent), dtype=np.float32)
                slope = float(np.polyfit(x, recent, 1)[0])
            else:
                slope = 0.0

            forecast = max(0.0, last + slope * horizon_step)
            out[pollutant] = forecast

        return out

    def _city_weather_now(self, city: str) -> dict[str, float]:
        # Reuse the same 24-hour weather response used by forecast/weather endpoints.
        # This removes duplicate Open-Meteo requests when the frontend loads a city.
        weather = self._get_weather_frame(city, hours=24)
        if weather.empty:
            return {"temperature_2m": 0.0, "relative_humidity_2m": 0.0, "wind_speed_10m": 0.0}
        row = weather.iloc[0]
        return {
            "temperature_2m": float(row["temperature_2m"]),
            "relative_humidity_2m": float(row["relative_humidity_2m"]),
            "wind_speed_10m": float(row["wind_speed_10m"]),
        }

    def _city_center(self, city: str) -> tuple[float, float]:
        if city in METRO_CITY_CENTERS:
            center = METRO_CITY_CENTERS[city]
            return float(center.lat), float(center.lon)
        profile = self._downloaded_cities.get(city)
        if profile is None:
            raise KeyError(f"Unknown city '{city}'")
        return float(profile.lat), float(profile.lon)

    def _city_weather_now_for_center(self, lat: float, lon: float) -> dict[str, float]:
        weather = self.weather.fetch_forecast_hourly(latitude=lat, longitude=lon, hours=2)
        if weather.empty:
            return {"temperature_2m": 0.0, "relative_humidity_2m": 0.0, "wind_speed_10m": 0.0}
        row = weather.iloc[0]
        return {
            "temperature_2m": float(row["temperature_2m"]),
            "relative_humidity_2m": float(row["relative_humidity_2m"]),
            "wind_speed_10m": float(row["wind_speed_10m"]),
        }

    def _city_weather_series(self, city: str, hours: int) -> pd.DataFrame:
        weather = self._get_weather_frame(city, hours=hours)
        if weather.empty:
            start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            timestamps = [start + pd.Timedelta(hours=i) for i in range(hours)]
            return pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "temperature_2m": [0.0] * hours,
                    "relative_humidity_2m": [0.0] * hours,
                    "wind_speed_10m": [0.0] * hours,
                }
            )
        return weather.rename(columns={"timestamp": "timestamp"}).reset_index(drop=True)
