from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import OPEN_METEO_ARCHIVE_URL, OPEN_METEO_FORECAST_URL, OPEN_METEO_GEOCODE_URL, WEATHER_FEATURES


class OpenMeteoClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    def fetch_historical_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_FEATURES),
            "timezone": "UTC",
        }
        response = self.session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        return self._hourly_payload_to_frame(payload)

    def fetch_forecast_hourly(self, latitude: float, longitude: float, hours: int = 48) -> pd.DataFrame:
        days = max(1, min(16, int(hours / 24) + 1))
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(WEATHER_FEATURES),
            "timezone": "UTC",
            "forecast_days": days,
        }
        response = self.session.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        frame = self._hourly_payload_to_frame(payload)
        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=hours)
        return frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].reset_index(drop=True)

    def geocode_city(self, city_name: str, country_code: str | None = None) -> tuple[float, float, str] | None:
        params = {
            "name": city_name,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        if country_code:
            params["countryCode"] = country_code
        response = self.session.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return None
        top = results[0]
        return float(top["latitude"]), float(top["longitude"]), str(top.get("name") or city_name)

    @staticmethod
    def _hourly_payload_to_frame(payload: dict) -> pd.DataFrame:
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        frame = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True, errors="coerce")})
        for field in WEATHER_FEATURES:
            frame[field] = pd.to_numeric(hourly.get(field, []), errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        return frame
