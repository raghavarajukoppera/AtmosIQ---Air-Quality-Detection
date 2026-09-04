from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from app.config import OPENAQ_API_BASE, OPENAQ_API_KEY, REQUIRED_POLLUTANTS
from app.ml.graph import haversine_distance_km
from app.services.logging_utils import get_logger


class OpenAQLiveClient:
    """Fast OpenAQ v3 adapter used when OPENAQ_API_KEY is configured."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or OPENAQ_API_KEY).strip()
        self.base_url = OPENAQ_API_BASE.rstrip("/")
        self.logger = get_logger("openaq_live")
        self.session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key, "Accept": "application/json"}

    def discover_city(self, city_name: str, lat: float, lon: float, target_stations: int = 4, radius_km: float = 45.0) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": int(radius_km * 1000),
            "limit": 100,
            "page": 1,
        }
        response = self.session.get(f"{self.base_url}/locations", params=params, headers=self._headers(), timeout=15)
        response.raise_for_status()
        results = response.json().get("results") or []

        candidates: list[dict[str, Any]] = []
        for item in results:
            coords = item.get("coordinates") or {}
            item_lat = coords.get("latitude")
            item_lon = coords.get("longitude")
            if item_lat is None or item_lon is None:
                continue
            parameters = set()
            for sensor in item.get("sensors") or []:
                parameter = sensor.get("parameter") or {}
                name = parameter.get("name") if isinstance(parameter, dict) else parameter
                if name:
                    parameters.add(str(name).lower().replace(".", ""))
            # Keep stations with at least one pollutant sensor. Latest values will be
            # filled in concurrently below.
            candidates.append({
                "city": city_name,
                "location_id": int(item.get("id")),
                "station_name": item.get("name") or f"location-{item.get('id')}",
                "lat": float(item_lat),
                "lon": float(item_lon),
                "distance_km": float(haversine_distance_km(float(item_lat), float(item_lon), lat, lon)),
                "parameters": sorted(parameters),
            })

        candidates.sort(key=lambda x: (len(set(x.get("parameters", [])) & set(REQUIRED_POLLUTANTS)) < 3, x["distance_km"]))
        selected = candidates[: max(target_stations, 1)]
        for item in selected:
            item.pop("parameters", None)
        return selected

    def latest_station_rows(self, location_id: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        response = self.session.get(
            f"{self.base_url}/locations/{int(location_id)}/latest",
            headers=self._headers(),
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        rows: list[dict[str, Any]] = []
        for item in results:
            parameter = item.get("parameter") or {}
            if isinstance(parameter, dict):
                name = parameter.get("name") or parameter.get("displayName")
                units = parameter.get("units") or item.get("unit") or ""
            else:
                name = parameter
                units = item.get("unit") or ""
            name = str(name or "").lower().replace(".", "")
            if name not in REQUIRED_POLLUTANTS:
                continue
            value = item.get("value")
            if value is None:
                continue
            date_obj = item.get("date")
            dt = item.get("datetime")
            if not dt and isinstance(date_obj, dict):
                dt = date_obj.get("utc")
            rows.append({
                "location_id": int(location_id),
                "datetime": dt,
                "parameter": name,
                "units": units,
                "value": value,
            })
        return rows

    def latest_for_stations(self, stations: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        result: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(stations)))) as pool:
            futures = {pool.submit(self.latest_station_rows, int(s["location_id"])): int(s["location_id"]) for s in stations}
            for future in as_completed(futures):
                loc_id = futures[future]
                try:
                    result[loc_id] = future.result()
                except Exception as exc:
                    self.logger.warning("OpenAQ live latest failed for %s: %s", loc_id, exc)
                    result[loc_id] = []
        return result
