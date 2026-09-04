from __future__ import annotations

import asyncio
import csv
import gzip
import io
import json
import random
import re
from typing import Iterable

import aiohttp
import pandas as pd
import requests
import xml.etree.ElementTree as ET

from app.config import (
    CITY_RADIUS_KM,
    DISCOVERY_BATCH_SIZE,
    DISCOVERY_CONCURRENCY,
    LOCATION_IDS_CACHE_PATH,
    MAX_STATIONS_PER_CITY,
    METRO_CITY_CENTERS,
    METRO_CITY_NAMES,
    METRO_STATIONS_PATH,
    OPENAQ_ARCHIVE_BASE,
    REQUIRED_POLLUTANTS,
    TRAINING_MONTH,
    TRAINING_YEAR,
)
from app.ml.graph import haversine_distance_km
from app.services.logging_utils import get_logger


class OpenAQArchiveClient:
    NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    def __init__(self) -> None:
        self.base_url = OPENAQ_ARCHIVE_BASE.rstrip("/")
        self.logger = get_logger("openaq_archive")
        self._requests_session = requests.Session()

    def load_location_ids(self, force_refresh: bool = False) -> list[int]:
        if LOCATION_IDS_CACHE_PATH.exists() and not force_refresh:
            ids = json.loads(LOCATION_IDS_CACHE_PATH.read_text(encoding="utf-8"))
            self.logger.info("Loaded %s location IDs from cache", len(ids))
            return [int(x) for x in ids]

        ids: list[int] = []
        continuation: str | None = None

        while True:
            params = {
                "list-type": "2",
                "prefix": "records/csv.gz/",
                "delimiter": "/",
                "max-keys": "1000",
            }
            if continuation:
                params["continuation-token"] = continuation

            response = self._requests_session.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            for prefix_node in root.findall("s3:CommonPrefixes", self.NS):
                prefix = prefix_node.find("s3:Prefix", self.NS)
                if prefix is None or not prefix.text:
                    continue
                match = re.search(r"locationid=(\d+)/", prefix.text)
                if match:
                    ids.append(int(match.group(1)))

            continuation = root.findtext("s3:NextContinuationToken", default=None, namespaces=self.NS)
            if not continuation:
                break

        LOCATION_IDS_CACHE_PATH.write_text(json.dumps(ids), encoding="utf-8")
        self.logger.info("Fetched and cached %s location IDs", len(ids))
        return ids

    async def discover_metro_stations(
        self,
        target_per_city: int = MAX_STATIONS_PER_CITY,
        radius_km: float = CITY_RADIUS_KM,
        year: int = TRAINING_YEAR,
        month: int = TRAINING_MONTH,
        force_refresh: bool = False,
        random_seed: int = 2026,
    ) -> list[dict]:
        if METRO_STATIONS_PATH.exists() and not force_refresh:
            payload = json.loads(METRO_STATIONS_PATH.read_text(encoding="utf-8"))
            if (
                payload.get("year") == year
                and payload.get("month") == month
                and payload.get("target_per_city") == target_per_city
                and payload.get("radius_km") == radius_km
            ):
                self.logger.info("Loaded metro station discovery from cache")
                return payload["stations"]

        location_ids = self.load_location_ids(force_refresh=False)
        rng = random.Random(random_seed)
        rng.shuffle(location_ids)

        counts = {city: 0 for city in METRO_CITY_NAMES}
        selected: list[dict] = []

        connector = aiohttp.TCPConnector(limit=DISCOVERY_CONCURRENCY, ssl=False)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for offset in range(0, len(location_ids), DISCOVERY_BATCH_SIZE):
                batch_ids = location_ids[offset : offset + DISCOVERY_BATCH_SIZE]
                tasks = [
                    self._inspect_location_candidate(
                        session=session,
                        location_id=location_id,
                        year=year,
                        month=month,
                        radius_km=radius_km,
                    )
                    for location_id in batch_ids
                ]
                results = await asyncio.gather(*tasks)

                for station in results:
                    if not station:
                        continue
                    city = station["city"]
                    if counts[city] >= target_per_city:
                        continue
                    counts[city] += 1
                    selected.append(station)

                if offset % (DISCOVERY_BATCH_SIZE * 10) == 0:
                    self.logger.info(
                        "Station discovery progress | scanned=%s | counts=%s",
                        min(offset + DISCOVERY_BATCH_SIZE, len(location_ids)),
                        counts,
                    )

                if all(counts[city] >= target_per_city for city in METRO_CITY_NAMES):
                    break

        missing = [city for city in METRO_CITY_NAMES if counts[city] < target_per_city]
        if missing:
            raise RuntimeError(
                f"Could not find enough stations for metro cities: {missing}. "
                f"Counts discovered: {counts}"
            )

        payload = {
            "year": year,
            "month": month,
            "target_per_city": target_per_city,
            "radius_km": radius_km,
            "stations": selected,
        }
        METRO_STATIONS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info("Saved %s discovered metro stations", len(selected))
        return selected

    async def _inspect_location_candidate(
        self,
        session: aiohttp.ClientSession,
        location_id: int,
        year: int,
        month: int,
        radius_km: float,
    ) -> dict | None:
        key = await self._find_any_key_for_month(session=session, location_id=location_id, year=year, month=month)
        if not key:
            return None

        csv_bytes = await self._fetch_bytes(session=session, url=f"{self.base_url}/{key}")
        if not csv_bytes:
            return None

        sample_row = self._read_first_row(csv_bytes)
        if sample_row is None:
            return None

        lat = sample_row.get("lat")
        lon = sample_row.get("lon")
        if lat is None or lon is None:
            return None

        nearest_city = None
        nearest_distance = float("inf")
        for city, center in METRO_CITY_CENTERS.items():
            distance = haversine_distance_km(lat, lon, center.lat, center.lon)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_city = city

        if nearest_city is None or nearest_distance > radius_km:
            return None

        return {
            "city": nearest_city,
            "location_id": int(location_id),
            "station_name": sample_row.get("location") or f"location-{location_id}",
            "lat": float(lat),
            "lon": float(lon),
            "distance_km": float(nearest_distance),
            "sample_key": key,
        }

    async def _find_any_key_for_month(
        self,
        session: aiohttp.ClientSession,
        location_id: int,
        year: int,
        month: int,
    ) -> str | None:
        prefix = f"records/csv.gz/locationid={location_id}/year={year}/month={month:02d}/"
        url = f"{self.base_url}?list-type=2&prefix={prefix}&max-keys=1"
        text = await self._fetch_text(session=session, url=url)
        if not text:
            return None

        match = re.search(r"<Key>([^<]+)</Key>", text)
        if not match:
            return None
        key = match.group(1)
        if not key.endswith(".csv.gz"):
            return None
        return key

    async def download_monthly_rows(self, stations: Iterable[dict], year: int = TRAINING_YEAR, month: int = TRAINING_MONTH) -> pd.DataFrame:
        station_list = list(stations)
        if not station_list:
            raise ValueError("No stations provided")

        station_map = {int(s["location_id"]): s for s in station_list}

        file_keys: list[tuple[int, str]] = []
        for location_id in station_map:
            keys = self._list_file_keys_for_month(location_id=location_id, year=year, month=month)
            file_keys.extend((location_id, key) for key in keys)

        if not file_keys:
            raise RuntimeError("No monthly files found for discovered stations")

        self.logger.info("Downloading %s monthly files from OpenAQ archive", len(file_keys))

        connector = aiohttp.TCPConnector(limit=200, ssl=False)
        timeout = aiohttp.ClientTimeout(total=30)
        rows: list[dict] = []

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self._download_and_parse_rows(session=session, location_id=loc_id, key=key) for loc_id, key in file_keys]
            parsed_batches = await asyncio.gather(*tasks)

        for loc_id, parsed_rows in parsed_batches:
            station_info = station_map.get(loc_id)
            if station_info is None:
                continue
            for row in parsed_rows:
                row["city"] = station_info["city"]
                row["station_name"] = station_info["station_name"]
                rows.append(row)

        if not rows:
            raise RuntimeError("Monthly download completed but no pollutant rows were parsed")

        frame = pd.DataFrame(rows)
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["location_id"] = frame["location_id"].astype(int)
        frame = frame.dropna(subset=["datetime", "value"]).reset_index(drop=True)

        return frame

    def fetch_latest_station_rows(self, location_id: int) -> pd.DataFrame:
        """Fetch only the newest archive file for a station.

        This is the fast archive fallback: the prediction only needs the latest
        pollutant snapshot, so downloading several days of history is unnecessary.
        """
        years = self._list_years_for_location(location_id)
        if not years:
            return pd.DataFrame()
        latest_year = max(years)
        months = self._list_months_for_location(location_id, latest_year)
        if not months:
            return pd.DataFrame()
        latest_month = max(months)
        keys = self._list_file_keys_for_month(location_id, latest_year, latest_month)
        if not keys:
            return pd.DataFrame()
        response = self._requests_session.get(f"{self.base_url}/{keys[-1]}", timeout=30)
        if response.status_code != 200:
            return pd.DataFrame()
        rows = self._parse_rows_from_csv_bytes(response.content)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["datetime", "value"]).reset_index(drop=True)
        frame["location_id"] = frame["location_id"].astype(int)
        return frame

    def fetch_latest_station_rows_many(self, location_ids: Iterable[int]) -> dict[int, pd.DataFrame]:
        """Fetch station snapshots concurrently for on-demand analysis."""
        ids = [int(x) for x in location_ids]
        result: dict[int, pd.DataFrame] = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(ids)))) as pool:
            futures = {pool.submit(self.fetch_latest_station_rows, loc_id): loc_id for loc_id in ids}
            for future in as_completed(futures):
                loc_id = futures[future]
                try:
                    result[loc_id] = future.result()
                except Exception as exc:
                    self.logger.warning("Latest archive fetch failed for %s: %s", loc_id, exc)
                    result[loc_id] = pd.DataFrame()
        return result


    def discover_city_by_center(
        self,
        city_name: str,
        center_lat: float,
        center_lon: float,
        target_stations: int = 4,
        radius_km: float = 45.0,
        year: int = TRAINING_YEAR,
        month: int = TRAINING_MONTH,
        random_seed: int = 1337,
    ) -> list[dict]:
        location_ids = self.load_location_ids(force_refresh=False)
        rng = random.Random(random_seed)
        rng.shuffle(location_ids)

        selected: list[dict] = []

        async def _run() -> list[dict]:
            connector = aiohttp.TCPConnector(limit=DISCOVERY_CONCURRENCY, ssl=False)
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                for offset in range(0, len(location_ids), DISCOVERY_BATCH_SIZE):
                    batch_ids = location_ids[offset : offset + DISCOVERY_BATCH_SIZE]
                    tasks = [self._inspect_custom_city_candidate(session, loc_id, center_lat, center_lon, city_name, radius_km, year, month) for loc_id in batch_ids]
                    results = await asyncio.gather(*tasks)
                    for station in results:
                        if station:
                            selected.append(station)
                            if len(selected) >= target_stations:
                                return selected
            return selected

        stations = asyncio.run(_run())
        return stations

    async def _inspect_custom_city_candidate(
        self,
        session: aiohttp.ClientSession,
        location_id: int,
        center_lat: float,
        center_lon: float,
        city_name: str,
        radius_km: float,
        year: int,
        month: int,
    ) -> dict | None:
        key = await self._find_any_key_for_month(session, location_id, year, month)
        if not key:
            return None

        csv_bytes = await self._fetch_bytes(session=session, url=f"{self.base_url}/{key}")
        if not csv_bytes:
            return None

        row = self._read_first_row(csv_bytes)
        if row is None:
            return None

        lat = row.get("lat")
        lon = row.get("lon")
        if lat is None or lon is None:
            return None

        distance = haversine_distance_km(lat, lon, center_lat, center_lon)
        if distance > radius_km:
            return None

        return {
            "city": city_name,
            "location_id": int(location_id),
            "station_name": row.get("location") or f"location-{location_id}",
            "lat": float(lat),
            "lon": float(lon),
            "distance_km": float(distance),
            "sample_key": key,
        }

    def _list_file_keys_for_month(self, location_id: int, year: int, month: int) -> list[str]:
        prefix = f"records/csv.gz/locationid={location_id}/year={year}/month={month:02d}/"
        continuation: str | None = None
        keys: list[str] = []

        while True:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if continuation:
                params["continuation-token"] = continuation
            response = self._requests_session.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            for content_node in root.findall("s3:Contents", self.NS):
                key_node = content_node.find("s3:Key", self.NS)
                if key_node is None or not key_node.text:
                    continue
                if key_node.text.endswith(".csv.gz"):
                    keys.append(key_node.text)

            continuation = root.findtext("s3:NextContinuationToken", default=None, namespaces=self.NS)
            if not continuation:
                break

        keys.sort()
        return keys

    def _list_years_for_location(self, location_id: int) -> list[int]:
        prefix = f"records/csv.gz/locationid={location_id}/"
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        response = self._requests_session.get(self.base_url, params=params, timeout=60)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        years: list[int] = []
        for prefix_node in root.findall("s3:CommonPrefixes", self.NS):
            value = prefix_node.find("s3:Prefix", self.NS)
            if value is None or not value.text:
                continue
            match = re.search(r"year=(\d+)/", value.text)
            if match:
                years.append(int(match.group(1)))

        return sorted(years)

    def _list_months_for_location(self, location_id: int, year: int) -> list[int]:
        prefix = f"records/csv.gz/locationid={location_id}/year={year}/"
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        response = self._requests_session.get(self.base_url, params=params, timeout=60)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        months: list[int] = []
        for prefix_node in root.findall("s3:CommonPrefixes", self.NS):
            value = prefix_node.find("s3:Prefix", self.NS)
            if value is None or not value.text:
                continue
            match = re.search(r"month=(\d+)/", value.text)
            if match:
                months.append(int(match.group(1)))

        return sorted(months)

    async def _download_and_parse_rows(
        self,
        session: aiohttp.ClientSession,
        location_id: int,
        key: str,
    ) -> tuple[int, list[dict]]:
        data = await self._fetch_bytes(session=session, url=f"{self.base_url}/{key}")
        if not data:
            return location_id, []
        return location_id, self._parse_rows_from_csv_bytes(data)

    async def _fetch_text(self, session: aiohttp.ClientSession, url: str) -> str | None:
        for attempt in range(3):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
            except Exception:
                await asyncio.sleep(0.2 * (attempt + 1))
        return None

    async def _fetch_bytes(self, session: aiohttp.ClientSession, url: str) -> bytes | None:
        for attempt in range(3):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
            except Exception:
                await asyncio.sleep(0.2 * (attempt + 1))
        return None

    @staticmethod
    def _read_first_row(csv_bytes: bytes) -> dict | None:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(csv_bytes)) as gz_file:
                lines = gz_file.read().decode("latin1", errors="ignore").splitlines()
            if len(lines) < 2:
                return None
            row = next(csv.DictReader(lines))
            return {
                "location": row.get("location"),
                "lat": float(row.get("lat") or 0.0),
                "lon": float(row.get("lon") or 0.0),
            }
        except Exception:
            return None

    @staticmethod
    def _parse_rows_from_csv_bytes(csv_bytes: bytes) -> list[dict]:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(csv_bytes)) as gz_file:
                lines = gz_file.read().decode("latin1", errors="ignore").splitlines()
        except Exception:
            return []

        rows: list[dict] = []
        for row in csv.DictReader(lines):
            parameter = (row.get("parameter") or "").strip().lower()
            if parameter not in REQUIRED_POLLUTANTS:
                continue
            rows.append(
                {
                    "location_id": int(row.get("location_id") or 0),
                    "sensors_id": int(row.get("sensors_id") or 0),
                    "datetime": row.get("datetime"),
                    "lat": float(row.get("lat") or 0.0),
                    "lon": float(row.get("lon") or 0.0),
                    "parameter": parameter,
                    "units": (row.get("units") or "").strip(),
                    "value": row.get("value"),
                }
            )
        return rows
