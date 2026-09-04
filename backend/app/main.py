from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import INDIAN_EXPANSION_CITY_NAMES, METRO_CITY_NAMES, MODEL_PATH
from app.ml.inference import GCNInferenceEngine
from app.services.runtime_prediction import RuntimePredictionService


app = FastAPI(title="GCN AQI Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine: GCNInferenceEngine | None = None
runtime_service: RuntimePredictionService | None = None


def _require_runtime() -> RuntimePredictionService:
    if runtime_service is None:
        raise HTTPException(status_code=503, detail="Model runtime is not initialized")
    return runtime_service


def _resolve_metro_city(city_name: str) -> str | None:
    normalized = city_name.strip().lower()
    aliases = {
        "bangalore": "Bengaluru",
        "new delhi": "Delhi",
        "delhi ncr": "Delhi",
    }
    if normalized in aliases:
        return aliases[normalized]
    for city in METRO_CITY_NAMES:
        if city.lower() == normalized:
            return city
    return None


def _resolve_other_city(city_name: str) -> str | None:
    normalized = city_name.strip().lower()
    aliases = {
        "allahabad": "Prayagraj",
        "pondicherry": "Puducherry",
        "trivandrum": "Thiruvananthapuram",
        "vizag": "Visakhapatnam",
    }
    if normalized in aliases:
        normalized = aliases[normalized].lower()
    for city in INDIAN_EXPANSION_CITY_NAMES:
        if city.lower() == normalized:
            return city
    return None


def _resolve_runtime_city(city_name: str) -> str | None:
    metro = _resolve_metro_city(city_name)
    if metro is not None:
        return metro
    if runtime_service is None:
        return None
    return runtime_service.resolve_runtime_city(city_name)


@app.on_event("startup")
def startup_event() -> None:
    global inference_engine, runtime_service
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact does not exist at {MODEL_PATH}. Run training first.")
    inference_engine = GCNInferenceEngine(model_path=MODEL_PATH)
    runtime_service = RuntimePredictionService(inference_engine=inference_engine)


@app.get("/cities")
def list_cities() -> dict[str, Any]:
    downloaded: list[str] = []
    if runtime_service is not None:
        downloaded = [city for city in runtime_service.available_cities() if city not in METRO_CITY_NAMES]
    available_cities = list(METRO_CITY_NAMES)
    available_cities.extend([city for city in downloaded if city not in available_cities])
    return {
        "metro_cities": list(METRO_CITY_NAMES),
        "available_cities": available_cities,
        "downloaded_cities": downloaded,
        "note": "Non-metro inference is available only via /city/{city_name}/download-and-run",
    }


@app.get("/other-cities")
def list_other_cities() -> dict[str, Any]:
    return {
        "other_cities": list(INDIAN_EXPANSION_CITY_NAMES),
        "note": "Only these validated Indian city names are accepted for on-demand download-and-run.",
    }


@app.get("/city/{city_name}/stations")
def city_stations(city_name: str) -> dict[str, Any]:
    city = _resolve_runtime_city(city_name)
    if city is None:
        raise HTTPException(
            status_code=404,
            detail=f"{city_name} is not available. Run /city/{{city_name}}/download-and-run first for non-metro cities.",
        )
    runtime = _require_runtime()
    stations = runtime.city_stations(city)
    return {"city": city, "station_count": len(stations), "stations": stations}


@app.get("/city/{city_name}/current-aqi")
def city_current_aqi(city_name: str) -> dict[str, Any]:
    city = _resolve_runtime_city(city_name)
    if city is None:
        raise HTTPException(
            status_code=404,
            detail=f"{city_name} is not available. Run /city/{{city_name}}/download-and-run first for non-metro cities.",
        )
    runtime = _require_runtime()
    try:
        return runtime.current_city_prediction(city)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/city/{city_name}/forecast")
def city_forecast(city_name: str, hours: int = Query(default=24, ge=1, le=72)) -> dict[str, Any]:
    city = _resolve_runtime_city(city_name)
    if city is None:
        raise HTTPException(
            status_code=404,
            detail=f"{city_name} is not available. Run /city/{{city_name}}/download-and-run first for non-metro cities.",
        )
    runtime = _require_runtime()
    try:
        return runtime.city_forecast(city, hours=hours)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/city/{city_name}/weather")
def city_weather(city_name: str, hours: int = Query(default=24, ge=1, le=72)) -> dict[str, Any]:
    city = _resolve_runtime_city(city_name)
    if city is None:
        raise HTTPException(status_code=404, detail=f"{city_name} is not available")
    runtime = _require_runtime()
    try:
        current = runtime.current_weather(city)
        forecast = runtime.weather_forecast(city, hours=hours)
        return {"city": city, "current": current, "forecast": forecast}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/city/{city_name}/download-and-run")
def download_and_run_other_city(city_name: str) -> dict[str, Any]:
    if _resolve_metro_city(city_name) is not None:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is reserved for non-metro cities. Use metro endpoints for default cities.",
        )
    resolved = _resolve_other_city(city_name)
    if resolved is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{city_name}' is not an allowed Indian city name for expansion. "
                "Use GET /other-cities to choose a valid city."
            ),
        )
    runtime = _require_runtime()
    try:
        return runtime.download_and_run_other_city(resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
