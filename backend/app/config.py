from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_DIR = BACKEND_ROOT / "data"
ARTIFACT_DIR = BACKEND_ROOT / "artifacts"
LOG_DIR = BACKEND_ROOT / "logs"

for _path in (DATA_DIR, ARTIFACT_DIR, LOG_DIR):
    _path.mkdir(parents=True, exist_ok=True)


OPENAQ_ARCHIVE_BASE = "https://openaq-data-archive.s3.amazonaws.com"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


REQUIRED_POLLUTANTS = ("pm25", "pm10", "no2", "so2", "co", "o3")
WEATHER_FEATURES = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m")
FEATURE_NAMES = (*REQUIRED_POLLUTANTS, *WEATHER_FEATURES)


TRAINING_YEAR = 2025
TRAINING_MONTH = 2

MAX_STATIONS_PER_CITY = 4
CITY_RADIUS_KM = 45.0
DISCOVERY_BATCH_SIZE = 300
DISCOVERY_CONCURRENCY = 300
GRAPH_K_NEIGHBORS = 3

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 2026

MODEL_PATH = ARTIFACT_DIR / "gcn_aqi_model.pt"
METRO_STATIONS_PATH = DATA_DIR / "metro_stations.json"
LOCATION_IDS_CACHE_PATH = DATA_DIR / "location_ids.json"
RAW_MONTHLY_PATH = DATA_DIR / "metro_monthly_rows.pkl"
FEATURE_TABLE_PATH = DATA_DIR / "metro_feature_table.pkl"
GRAPH_META_PATH = DATA_DIR / "graph_meta.json"
RUNTIME_CITY_CACHE_PATH = DATA_DIR / "runtime_city_cache.json"

# Optional OpenAQ v3 live API key. If set, on-demand city analysis uses the live API
# instead of scanning the archive bucket, which is substantially faster.
OPENAQ_API_BASE = "https://api.openaq.org/v3"
OPENAQ_API_KEY = __import__("os").getenv("OPENAQ_API_KEY", "").strip()
TRAINING_SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"


@dataclass(frozen=True)
class CityCenter:
    name: str
    lat: float
    lon: float


METRO_CITY_CENTERS = {
    "Delhi": CityCenter("Delhi", 28.6139, 77.2090),
    "Mumbai": CityCenter("Mumbai", 19.0760, 72.8777),
    "Bengaluru": CityCenter("Bengaluru", 12.9716, 77.5946),
    "Chennai": CityCenter("Chennai", 13.0827, 80.2707),
    "Kolkata": CityCenter("Kolkata", 22.5726, 88.3639),
    "Hyderabad": CityCenter("Hyderabad", 17.3850, 78.4867),
    "Pune": CityCenter("Pune", 18.5204, 73.8567),
    "Ahmedabad": CityCenter("Ahmedabad", 23.0225, 72.5714),
}

METRO_CITY_NAMES = tuple(METRO_CITY_CENTERS.keys())

# User-initiated on-demand expansion list. Restricted to valid Indian city names only.
INDIAN_EXPANSION_CITY_NAMES = (
    "Agra",
    "Ajmer",
    "Aligarh",
    "Amritsar",
    "Asansol",
    "Aurangabad",
    "Bareilly",
    "Bhopal",
    "Bhubaneswar",
    "Bikaner",
    "Bilaspur",
    "Chandigarh",
    "Coimbatore",
    "Cuttack",
    "Dehradun",
    "Dhanbad",
    "Faridabad",
    "Ghaziabad",
    "Guntur",
    "Gurugram",
    "Guwahati",
    "Gwalior",
    "Howrah",
    "Indore",
    "Jabalpur",
    "Jaipur",
    "Jalandhar",
    "Jamshedpur",
    "Jhansi",
    "Jodhpur",
    "Kanpur",
    "Kochi",
    "Kolhapur",
    "Kota",
    "Lucknow",
    "Ludhiana",
    "Madurai",
    "Mangalore",
    "Meerut",
    "Moradabad",
    "Mysuru",
    "Nagpur",
    "Nainital",
    "Nashik",
    "Navi Mumbai",
    "Noida",
    "Panaji",
    "Patna",
    "Pimpri-Chinchwad",
    "Prayagraj",
    "Puducherry",
    "Raipur",
    "Rajkot",
    "Ranchi",
    "Salem",
    "Shimla",
    "Siliguri",
    "Srinagar",
    "Surat",
    "Thane",
    "Thiruvananthapuram",
    "Tiruchirappalli",
    "Udaipur",
    "Ujjain",
    "Vadodara",
    "Varanasi",
    "Vijayawada",
    "Visakhapatnam",
    "Warangal",
)
