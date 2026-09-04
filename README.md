# AirSight — Fast AI Air Quality & Live Weather

A responsive React + FastAPI dashboard for Indian city air quality using the existing GCN inference engine, OpenAQ observations, and Open-Meteo weather.

## Performance upgrades

The on-demand city analysis has been optimized to avoid unnecessary work:

- OpenAQ station observations are fetched concurrently instead of one station at a time.
- The prediction path downloads only the newest archive file when using the archive fallback; the old 3-day download is no longer used for the current snapshot.
- Forecast station history uses the latest available snapshot and performs concurrent I/O.
- Open-Meteo 24-hour data is cached for 5 minutes and reused across current-weather, weather, current-AQI, and forecast requests.
- Discovered non-metro city profiles are persisted in `backend/data/runtime_city_cache.json`, so a city that has already been analyzed does not need to be rediscovered after a normal restart.
- Optional OpenAQ v3 live API support is included. When `OPENAQ_API_KEY` is configured, on-demand city discovery and latest measurements use the live API instead of the much slower archive-bucket scan. This is the fastest path.

## Local setup

### Backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
# Optional: set OPENAQ_API_KEY for fast OpenAQ v3 mode.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Windows users can use `Copy-Item .env.example .env` instead of `cp`.

Set `VITE_API_BASE_URL` to the backend URL when deployed.

## Render

The repository includes `backend/render.yaml`.

- Root directory: `backend`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Optional environment variable: `OPENAQ_API_KEY`

For maximum speed on the **Analyze city** feature, add a valid OpenAQ v3 API key to the Render service environment.

## Vercel

Set the Vercel project root directory to `frontend`.

- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com`

## API additions

- `GET /city/{city}/weather` — live current weather + hourly weather forecast.
- Existing GCN AQI and expansion endpoints remain available.

## Important

Do not commit API keys. Use environment variables for deployment secrets.
