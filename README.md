# AtomsIQ — AI-Powered Air Quality & Weather Intelligence
 
AtomsIQ is a responsive AI-powered air quality and weather intelligence platform for Indian cities. It combines real-time OpenAQ observations, Open-Meteo weather data, and a Graph Convolutional Network (GCN) inference engine to analyze air quality and provide weather-aware insights.
 
The platform provides city-level AQI analysis, pollutant information, forecasts, live weather conditions, and AI-driven environmental insights through a modern responsive dashboard.
 
## Key Features
 
- 🌍 City-level air quality analysis

- 📊 GCN-based AQI prediction and inference

- 🌤️ Live weather conditions using Open-Meteo

- 📅 Hourly and forecast weather information

- 💨 Pollutant and monitoring-station information from OpenAQ

- 🤖 AI-powered air quality and weather insights

- 🗺️ Support for multiple Indian cities

- 📱 Responsive design optimized for desktop, tablet, and mobile

- ⚡ Optimized data fetching and concurrent API requests

- 🚀 Production-ready deployment with Vercel and Render
 
## Performance Optimizations
 
The city analysis pipeline has been optimized to reduce unnecessary processing and API latency.
 
- OpenAQ station observations are fetched concurrently instead of sequentially.

- The current prediction path downloads only the newest available archive file when using the archive fallback.

- Forecast station history uses the latest available snapshot with concurrent I/O.

- Open-Meteo weather responses are cached for 5 minutes and reused across weather, AQI, and forecast requests.

- Discovered non-metro city profiles are persisted in `backend/data/runtime_city_cache.json`, reducing repeated city discovery.

- Optional OpenAQ v3 API support provides a faster path for city discovery and latest measurements when `OPENAQ_API_KEY` is configured.
 
## Technology Stack
 
### Frontend
 
- React

- TypeScript

- Vite

- Responsive CSS/UI

- Recharts

- Leaflet / Maps
 
### Backend
 
- Python

- FastAPI

- Uvicorn

- OpenAQ

- Open-Meteo

- Graph Convolutional Network (GCN)

- Machine Learning inference
 
## Project Structure
 
```text

AtomsIQ/

│

├── backend/

│   ├── app/

│   ├── artifacts/

│   ├── data/

│   ├── scripts/

│   ├── requirements.txt

│   └── render.yaml

│

├── frontend/

│   ├── src/

│   ├── public/

│   ├── package.json

│   └── vite.config.*

│

├── vercel.json

├── .gitignore

└── README.md
 
