import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { MapContainer, CircleMarker, Popup, TileLayer } from 'react-leaflet';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

interface Station { location_id: number; station_name: string; lat: number; lon: number; }
interface StationFeatures { pm25:number|null; pm10:number|null; no2:number|null; so2:number|null; co:number|null; o3:number|null; timestamp:string|null; }
interface Current { city:string; as_of_utc:string; city_avg_aqi:number; stations:Array<Station & {predicted_aqi:number; features:StationFeatures}>; }
interface Forecast { city:string; hours:number; forecast:Array<{timestamp:string;city_avg_aqi:number}>; station_series:any[]; }
interface Weather { city:string; current:{timestamp:string;temperature_c:number;humidity_pct:number;wind_kmh:number}; forecast:Array<{timestamp:string;temperature_c:number;humidity_pct:number;wind_kmh:number}>; }
interface Expansion { city:string; warning:string; station_count:number; predictions:any[]; graph:{nodes:number;edges:number;components:number}; }

function aqiBand(aqi:number){ if(aqi<=50)return'Good'; if(aqi<=100)return'Satisfactory'; if(aqi<=200)return'Moderate'; if(aqi<=300)return'Poor'; if(aqi<=400)return'Very Poor'; return'Severe'; }
function aqiColor(aqi:number){ if(aqi<=50)return'#16a34a'; if(aqi<=100)return'#65a30d'; if(aqi<=200)return'#f59e0b'; if(aqi<=300)return'#f97316'; if(aqi<=400)return'#ef4444'; return'#991b1b'; }
function metric(v:number|null){ return v == null || Number.isNaN(v) ? '—' : v.toFixed(1); }
function time(v:string){ return new Date(v).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }
function insight(aqi:number, weather:Weather|null){
  const band=aqiBand(aqi).toLowerCase();
  if(!weather) return `Air quality is currently ${band}. The AI model is combining station observations, graph relationships and meteorological features.`;
  const w=weather.current;
  const wind=w.wind_kmh;
  const heat=w.temperature_c;
  let note = wind < 8 ? 'Low wind may allow pollutants to linger.' : 'Moderate airflow may help disperse pollutants.';
  if(heat >= 32) note += ' Higher temperatures can increase outdoor discomfort.';
  return `AI insight: Air quality is ${band} at an average AQI of ${aqi.toFixed(0)}. ${note} Current conditions are ${heat.toFixed(0)}°C with ${w.humidity_pct.toFixed(0)}% humidity.`;
}

export default function App(){
 const [cities,setCities]=useState<string[]>([]),[others,setOthers]=useState<string[]>([]),[city,setCity]=useState('');
 const [current,setCurrent]=useState<Current|null>(null),[forecast,setForecast]=useState<Forecast|null>(null),[weather,setWeather]=useState<Weather|null>(null);
 const [stations,setStations]=useState<Station[]>([]),[loading,setLoading]=useState(false),[error,setError]=useState('');
 const [other,setOther]=useState(''),[expansion,setExpansion]=useState<Expansion|null>(null),[expanding,setExpanding]=useState(false),[expansionError,setExpansionError]=useState('');
 const [lastUpdated,setLastUpdated]=useState('');

 useEffect(()=>{(async()=>{try{const [a,b]=await Promise.all([axios.get(`${API_BASE}/cities`),axios.get(`${API_BASE}/other-cities`)]);const list=a.data.available_cities?.length?a.data.available_cities:a.data.metro_cities;setCities(list);setCity(list[0]||'');setOthers(b.data.other_cities||[]);}catch(e:any){setError(e?.response?.data?.detail||e.message||'Could not load cities');}})();},[]);
 useEffect(()=>{if(!city)return;let cancelled=false;(async()=>{setLoading(true);setError('');setExpansion(null);try{const path=encodeURIComponent(city);const [s,c,f,w]=await Promise.all([axios.get(`${API_BASE}/city/${path}/stations`),axios.get(`${API_BASE}/city/${path}/current-aqi`),axios.get(`${API_BASE}/city/${path}/forecast`,{params:{hours:24}}),axios.get(`${API_BASE}/city/${path}/weather`,{params:{hours:24}})]);if(cancelled)return;setStations(s.data.stations||[]);setCurrent(c.data);setForecast(f.data);setWeather(w.data);setLastUpdated(new Date().toLocaleTimeString());}catch(e:any){if(!cancelled)setError(e?.response?.data?.detail||e.message||'Could not load city data');}finally{if(!cancelled)setLoading(false);}})();return()=>{cancelled=true;};},[city]);
 const center=useMemo<[number,number]>(()=>{const src=current?.stations?.length?current.stations:stations;if(!src.length)return[20.5937,78.9629];return[src.reduce((a,s)=>a+s.lat,0)/src.length,src.reduce((a,s)=>a+s.lon,0)/src.length];},[current,stations]);
 const runExpansion=async()=>{if(!other)return;setExpanding(true);setExpansionError('');try{const r=await axios.post(`${API_BASE}/city/${encodeURIComponent(other)}/download-and-run`);setExpansion(r.data);setCities(p=>p.includes(r.data.city)?p:[...p,r.data.city].sort());setCity(r.data.city);}catch(e:any){setExpansionError(e?.response?.data?.detail||e.message||'Expansion failed');}finally{setExpanding(false);}};
 const chartData=forecast?.forecast||[];
 const maxForecast=Math.max(...chartData.map(x=>x.city_avg_aqi),current?.city_avg_aqi||0);

 return <div className="app-shell">
  <header className="topbar"><div className="brand"><div className="brand-mark">AQ</div><div><div className="eyebrow">INTELLIGENT AIR MONITOR</div><h1>AirSight</h1></div></div><div className="top-actions"><span className="live-dot"><i/> Live data</span><select value={city} onChange={e=>setCity(e.target.value)} disabled={loading||!cities.length}>{cities.map(c=><option key={c}>{c}</option>)}</select></div></header>
  <main className="content">
   <section className="hero"><div><span className="pill">AI + LIVE WEATHER</span><h2>Know the air.<br/><span>Plan your day.</span></h2><p>Graph neural predictions meet live meteorological conditions to give you a clearer picture of urban air quality.</p></div><div className="hero-meta"><div><strong>{lastUpdated||'—'}</strong><small>Last refresh</small></div><div><strong>OpenAQ</strong><small>Station observations</small></div><div><strong>Open-Meteo</strong><small>Live forecast</small></div></div></section>
   {error&&<div className="alert">{error}</div>}
   {current&&<section className="stats-grid">
    <article className="aqi-card"><div className="card-top"><span>City average AQI</span><span className="status">{aqiBand(current.city_avg_aqi)}</span></div><div className="aqi-number" style={{color:aqiColor(current.city_avg_aqi)}}>{current.city_avg_aqi.toFixed(0)}</div><div className="scale"><span style={{width:`${Math.min(100,current.city_avg_aqi/5)}%`,background:aqiColor(current.city_avg_aqi)}}/></div><p>AI prediction across {current.stations.length} monitored stations.</p></article>
    <article className="weather-card"><div className="card-top"><span>Live weather</span><span>Open-Meteo</span></div><div className="weather-main"><div className="temp">{weather?.current.temperature_c.toFixed(0) ?? '—'}°</div><div><strong>Current conditions</strong><p>{weather?.current.humidity_pct.toFixed(0)}% humidity · {weather?.current.wind_kmh.toFixed(0)} km/h wind</p></div></div><div className="mini-forecast">{weather?.forecast.slice(0,6).map(x=><div key={x.timestamp}><b>{time(x.timestamp)}</b><span>{x.temperature_c.toFixed(0)}°</span></div>)}</div></article>
    <article className="insight-card"><div className="ai-icon">✦</div><div><div className="card-label">AI AIR INSIGHT</div><h3>What does it mean?</h3><p>{insight(current.city_avg_aqi,weather)}</p></div></article>
   </section>}

   <section className="dashboard-grid">
    <article className="panel map-panel"><div className="panel-head"><div><span className="card-label">SPATIAL VIEW</span><h3>Station intelligence</h3></div><span className="muted">{stations.length} stations</span></div><div className="map-wrap">{loading?<div className="loading">Updating live station data…</div>:<MapContainer center={center} zoom={11} scrollWheelZoom><TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/>{current?.stations.map(s=><CircleMarker key={s.location_id} center={[s.lat,s.lon]} radius={9} pathOptions={{color:'#fff',weight:2,fillColor:aqiColor(s.predicted_aqi),fillOpacity:.9}}><Popup><b>{s.station_name}</b><br/>AQI {s.predicted_aqi.toFixed(0)} · {aqiBand(s.predicted_aqi)}</Popup></CircleMarker>)}</MapContainer>}</div></article>
    <article className="panel forecast-panel"><div className="panel-head"><div><span className="card-label">NEXT 24 HOURS</span><h3>Predicted AQI trend</h3></div><span className="trend-badge">Peak {maxForecast.toFixed(0)}</span></div><div className="chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData}><defs><linearGradient id="aqiFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopOpacity={.28}/><stop offset="100%" stopOpacity={0}/></linearGradient></defs><CartesianGrid vertical={false} stroke="#e7eaf0"/><XAxis dataKey="timestamp" tickFormatter={time} minTickGap={28} stroke="#7c8495"/><YAxis stroke="#7c8495" width={35}/><Tooltip labelFormatter={v=>new Date(v).toLocaleString()} formatter={(v:any)=>[Number(v).toFixed(1),'AQI']}/><Area type="monotone" dataKey="city_avg_aqi" stroke="#111827" fill="url(#aqiFill)" strokeWidth={2.5}/></AreaChart></ResponsiveContainer></div></article>
   </section>

   <section className="panel stations-panel"><div className="panel-head"><div><span className="card-label">MONITORED NETWORK</span><h3>Station breakdown</h3></div><span className="muted">Live model output</span></div><div className="station-grid">{current?.stations.map(s=><div className="station" key={s.location_id}><div className="station-title"><div className="station-dot" style={{background:aqiColor(s.predicted_aqi)}}/><div><strong>{s.station_name}</strong><small>ID {s.location_id}</small></div><div className="station-aqi" style={{color:aqiColor(s.predicted_aqi)}}>{s.predicted_aqi.toFixed(0)}<small>{aqiBand(s.predicted_aqi)}</small></div></div><div className="metrics"><span>PM2.5 <b>{metric(s.features.pm25)}</b></span><span>PM10 <b>{metric(s.features.pm10)}</b></span><span>NO₂ <b>{metric(s.features.no2)}</b></span><span>SO₂ <b>{metric(s.features.so2)}</b></span><span>CO <b>{metric(s.features.co)}</b></span><span>O₃ <b>{metric(s.features.o3)}</b></span></div></div>)}</div></section>

   <section className="expand panel"><div><span className="card-label">EXPAND COVERAGE</span><h3>Analyze another Indian city</h3><p>Discover nearby OpenAQ stations, build a city graph and run the trained GCN on demand.</p></div><div className="expand-controls"><select value={other} onChange={e=>setOther(e.target.value)}><option value="">Choose a city</option>{others.map(c=><option key={c}>{c}</option>)}</select><button onClick={runExpansion} disabled={!other||expanding}>{expanding?'Analyzing…':'Analyze city'}</button></div>{expansion&&<div className="success">{expansion.city} ready · {expansion.station_count} stations · {expansion.graph.edges} graph edges</div>}{expansionError&&<div className="error-text">{expansionError}</div>}</section>
  </main><footer>AirSight · GCN AQI Engine · Built with React, FastAPI, OpenAQ & Open-Meteo</footer>
 </div>
}
