"""GeoGuide API: place + time outdoor assistant grounded in live data.
Data: Open-Meteo (geocoding, forecast, air quality) + OpenStreetMap Overpass.
Language model: Llama-3.2-3B-Instruct via any OpenAI-compatible endpoint."""
import asyncio, math, os, re, time
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN", "")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Powai")
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
SOURCES = ["Open-Meteo Geocoding", "Open-Meteo Forecast", "Open-Meteo Air Quality",
           "OpenStreetMap via Overpass API", f"{LLM_MODEL} (language model)"]

app = FastAPI(title="GeoGuide API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    question: str
    location: str | None = None


def pick_location(q: str, loc: str | None) -> str:
    if loc and loc.strip():
        return loc.strip()
    m = re.search(r"\b(?:near|around|in|at)\s+([A-Z][\w .'-]{2,40}?)(?=\s+(?:in|for|during|over|within|today|tomorrow|tonight)\b|[?.,]|$)", q)
    return m.group(1).strip() if m else DEFAULT_LOCATION


def hour_score(t, rain, wind, uv, aqi, sensitive):
    s = 100 - max(0, t - 26) * 4 - max(0, 16 - t) * 3 - (rain or 0) * 0.3
    s -= max(0, wind - 20) * 1.5 + max(0, uv - 5) * 5 + max(0, aqi - 50) * (0.9 if sensitive else 0.5)
    return max(0.0, min(100.0, s))


def best_windows(w, a, sensitive, hours=48):
    now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=w.get("utc_offset_seconds", 0))
    H, A, out = w["hourly"], a["hourly"], []
    for i, ts in enumerate(H["time"][:-1]):
        t0 = datetime.fromisoformat(ts)
        if not (now - timedelta(hours=1) <= t0 < now + timedelta(hours=hours)) or not (5 <= t0.hour <= 18):
            continue
        idx = [i, i + 1]
        g = lambda d, k: [d[k][j] if d[k][j] is not None else 0 for j in idx]
        t, r, wi, uv = g(H, "temperature_2m"), g(H, "precipitation_probability"), g(H, "wind_speed_10m"), g(H, "uv_index")
        aq, pm = g(A, "us_aqi"), g(A, "pm2_5")
        sc = sum(hour_score(t[k], r[k], wi[k], uv[k], aq[k], sensitive) for k in (0, 1)) / 2
        out.append({"start": ts.replace("T", " "), "end": H["time"][i + 2].replace("T", " ") if i + 2 < len(H["time"]) else ts,
                    "score": round(sc), "temp_c": round(sum(t) / 2, 1), "rain_prob_pct": max(r), "wind_kmh": round(max(wi), 1),
                    "uv_index": round(max(uv), 1), "us_aqi": round(sum(aq) / 2), "pm2_5": round(sum(pm) / 2, 1)})
    out.sort(key=lambda x: -x["score"])
    return out


def haversine(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))


async def find_places(c, lat, lon, hike):
    q = (f'[out:json][timeout:20];(way(around:3000,{lat},{lon})["leisure"~"^(park|nature_reserve)$"]["name"];'
         f'relation(around:3000,{lat},{lon})["leisure"~"^(park|nature_reserve)$"]["name"];'
         f'node(around:4000,{lat},{lon})["natural"="peak"]["name"];);out center tags 60;')
    for url in OVERPASS:
        try:
            r = await c.post(url, data={"data": q}, timeout=25)
            r.raise_for_status()
            seen, places = set(), []
            for el in r.json().get("elements", []):
                tags, name = el.get("tags", {}), el.get("tags", {}).get("name")
                pt = el if "lat" in el else el.get("center")
                if not name or not pt or name in seen:
                    continue
                seen.add(name)
                km = haversine(lat, lon, pt["lat"], pt["lon"])
                kind = "peak" if tags.get("natural") == "peak" else tags.get("leisure", "park")
                bonus = {"nature_reserve": 8, "park": 4, "peak": 6 if hike else 0}.get(kind, 0)
                places.append({"name": name, "type": kind, "lat": round(pt["lat"], 5), "lon": round(pt["lon"], 5),
                               "distance_km": round(km, 2), "est_walk_min": round(km * 1.3 / 4.5 * 60),
                               "place_score": round(max(0, 100 - km * 12 + bonus))})
            places.sort(key=lambda p: -p["place_score"])
            return places[:4]
        except Exception:
            continue
    return []


def flags_for(win, sensitive):
    f = []
    if win["us_aqi"] > 100:
        f.append("Air quality is unhealthy for sensitive groups; keep exertion light." if sensitive else "Air quality is moderate-to-poor; reduce strenuous exercise.")
    if sensitive and win["us_aqi"] > 50:
        f.append("Carry any prescribed reliever medication.")
    if win["us_aqi"] > 150:
        f.append("AQI above 150: avoid outdoor exercise.")
    if win["uv_index"] >= 8:
        f.append("Very high UV: sun protection needed.")
    if win["temp_c"] > 32:
        f.append("Hot: carry water, avoid midday.")
    if win["rain_prob_pct"] >= 50:
        f.append("Rain likely: carry rain gear.")
    return f


async def ask_llm(c, system, user):
    h = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    r = await c.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", headers=h, timeout=90,
                     json={"model": LLM_MODEL, "max_tokens": 350, "temperature": 0.2,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


SYSTEM = ("You are GeoGuide, an assistant that helps people decide WHERE to go outdoors and WHEN. "
          "Use ONLY the FACTS JSON. Quote numbers exactly; never invent places, numbers or times. "
          "If candidate_places is empty, say no named places were found nearby and advise on timing only. "
          "Reply in under 170 words: (1) recommendation: place + time window, (2) why, with numbers, "
          "(3) precautions from caution_flags. Note that walk times are estimates. This is not medical advice.")


@app.post("/api/ask")
async def api_ask(body: Ask):
    t0 = time.time()
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "question is required")
    loc_text = pick_location(q, body.location)
    sensitive = bool(re.search(r"asthma|copd|allerg|respirator|child|elderly|pregnan|heart", q, re.I))
    hike = bool(re.search(r"hike|trek|climb|view|peak", q, re.I))
    async with httpx.AsyncClient(timeout=25, headers={"User-Agent": "GeoGuide/1.0 (hackathon)"}) as c:
        g = await c.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": loc_text.split(",")[0].strip(), "count": 1})
        res = g.json().get("results")
        if not res:
            raise HTTPException(404, f"Could not find location '{loc_text}'")
        lat, lon, name = res[0]["latitude"], res[0]["longitude"], res[0].get("name", loc_text)
        wr, ar, places = await asyncio.gather(
            c.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
                  "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,uv_index"}),
            c.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={"latitude": lat, "longitude": lon, "timezone": "auto",
                  "forecast_days": 3, "hourly": "us_aqi,pm2_5"}),
            find_places(c, lat, lon, hike))
        wins = best_windows(wr.json(), ar.json(), sensitive)
        if not wins:
            raise HTTPException(502, "No forecast data available for the next 48 hours")
        facts = {"location": name, "sensitive_to_air_quality": sensitive, "best_windows": wins[:3], "worst_window": wins[-1],
                 "candidate_places": places, "caution_flags": flags_for(wins[0], sensitive),
                 "note": "est_walk_min = straight-line distance x1.3 at 4.5 km/h, not routed"}
        user = f"Question: {q}\nFACTS: {facts}"
        try:
            answer, used, err = await ask_llm(c, SYSTEM, user), True, None
        except Exception as e:
            b = wins[0]
            p = f" at {places[0]['name']} ({places[0]['distance_km']} km)" if places else ""
            answer = (f"[LLM unavailable, deterministic summary] Best window: {b['start']} to {b['end']}{p}: {b['temp_c']}C, "
                      f"wind {b['wind_kmh']} km/h, US AQI {b['us_aqi']}, UV {b['uv_index']}. " + " ".join(facts["caution_flags"]))
            used, err = False, f"{type(e).__name__}: {e}"
    return {"answer": answer, "model": LLM_MODEL if used else None, "llm_used": used, "llm_error": err,
            "evidence": facts, "data_sources": SOURCES, "latency_ms": round((time.time() - t0) * 1000)}


@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL}


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
