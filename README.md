---
title: GeoGuide
emoji: 🧭
sdk: docker
app_port: 7860
---
# GeoGuide: place + time outdoor assistant

`POST /api/ask` with `{"question": "...", "location": "optional"}`. The API geocodes the place, pulls live
Open-Meteo weather + air quality, finds nearby parks/reserves/peaks from OpenStreetMap, ranks the best 2-hour
windows in the next 48 h, and has Llama-3.2-3B-Instruct write the grounded answer.
`GET /health` for uptime checks, `GET /` serves the website.

## Deploy on Hugging Face Spaces
1. Accept the Llama 3.2 license at huggingface.co/meta-llama/Llama-3.2-3B-Instruct, then create a token (Read).
2. New Space, SDK: Docker. Upload all files in this folder (or push this repo to the Space).
3. Space Settings, Secrets: add `HF_TOKEN`. Optional variables: `LLM_MODEL`, `LLM_BASE_URL`, `DEFAULT_LOCATION`.
4. Endpoint: `https://<user>-<space>.hf.space/api/ask`
5. Keep it awake for judging: ping `/health` every 5 min (UptimeRobot). Free Spaces sleep when idle.

## Run locally with Ollama (no HF)
    ollama pull llama3.2:3b
    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2:3b uvicorn app:app --port 7860

## Test
    curl -X POST http://localhost:7860/api/ask -H "Content-Type: application/json" \
      -d '{"question":"I have mild asthma. Where is a good place for a walk near Powai in the next two days?","location":"Powai, Mumbai"}'

Check `llm_used` is `true` in the response: if `false`, the model call failed and `llm_error` says why.
