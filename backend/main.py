from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys

sys.path.append(os.path.dirname(__file__))
from models.crop_model import CropRecommender
from routes.chat import router as chat_router

app = FastAPI(title="Smart Farming Decision Support System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = CropRecommender()

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(os.path.join(frontend_path, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "static")), name="static")

app.include_router(chat_router, prefix="/api")

class SoilData(BaseModel):
    # Depth 1 (0-0.6 ft)
    ph_d1: float
    ec_d1: float
    oc_d1: float
    n_d1: float
    p2o5_d1: float
    k2o_d1: float
    cec_d1: float
    # Depth 2 (0.6-1 ft)
    ph_d2: float
    ec_d2: float
    oc_d2: float
    n_d2: float
    p2o5_d2: float
    k2o_d2: float
    cec_d2: float
    # Additional
    temperature: float
    moisture: float

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(frontend_path, "static", "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    return RedirectResponse(url="/static/favicon.svg")

@app.get("/")
async def serve_index():
    index_path = os.path.join(frontend_path, "templates", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Smart Farming API is running"}

@app.get("/analyze")
async def serve_analyze():
    path = os.path.join(frontend_path, "templates", "analyze.html")
    return FileResponse(path)

@app.get("/assistant")
async def serve_assistant():
    path = os.path.join(frontend_path, "templates", "assistant.html")
    return FileResponse(path)

@app.post("/api/recommend")
async def recommend_crop(data: SoilData):
    try:
        result = recommender.predict(data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "Smart Farming API is running"}

@app.get("/api/tts")
async def kannada_tts(text: str, voice: str = "female"):
    """
    Kannada Neural TTS using Microsoft Edge TTS (edge-tts library).

    Voices:
        female  →  kn-IN-SapnaNeural   (default)
        male    →  kn-IN-GaganNeural

    The full text is split into ≤180-char chunks at sentence boundaries,
    each chunk is synthesised in parallel, then the MP3 buffers are
    concatenated in order and returned as a single audio/mpeg response.
    The frontend needs no changes — it still does:
        fetch("/api/tts?text=...") → blob → new Audio(objectURL).play()
    """
    import io
    import re
    import asyncio
    import edge_tts
    from fastapi.responses import Response

    VOICE_MAP = {
        "female": "kn-IN-SapnaNeural",
        "male":   "kn-IN-GaganNeural",
    }
    chosen_voice = VOICE_MAP.get(voice, "kn-IN-SapnaNeural")

    # ── 1. Split text into ≤180-char chunks at sentence boundaries ──────────
    def split_text(raw: str, limit: int = 180) -> list[str]:
        # Normalise whitespace
        raw = raw.strip()
        if not raw:
            return []
        # Split at Kannada (।) and standard sentence-end punctuation
        tokens = re.split(r'(?<=[.।!?\n])\s*', raw)
        chunks, buf = [], ""
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if buf and len(buf) + 1 + len(tok) > limit:
                chunks.append(buf)
                buf = tok
            else:
                buf = (buf + " " + tok).strip() if buf else tok
        if buf:
            chunks.append(buf)
        return chunks or [raw]

    chunks = split_text(text)

    # ── 2. Synthesise each chunk (in parallel, preserve order) ──────────────
    async def synth_chunk(chunk_text: str) -> bytes:
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(chunk_text, chosen_voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    try:
        # asyncio.gather keeps results in submission order
        audio_parts: list[bytes] = await asyncio.gather(
            *[synth_chunk(c) for c in chunks]
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"edge-tts synthesis failed: {e}"
        )

    # ── 3. Concatenate MP3 buffers and return ────────────────────────────────
    # Raw MP3 concatenation is valid — decoders handle multiple ID3/frame streams.
    combined = b"".join(audio_parts)
    if not combined:
        raise HTTPException(status_code=502, detail="edge-tts returned empty audio")

    return Response(
        content=combined,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-TTS-Voice": chosen_voice,
            "X-TTS-Chunks": str(len(chunks)),
        },
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
