"""API FastAPI.  Lancer :  uvicorn app.main:app --reload   puis  http://127.0.0.1:8000/docs"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .asr import ASRUnavailable, transcribe
from .lookup import find_pharmacies
from .pipeline import Assistant
from .respond import build_box_reply
from .schemas import AnalysisResponse, BoxRecognitionResponse, TextRequest
from .vision.ocr import OCRUnavailable, ocr_available
from .vision.preprocess import decode_image
from .vision.recognize import BoxRecognizer

state: dict = {}
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["assistant"] = Assistant()
    state["recognizer"] = BoxRecognizer(state["assistant"].matcher)
    yield
    state.clear()


app = FastAPI(
    title="Assistant pharmacie Darija / Arabe / Français",
    version="0.1.0",
    description="Transforme une demande (texte ou voix) en interprétation structurée : intention, "
                "médicaments, quantité, pharmacie, ville. Données pharmacies/stock : démonstration.",
    lifespan=lifespan,
)


def _assistant() -> Assistant:
    return state["assistant"]


@app.get("/", include_in_schema=False)
def ui():
    """Interface de chat (texte + voix)."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    import importlib.util
    a = _assistant()
    return {"status": "ok", "ocr_available": ocr_available(), "asr_available": importlib.util.find_spec("faster_whisper") is not None, "medications": len(a.store.meds), "pharmacies": len(a.store.pharmacies),
            "intent_mode": a.intents.mode, "ml_model_loaded": a.intents.model is not None}


@app.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(req: TextRequest):
    return _assistant().analyze(req.text, modality="text", do_fulfill=req.fulfill)


@app.post("/analyze/voice", response_model=AnalysisResponse)
async def analyze_voice(file: UploadFile = File(..., description="wav / mp3 / m4a / ogg"),
                        fulfill: bool = Query(True)):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        text, _lang = transcribe(path)
    except ASRUnavailable as e:
        raise HTTPException(status_code=501, detail=str(e))
    finally:
        os.unlink(path)
    if not text:
        raise HTTPException(status_code=422, detail="Aucune parole détectée dans l'audio.")
    return _assistant().analyze(text, modality="voice", transcript=text, do_fulfill=fulfill)


@app.get("/medications/search")
def search_medications(q: str = Query(..., min_length=2), limit: int = Query(5, le=20)):
    """Recherche floue de médicaments (utile pour déboguer le rapprochement)."""
    from .textnorm import make_doc
    a = _assistant()
    doc = make_doc(q)
    out = []
    for m in a.matcher.find(doc, set())[:limit]:
        rows, _ = a.matcher.products(m)
        out.append({"query": m.raw, "name": m.name, "kind": m.kind, "score": m.score, "n_products": len(rows)})
    return out


@app.get("/pharmacies")
def list_pharmacies(city: str | None = None, name: str | None = None, neighborhood: str | None = None,
                    duty_only: bool = False, limit: int = Query(10, le=50)):
    return find_pharmacies(_assistant().store, name, city, neighborhood, duty_only, limit)


@app.post("/recognize/box", response_model=BoxRecognitionResponse)
def recognize_box(file: UploadFile = File(..., description="photo de la boîte (jpg, png, webp)"),
                  top_k: int = Query(3, ge=1, le=10), lang: str = Query("fr", pattern="^(fr|dl|ar)$"),
                  arabic: bool = Query(True, description="lire aussi le texte arabe (un peu plus lent)")):
    """Challenge 2 : reconnaît un médicament à partir d'une photo de sa boîte (vision + OCR arabe/français)."""
    import uuid
    data = file.file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image trop volumineuse (12 Mo max).")
    try:
        img = decode_image(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        res = state["recognizer"].recognize(img, top_k=top_k, use_arabic=arabic)
    except OCRUnavailable as e:
        raise HTTPException(status_code=501, detail=str(e))
    res["request_id"] = str(uuid.uuid4())
    res["reply"] = build_box_reply(res, lang)
    return res
