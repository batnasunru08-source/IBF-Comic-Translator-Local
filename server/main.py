from __future__ import annotations

from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.pipeline import process_image_bytes

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Comic Translator API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateFromUrlRequest(BaseModel):
    image_url: str = Field(..., description="Прямой URL исходного изображения")
    page_url: str = Field(default="", description="URL страницы, где размещено изображение")
    referer: str = Field(default="", description="Referer для скачивания изображения")
    source_ocr_lang: str = Field(default="en", description="Язык OCR")
    target_lang: str = Field(default="Russian", description="Язык перевода")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/translate-from-url")
def translate_from_url(payload: TranslateFromUrlRequest, request: Request):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )
    }
    if payload.referer:
        headers["Referer"] = payload.referer

    try:
        response = requests.get(payload.image_url, headers=headers, timeout=40)
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось скачать изображение: {exc}") from exc

    out_path, meta = process_image_bytes(
        response.content,
        RESULTS_DIR,
        source_ocr_lang=payload.source_ocr_lang,
        target_lang=payload.target_lang,
    )
    return {
        "ok": True,
        "result_url": str(request.base_url).rstrip("/") + f"/results/{out_path.name}",
        "meta": meta,
    }


@app.post("/translate-upload")
async def translate_upload(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    out_path, meta = process_image_bytes(content, RESULTS_DIR)
    base_url = str(request.base_url).rstrip("/")
    return {
        "ok": True,
        "result_url": f"{base_url}/results/{out_path.name}",
        "meta": meta,
    }


@app.get("/results/{filename}")
def get_result(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Результат не найден")
    return FileResponse(path)