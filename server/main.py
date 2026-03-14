from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
    module=r"requests(\..*)?",
)

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.ocr import get_paddleocr_engine
from app.pipeline import process_image_bytes

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DOWNLOAD_TIMEOUT = (5, 10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Preloading translation model...")
    try:
        from app.translator import get_translator

        translator = get_translator()
        _ = translator.translate_batch(["Hello"], target_language="Russian")
        print("[STARTUP] Translator warm-up complete")
    except Exception as exc:
        print(f"[STARTUP] Translator warm-up failed: {exc}")

    print("[STARTUP] Preloading PaddleOCR readers...")
    try:
        get_paddleocr_engine("en")
        print("[STARTUP] PaddleOCR en ready")
    except Exception as exc:
        print(f"[STARTUP] PaddleOCR preload failed: {exc}")

    yield

    print("[SHUTDOWN] Server is stopping")


app = FastAPI(title="Comic Translator API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateFromUrlRequest(BaseModel):
    image_url: str = Field(..., description="Direct URL of the source image")
    page_url: str = Field(default="", description="Page URL hosting the image")
    referer: str = Field(default="", description="Referer used for image download")
    source_ocr_lang: str = Field(default="en", description="OCR language")
    target_lang: str = Field(default="Russian", description="Translation target language")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/translate-from-url")
def translate_from_url(payload: TranslateFromUrlRequest, request: Request):
    headers = {
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )
    }
    referer = (payload.referer or payload.page_url or "").strip()
    if referer:
        headers["Referer"] = referer
        origin = urlsplit(referer)
        if origin.scheme and origin.netloc:
            headers["Origin"] = f"{origin.scheme}://{origin.netloc}"

    try:
        response = requests.get(payload.image_url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:
        print(
            f"[DOWNLOAD] failed image_url={payload.image_url!r} "
            f"referer={referer!r} error={exc}"
        )
        raise HTTPException(status_code=400, detail=f"Failed to download image: {exc}") from exc

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
async def translate_upload(
    request: Request,
    file: UploadFile = File(...),
    source_ocr_lang: str = Form("en"),
    target_lang: str = Form("Russian"),
):
    content = await file.read()
    out_path, meta = process_image_bytes(
        content,
        RESULTS_DIR,
        source_ocr_lang=source_ocr_lang,
        target_lang=target_lang,
    )
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
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path)
