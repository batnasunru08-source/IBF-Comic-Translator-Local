const API_BASE = "http://127.0.0.1:8000";
const MAX_UPLOAD_DIMENSION = 2200;
const MAX_CACHE_ENTRIES = 64;
const TRANSLATION_CACHE = new Map();
const RESULT_DATA_URL_CACHE = new Map();

function rememberCacheValue(cache, key, value) {
  if (cache.has(key)) {
    cache.delete(key);
  }
  cache.set(key, value);

  while (cache.size > MAX_CACHE_ENTRIES) {
    const oldestKey = cache.keys().next().value;
    cache.delete(oldestKey);
  }
}

function makeTranslationCacheKey(message) {
  return JSON.stringify([
    message.imageUrl || "",
    message.sourceOcrLang || "en",
    message.targetLang || "Russian"
  ]);
}

async function readJsonResponse(response) {
  const text = await response.text();
  let data = {};

  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!response.ok) {
    return {
      ok: false,
      error: data?.detail || data?.error || `HTTP ${response.status}`,
      status: response.status
    };
  }

  return data;
}

async function requestTranslateFromUrl(message) {
  const response = await fetch(`${API_BASE}/translate-from-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_url: message.imageUrl,
      page_url: message.pageUrl || "",
      referer: message.pageUrl || "",
      source_ocr_lang: message.sourceOcrLang || "en",
      target_lang: message.targetLang || "Russian"
    })
  });

  return readJsonResponse(response);
}

async function fetchSourceImageBlob(message) {
  const response = await fetch(message.imageUrl, {
    method: "GET",
    referrer: message.pageUrl || undefined,
    referrerPolicy: "strict-origin-when-cross-origin",
    headers: {
      Accept: "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }
  });

  if (!response.ok) {
    throw new Error(`source image fetch failed: HTTP ${response.status}`);
  }

  return response.blob();
}

async function maybeResizeImageBlob(blob) {
  if (
    !blob ||
    blob.size === 0 ||
    typeof createImageBitmap !== "function" ||
    typeof OffscreenCanvas === "undefined" ||
    blob.type === "image/gif" ||
    blob.type === "image/svg+xml"
  ) {
    return blob;
  }

  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
    const maxSide = Math.max(bitmap.width, bitmap.height);
    if (maxSide <= MAX_UPLOAD_DIMENSION) {
      return blob;
    }

    const scale = MAX_UPLOAD_DIMENSION / maxSide;
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) {
      return blob;
    }

    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, width, height);

    const targetType = blob.type === "image/webp" ? "image/webp" : "image/jpeg";
    const resizedBlob = await canvas.convertToBlob({
      type: targetType,
      quality: 0.92
    });

    if (!resizedBlob || resizedBlob.size >= blob.size) {
      return blob;
    }

    console.log(
      "[Comic Translator] resized upload blob:",
      `${bitmap.width}x${bitmap.height} -> ${width}x${height}`,
      `${blob.size} -> ${resizedBlob.size}`
    );
    return resizedBlob;
  } catch (error) {
    console.warn("[Comic Translator] resize skipped:", error);
    return blob;
  } finally {
    bitmap?.close?.();
  }
}

async function requestTranslateUpload(message, blob) {
  const formData = new FormData();
  const mime = blob.type || "image/png";
  const subtype = (mime.split("/")[1] || "png").split(";")[0];
  const filename = `source.${subtype}`;

  formData.append("file", blob, filename);
  formData.append("source_ocr_lang", message.sourceOcrLang || "en");
  formData.append("target_lang", message.targetLang || "Russian");

  const response = await fetch(`${API_BASE}/translate-upload`, {
    method: "POST",
    body: formData
  });

  return readJsonResponse(response);
}

async function blobToDataUrl(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(arrayBuffer);
  const chunkSize = 0x8000;

  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  const mime = blob.type || "image/png";
  return `data:${mime};base64,${btoa(binary)}`;
}

async function fetchResultDataUrl(resultUrl) {
  const cached = RESULT_DATA_URL_CACHE.get(resultUrl);
  if (cached) {
    return cached;
  }

  const response = await fetch(resultUrl, { method: "GET" });
  if (!response.ok) {
    throw new Error(`result fetch failed: HTTP ${response.status}`);
  }

  const blob = await response.blob();
  const dataUrl = await blobToDataUrl(blob);
  rememberCacheValue(RESULT_DATA_URL_CACHE, resultUrl, dataUrl);
  return dataUrl;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "translate-image") {
    (async () => {
      try {
        const cacheKey = makeTranslationCacheKey(message);
        const cached = TRANSLATION_CACHE.get(cacheKey);
        if (cached) {
          sendResponse(cached);
          return;
        }

        let data;
        try {
          const sourceBlob = await fetchSourceImageBlob(message);
          const uploadBlob = await maybeResizeImageBlob(sourceBlob);
          data = await requestTranslateUpload(message, uploadBlob);
        } catch (uploadError) {
          console.warn(
            "[Comic Translator] upload-first path failed, falling back to translate-from-url:",
            uploadError
          );
          data = await requestTranslateFromUrl(message);
        }

        if (data?.ok && data.result_url) {
          try {
            data.dataUrl = await fetchResultDataUrl(data.result_url);
          } catch (resultError) {
            console.warn("[Comic Translator] eager result fetch failed:", resultError);
          }

          rememberCacheValue(TRANSLATION_CACHE, cacheKey, data);
        }

        console.log("[Comic Translator] translate response:", data);
        sendResponse(data);
      } catch (error) {
        console.error("[Comic Translator] translate error:", error);
        sendResponse({
          ok: false,
          error: String(error)
        });
      }
    })();

    return true;
  }

  if (message?.type === "fetch-image-as-data-url") {
    (async () => {
      try {
        const dataUrl = await fetchResultDataUrl(message.url);
        sendResponse({
          ok: true,
          dataUrl
        });
      } catch (error) {
        console.error("[Comic Translator] fetch-image-as-data-url error:", error);
        sendResponse({
          ok: false,
          error: String(error)
        });
      }
    })();

    return true;
  }
});
