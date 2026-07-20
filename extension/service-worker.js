const DEFAULT_API_BASE = "http://127.0.0.1:8000";
let API_BASE = DEFAULT_API_BASE;
const MAX_UPLOAD_DIMENSION = 2200;
const SCREENSHOT_SETTLE_MS = 200;
const MAX_CACHE_BYTES = 50 * 1024 * 1024; // 50MB на все кеши dataUrl
const MAX_CACHE_ENTRIES = 64;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const TRANSLATION_CACHE = new Map();
const RESULT_DATA_URL_CACHE = new Map();
let resultCacheBytes = 0;

function normalizeApiBase(raw) {
  const trimmed = String(raw ?? "").trim().replace(/\/+$/, "");
  if (!trimmed) return DEFAULT_API_BASE;
  try {
    const url = new URL(trimmed);
    if (url.protocol !== "http:" && url.protocol !== "https:") return DEFAULT_API_BASE;
    return `${url.protocol}//${url.host}`;
  } catch {
    return DEFAULT_API_BASE;
  }
}

chrome.storage.local.get({ apiBase: DEFAULT_API_BASE }).then(({ apiBase }) => {
  API_BASE = normalizeApiBase(apiBase);
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local" || !changes.apiBase) return;
  API_BASE = normalizeApiBase(changes.apiBase.newValue);
});

function roundMs(value) {
  return Math.round(Number(value || 0));
}

function estimateDataUrlBytes(value) {
  if (typeof value === "string") return value.length;
  if (value && typeof value === "object") {
    if (typeof value.dataUrl === "string") return value.dataUrl.length;
    if (typeof value.size === "number") return value.size;
  }
  return 0;
}

function rememberCacheValue(cache, key, value) {
  if (cache.has(key)) {
    const old = cache.get(key);
    resultCacheBytes -= estimateDataUrlBytes(old);
    cache.delete(key);
  }
  const size = estimateDataUrlBytes(value);
  while (cache.size >= MAX_CACHE_ENTRIES ||
         (resultCacheBytes + size > MAX_CACHE_BYTES && cache.size > 0)) {
    const oldestKey = cache.keys().next().value;
    if (!oldestKey) break;
    const oldestValue = cache.get(oldestKey);
    resultCacheBytes -= estimateDataUrlBytes(oldestValue);
    cache.delete(oldestKey);
  }
  cache.set(key, value);
  resultCacheBytes += size;
}

function dataUrlToBlob(dataUrl) {
  const match = /^data:([^;,]+)?(?:;base64)?,(.*)$/i.exec(String(dataUrl || ""));
  if (!match) throw new Error("invalid data URL");
  const mime = match[1] || "application/octet-stream";
  const binary = atob(match[2] || "");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function rememberCacheValue(cache, key, value) {
  if (cache.has(key)) cache.delete(key);
  cache.set(key, value);
  while (cache.size > MAX_CACHE_ENTRIES) cache.delete(cache.keys().next().value);
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
    try { data = JSON.parse(text); }
    catch { data = { detail: text }; }
  }
  if (!response.ok) {
    const error = data?.detail ?? data?.error ?? `HTTP ${response.status}`;
    return { ok: false, error, status: response.status };
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
  const requestOptions = {
    method: "GET",
    referrer: message.pageUrl || undefined,
    referrerPolicy: "strict-origin-when-cross-origin",
    headers: { Accept: "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8" }
  };

  let lastError = null;
  for (const cacheMode of ["force-cache", "default"]) {
    try {
      const response = await fetch(message.imageUrl, { ...requestOptions, cache: cacheMode });
      if (!response.ok) throw new Error(`source image fetch failed: HTTP ${response.status}`);
      const blob = await response.blob();
      if (!blob || blob.size === 0) throw new Error("source image fetch returned an empty blob");
      return { blob, cacheMode };
    } catch (error) {
      lastError = error;
      if (cacheMode === "force-cache") {
        console.info("[Comic Translator] source fetch via browser cache failed, retrying network fetch:", {
          imageUrl: message.imageUrl, reason: String(error)
        });
      }
    }
  }
  throw lastError || new Error("source image fetch failed");
}

async function captureVisibleImageBlob(sender, message) {
  const capture = message.visibleCapture;
  if (!capture) throw new Error("visible capture info is missing");
  if (!chrome.tabs?.captureVisibleTab) throw new Error("chrome.tabs.captureVisibleTab is unavailable");

  const windowId = sender?.tab?.windowId;
  await delay(SCREENSHOT_SETTLE_MS);
  const screenshotDataUrl = await chrome.tabs.captureVisibleTab(windowId, { format: "png" });
  const screenshotBlob = dataUrlToBlob(screenshotDataUrl);
  const bitmap = await createImageBitmap(screenshotBlob);

  try {
    const { devicePixelRatio, left = 0, top = 0, width = 0, height = 0 } = capture;
    const dpr = Math.max(1, Number(devicePixelRatio ?? 1));
    const sx = Math.max(0, Math.floor(Number(left) * dpr));
    const sy = Math.max(0, Math.floor(Number(top) * dpr));
    const sw = Math.max(1, Math.floor(Number(width) * dpr));
    const sh = Math.max(1, Math.floor(Number(height) * dpr));
    const cropRight = Math.min(bitmap.width, sx + sw);
    const cropBottom = Math.min(bitmap.height, sy + sh);
    const cropWidth = cropRight - sx;
    const cropHeight = cropBottom - sy;
    if (cropWidth < 1 || cropHeight < 1) throw new Error("visible capture rectangle is outside the screenshot bounds");

    const canvas = new OffscreenCanvas(cropWidth, cropHeight);
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("failed to create crop canvas");
    context.drawImage(bitmap, sx, sy, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

    const blob = await canvas.convertToBlob({ type: "image/png" });
    if (!blob || blob.size === 0) throw new Error("visible capture produced an empty blob");
    return blob;
  } finally {
    bitmap.close?.();
  }
}

async function loadSourceBlob(sender, message, timings) {
  const { domImageBuffer, domImageDataUrl, domImageInfo, visibleCapture } = message;
  const { mime, elapsedMs = 0, bytes = 0 } = domImageInfo ?? {};

  if (domImageBuffer) {
    const t0 = performance.now();
    const blob = new Blob([domImageBuffer], { type: mime ?? "image/png" });
    Object.assign(timings, {
      domCaptureMs: Number(elapsedMs),
      domDecodeMs: performance.now() - t0,
      sourceKind: "dom",
      sourceBytes: Number(bytes || blob.size)
    });
    return blob;
  }

  if (domImageDataUrl) {
    // Backward-compat со старыми версиями расширения
    const t0 = performance.now();
    const blob = dataUrlToBlob(domImageDataUrl);
    Object.assign(timings, {
      domCaptureMs: Number(elapsedMs),
      domDecodeMs: performance.now() - t0,
      sourceKind: "dom",
      sourceBytes: Number(bytes || blob.size)
    });
    return blob;
  }

  if (visibleCapture) {
    const t0 = performance.now();
    const blob = await captureVisibleImageBlob(sender, message);
    Object.assign(timings, {
      screenshotCaptureMs: performance.now() - t0,
      sourceKind: "screenshot",
      sourceBytes: blob?.size ?? 0
    });
    return blob;
  }

  const t0 = performance.now();
  const { blob, cacheMode } = await fetchSourceImageBlob(message);
  Object.assign(timings, {
    sourceFetchMs: performance.now() - t0,
    sourceKind: "network",
    sourceBytes: blob?.size ?? 0,
    sourceFetchMode: cacheMode
  });
  return blob;
}

async function maybeResizeImageBlob(blob) {
  if (!blob || blob.size === 0 || typeof createImageBitmap !== "function" ||
      typeof OffscreenCanvas === "undefined" || blob.type === "image/gif" || blob.type === "image/svg+xml") {
    if (blob?.type === "image/gif") {
      console.warn("[Comic Translator] GIF detected — translating only the first frame, animation will be lost");
    }
    return blob;
  }

  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
    const maxSide = Math.max(bitmap.width, bitmap.height);
    if (maxSide <= MAX_UPLOAD_DIMENSION) return blob;

    const scale = MAX_UPLOAD_DIMENSION / maxSide;
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return blob;

    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, width, height);

    // Сохраняем PNG и WebP без изменений при ресайзе — иначе теряется
    // альфа-канал и OCR получает чёрный фон вместо прозрачности.
    const preserveTypes = new Set(["image/webp", "image/png", "image/avif"]);
    const targetType = preserveTypes.has(blob.type) ? blob.type : "image/jpeg";
    const opts = targetType === "image/jpeg" ? { quality: 0.92 } : {};
    const resizedBlob = await canvas.convertToBlob({ type: targetType, ...opts });
    if (!resizedBlob || resizedBlob.size >= blob.size) return blob;

    console.log("[Comic Translator] resized upload blob:",
      `${bitmap.width}x${bitmap.height} -> ${width}x${height}`,
      `${blob.size} -> ${resizedBlob.size}`);
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
  formData.append("file", blob, `source.${subtype}`);
  formData.append("source_ocr_lang", message.sourceOcrLang || "en");
  formData.append("target_lang", message.targetLang || "Russian");

  const response = await fetch(`${API_BASE}/translate-upload`, { method: "POST", body: formData });
  return readJsonResponse(response);
}

async function blobToDataUrl(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(arrayBuffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return `data:${blob.type || "image/png"};base64,${btoa(binary)}`;
}

async function fetchResultDataUrl(resultUrl) {
  const cached = RESULT_DATA_URL_CACHE.get(resultUrl);
  if (cached) return cached;

  const response = await fetch(resultUrl, { method: "GET" });
  if (!response.ok) throw new Error(`result fetch failed: HTTP ${response.status}`);

  const blob = await response.blob();
  const dataUrl = await blobToDataUrl(blob);
  rememberCacheValue(RESULT_DATA_URL_CACHE, resultUrl, dataUrl);
  return dataUrl;
}

// ---------------------------------------------------------------------------
// Основная логика перевода — вынесена отдельно, используется и портом и
// старым onMessage-хендлером для "fetch-image-as-data-url"
// ---------------------------------------------------------------------------
async function handleTranslateImage(message, sender) {
  const totalStart = performance.now();
  const cacheKey = makeTranslationCacheKey(message);
  const cachedMeta = TRANSLATION_CACHE.get(cacheKey);
  if (cachedMeta) {
    // DataUrl берём из отдельного кеша, чтобы не дублировать ~2MB в двух местах.
    const dataUrl = cachedMeta.result_url ? RESULT_DATA_URL_CACHE.get(cachedMeta.result_url) : null;
    if (dataUrl) {
      console.log("[Comic Translator] translate-image cache hit:", { imageUrl: message.imageUrl });
      return { ...cachedMeta, dataUrl };
    }
  }

  let data;
  const timings = {};

  try {
    const sourceBlob = await loadSourceBlob(sender, message, timings);

    const resizeStart = performance.now();
    const uploadBlob = await maybeResizeImageBlob(sourceBlob);
    timings.resizeMs = performance.now() - resizeStart;

    const uploadStart = performance.now();
    data = await requestTranslateUpload(message, uploadBlob);
    timings.serverRequestMs = performance.now() - uploadStart;
    timings.uploadBytes = uploadBlob?.size ?? 0;
  } catch (uploadError) {
    console.warn("[Comic Translator] upload-first path failed, falling back to translate-from-url:", uploadError);
    timings.uploadFirstError = String(uploadError);
    const fallbackStart = performance.now();
    data = await requestTranslateFromUrl(message);
    timings.serverRequestMs = performance.now() - fallbackStart;
  }

  if (data?.ok && data.result_url) {
    try {
      const resultFetchStart = performance.now();
      data.dataUrl = await fetchResultDataUrl(data.result_url);
      timings.resultFetchMs = performance.now() - resultFetchStart;
    } catch (resultError) {
      console.warn("[Comic Translator] eager result fetch failed:", resultError);
      timings.resultFetchError = String(resultError);
    }

    timings.totalMs = performance.now() - totalStart;
    data.client_timings = Object.fromEntries(
      Object.entries(timings).map(([k, v]) => [k, typeof v === "number" ? roundMs(v) : v])
    );
    // В TRANSLATION_CACHE кладём метаданные (без dataUrl) — dataUrl живёт
    // отдельно в RESULT_DATA_URL_CACHE, не дублируем ~2MB.
    const { dataUrl: _unused, ...metaOnly } = data;
    rememberCacheValue(TRANSLATION_CACHE, cacheKey, metaOnly);
  }

  console.log("[Comic Translator] translate response:", data);
  return data;
}

// ---------------------------------------------------------------------------
// Контекстное меню — правый клик по картинке → "Translate with Comic Translator"
// ---------------------------------------------------------------------------
function getContextMenuTitle() {
  try { return chrome.i18n.getMessage("context_translate") ?? "Translate with Comic Translator"; }
  catch { return "Translate with Comic Translator"; }
}

chrome.runtime.onInstalled.addListener(() => {
  try {
    chrome.contextMenus.create({
      id: "comic-translator-image",
      title: getContextMenuTitle(),
      contexts: ["image"]
    });
  } catch (error) {
    console.warn("[Comic Translator] context menu create failed:", error);
  }
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== "comic-translator-image" || !info.srcUrl || !tab?.id) return;
  chrome.tabs.sendMessage(tab.id, {
    type: "translate-image-by-url",
    imageUrl: info.srcUrl,
    pageUrl: tab.url || ""
  }).catch((error) => {
    console.info("[Comic Translator] content script did not handle context menu click:", String(error));
  });
});

// ---------------------------------------------------------------------------
// Порт-based коммуникация — решает проблему "message channel closed"
// Service worker остаётся живым пока открыт порт (приоритет: всегда активен)
// ---------------------------------------------------------------------------
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "comic-translator") return;

  port.onMessage.addListener(async (message) => {
    if (message?.type !== "translate-image") return;

    // Периодически пингуем порт чтобы воркер не засыпал во время перевода
    const keepAlive = setInterval(() => {
      try { port.postMessage({ type: "keep-alive" }); }
      catch { clearInterval(keepAlive); }
    }, 20_000);

    try {
      const data = await handleTranslateImage(message, port.sender);
      port.postMessage({ type: "translate-result", data });
    } catch (error) {
      console.error("[Comic Translator] port translate error:", error);
      port.postMessage({ type: "translate-result", data: { ok: false, error: String(error) } });
    } finally {
      clearInterval(keepAlive);
    }
  });
});

// ---------------------------------------------------------------------------
// Старый onMessage — оставляем только для "fetch-image-as-data-url"
// (лёгкая операция, не требует долгого ожидания)
// ---------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "fetch-image-as-data-url") {
    (async () => {
      try {
        const dataUrl = await fetchResultDataUrl(message.url);
        sendResponse({ ok: true, dataUrl });
      } catch (error) {
        console.error("[Comic Translator] fetch-image-as-data-url error:", error);
        sendResponse({ ok: false, error: String(error) });
      }
    })();
    return true;
  }

  if (message?.type === "ping-server") {
    (async () => {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        const response = await fetch(`${API_BASE}/health`, {
          method: "GET",
          signal: controller.signal
        });
        clearTimeout(timeout);
        sendResponse({ ok: response.ok, status: response.status });
      } catch (error) {
        sendResponse({ ok: false, error: String(error) });
      }
    })();
    return true;
  }
});
