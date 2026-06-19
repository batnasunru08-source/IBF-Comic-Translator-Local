const STATES = new Map();
const PROCESSED_ATTR = "data-comic-translator-bound";
const MIN_SIDE = 80;
const MAX_CAPTURE_DIMENSION = 2200;

const SETTINGS = {
  enabled: true,
  autoTranslate: false,
  renderMode: "replace",
  sourceOcrLang: "en",
  targetLang: "Russian"
};

function getImageUrl(img) {
  return img.currentSrc || img.src || "";
}

function isSupportedImageUrl(rawUrl) {
  if (!rawUrl) return false;

  try {
    const url = new URL(rawUrl, window.location.href);
    const full = `${url.pathname}${url.search}`;
    return /\.(jpe?g|png|webp|gif)(\?|#|$)/i.test(full) || /^https?:/i.test(url.href);
  } catch {
    return /\.(jpe?g|png|webp|gif)(\?|#|$)/i.test(rawUrl) || /^https?:/i.test(rawUrl);
  }
}

function looksLikeImage(img) {
  const src = getImageUrl(img);
  if (!src) return false;

  const width = img.clientWidth || img.naturalWidth || 0;
  const height = img.clientHeight || img.naturalHeight || 0;

  return isSupportedImageUrl(src) && width >= MIN_SIDE && height >= MIN_SIDE;
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
    reader.readAsDataURL(blob);
  });
}

function waitForImageReady(img) {
  if (img.complete && (img.naturalWidth || img.width || img.clientWidth)) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const onLoad = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("image load failed"));
    };
    const cleanup = () => {
      img.removeEventListener("load", onLoad);
      img.removeEventListener("error", onError);
    };

    img.addEventListener("load", onLoad, { once: true });
    img.addEventListener("error", onError, { once: true });
  });
}

function guessCanvasMime(imageUrl) {
  const url = String(imageUrl || "").toLowerCase();
  if (/\.png(\?|#|$)/i.test(url)) return "image/png";
  if (/\.webp(\?|#|$)/i.test(url)) return "image/webp";
  if (/\.jpe?g(\?|#|$)/i.test(url)) return "image/jpeg";
  if (/\.avif(\?|#|$)/i.test(url)) return "image/avif";
  // Нет расширения — по умолчанию PNG, чтобы сохранить альфа-канал.
  return "image/png";
}

function getVisibleCaptureRect(img) {
  const rect = img.getBoundingClientRect();
  if (rect.width < MIN_SIDE || rect.height < MIN_SIDE) {
    return null;
  }

  const fullyVisible =
    rect.left >= 0 &&
    rect.top >= 0 &&
    rect.right <= window.innerWidth &&
    rect.bottom <= window.innerHeight;
  if (!fullyVisible) {
    return null;
  }

  const left = rect.left;
  const top = rect.top;
  const width = rect.width;
  const height = rect.height;

  if (width < MIN_SIDE || height < MIN_SIDE) {
    return null;
  }

  return {
    left,
    top,
    width,
    height,
    devicePixelRatio: window.devicePixelRatio || 1
  };
}

async function captureImageDataFromDom(img) {
  const started = performance.now();

  try {
    await waitForImageReady(img);
    if (typeof img.decode === "function") {
      try {
        await img.decode();
      } catch {
        // Ignore decode errors; drawImage can still succeed for already loaded images.
      }
    }

    const naturalWidth = img.naturalWidth || img.width || img.clientWidth;
    const naturalHeight = img.naturalHeight || img.height || img.clientHeight;
    if (!naturalWidth || !naturalHeight) {
      return null;
    }

    const maxSide = Math.max(naturalWidth, naturalHeight);
    const scale = maxSide > MAX_CAPTURE_DIMENSION ? MAX_CAPTURE_DIMENSION / maxSide : 1;
    const width = Math.max(1, Math.round(naturalWidth * scale));
    const height = Math.max(1, Math.round(naturalHeight * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d", { alpha: false });
    if (!context) {
      return null;
    }

    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(img, 0, 0, width, height);

    const mime = guessCanvasMime(getImageUrl(img));
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        (value) => {
          if (value) {
            resolve(value);
            return;
          }
          reject(new Error("canvas.toBlob returned null"));
        },
        mime,
        mime === "image/png" ? undefined : 0.92
      );
    });

    // Не конвертируем в data URL — отдаём ArrayBuffer для трансфера через порт.
    // base64 data URL для 2200×2200 PNG ≈ 15-20MB и медленно сериализуется.
    const arrayBuffer = await blob.arrayBuffer();
    return {
      arrayBuffer,
      mime,
      width,
      height,
      bytes: blob.size,
      elapsedMs: Math.round(performance.now() - started)
    };
  } catch (error) {
    const reason = `${error?.name || "Error"}: ${error?.message || String(error)}`;
    console.info("[Comic Translator] DOM image capture unavailable, falling back to network fetch:", {
      imageUrl: getImageUrl(img),
      reason
    });
    return null;
  }
}

function t(key, fallback) {
  try {
    const msg = chrome.i18n.getMessage(key);
    if (msg) return msg;
  } catch {}
  return fallback ?? key;
}

function showToast(message, kind) {
  const layer = ensureLayer();
  const toast = document.createElement("div");
  toast.dataset.comicTranslatorToast = "1";
  const isError = kind === "error";
  Object.assign(toast.style, {
    position: "fixed",
    bottom: "24px",
    left: "50%",
    transform: "translateX(-50%)",
    background: isError ? "rgba(220, 50, 47, 0.95)" : "rgba(35, 42, 114, 0.92)",
    color: "rgba(255, 255, 255, 0.96)",
    padding: "10px 16px",
    borderRadius: "10px",
    fontFamily: "Arial, sans-serif",
    fontSize: "13px",
    fontWeight: "600",
    letterSpacing: "0.3px",
    lineHeight: "1.35",
    maxWidth: "min(420px, 80vw)",
    boxShadow: "0 4px 18px rgba(0, 0, 0, 0.25)",
    pointerEvents: "none",
    zIndex: "2147483647",
    opacity: "0",
    transition: "opacity 180ms ease"
  });
  toast.textContent = message;
  layer.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = "1"; });
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 220);
  }, isError ? 5000 : 2200);
}

async function loadSettings() {
  try {
    const { enabled, autoTranslate, renderMode, sourceOcrLang, targetLang } = await chrome.storage.local.get({
      enabled: true,
      autoTranslate: false,
      renderMode: "replace",
      sourceOcrLang: "en",
      targetLang: "Russian"
    });

    SETTINGS.enabled = enabled !== false;
    SETTINGS.autoTranslate = autoTranslate === true;
    SETTINGS.renderMode = renderMode === "overlay" ? "overlay" : "replace";
    SETTINGS.sourceOcrLang = sourceOcrLang ?? "en";
    SETTINGS.targetLang = targetLang ?? "Russian";
  } catch (error) {
    console.error("[Comic Translator] loadSettings error:", error);
  }
}

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "local") return;

  const { enabled, autoTranslate, renderMode, sourceOcrLang, targetLang } = changes;
  if (enabled) SETTINGS.enabled = enabled.newValue !== false;
  if (autoTranslate) {
    SETTINGS.autoTranslate = autoTranslate.newValue === true;
    if (SETTINGS.autoTranslate) scanImages();
  }
  if (renderMode) SETTINGS.renderMode = renderMode.newValue === "overlay" ? "overlay" : "replace";
  if (sourceOcrLang) SETTINGS.sourceOcrLang = sourceOcrLang.newValue ?? "en";
  if (targetLang) SETTINGS.targetLang = targetLang.newValue ?? "Russian";

  applyGlobalState();
});

function ensureLayer() {
  let layer = document.getElementById("comic-translator-layer");
  if (layer) return layer;

  layer = document.createElement("div");
  layer.id = "comic-translator-layer";
  Object.assign(layer.style, {
    position: "fixed",
    inset: "0",
    pointerEvents: "none",
    zIndex: "2147483647"
  });

  if (!document.getElementById("comic-translator-styles")) {
    const style = document.createElement("style");
    style.id = "comic-translator-styles";
    style.textContent = `
      @keyframes ct-spin { to { transform: rotate(360deg); } }
      [data-comic-translator-button].is-loading {
        animation: ct-spin 0.9s linear infinite;
        pointer-events: none;
      }
    `;
    document.documentElement.appendChild(style);
  }

  document.documentElement.appendChild(layer);
  return layer;
}

function getState(img) {
  let state = STATES.get(img);
  if (!state) {
    state = {
      button: null,
      overlay: null,
      clone: null,
      translatedDataUrl: null,
      translationKey: null,
      originalDisplay: null,
      resizeObserver: null
    };
    STATES.set(img, state);
  }
  return state;
}

function makeTranslationKey(imageUrl) {
  return JSON.stringify([imageUrl || "", SETTINGS.sourceOcrLang || "en", SETTINGS.targetLang || "Russian"]);
}

function cleanupState(img) {
  const state = STATES.get(img);
  if (!state) return;

  if (state.button?.isConnected) state.button.remove();
  if (state.overlay?.isConnected) state.overlay.remove();
  if (state.clone?.isConnected) state.clone.remove();
  state.resizeObserver?.disconnect();

  STATES.delete(img);
}

function getOrCreateClone(img) {
  const state = getState(img);

  if (!state.clone?.isConnected) {
    const clone = document.createElement("img");
    clone.dataset.comicTranslatorClone = "1";
    clone.alt = img.alt || "";
    clone.loading = "eager";
    clone.decoding = "async";
    Object.assign(clone.style, {
      display: "none",
      maxWidth: "100%",
      height: "auto"
    });

    img.insertAdjacentElement("afterend", clone);
    state.clone = clone;
  }

  return state.clone;
}

function getOrCreateOverlay(img) {
  const state = getState(img);
  const layer = ensureLayer();

  if (!state.overlay?.isConnected) {
    const overlay = document.createElement("img");
    overlay.dataset.comicTranslatorOverlay = "1";
    Object.assign(overlay.style, {
      position: "fixed",
      display: "none",
      pointerEvents: "none",
      zIndex: "2147483646",
      margin: "0",
      padding: "0",
      border: "0"
    });

    layer.appendChild(overlay);
    state.overlay = overlay;
  }

  return state.overlay;
}

function ensureOriginalVisible(img) {
  const { originalDisplay } = getState(img);
  img.style.display = originalDisplay ?? "";
}

function hideOriginalForReplace(img) {
  const state = getState(img);
  state.originalDisplay ??= img.style.display || "";
  img.style.display = "none";
}

const hideClone = (img) => {
  const { clone } = getState(img);
  if (clone) clone.style.display = "none";
};
const hideOverlay = (img) => {
  const { overlay } = getState(img);
  if (overlay) overlay.style.display = "none";
};
const hideButton = (img) => {
  const { button } = getState(img);
  if (button) button.style.display = "none";
};

function syncCloneGeometry(img, clone) {
  clone.style.width = `${img.clientWidth || img.naturalWidth}px`;
  clone.style.height = "auto";
}

function updateOverlayGeometry(img) {
  const { overlay, translatedDataUrl } = getState(img);
  if (!overlay || !translatedDataUrl) return;

  const rect = img.getBoundingClientRect();
  const offScreen =
    rect.width < MIN_SIDE ||
    rect.height < MIN_SIDE ||
    rect.bottom < 0 ||
    rect.top > window.innerHeight ||
    rect.right < 0 ||
    rect.left > window.innerWidth;

  if (offScreen) {
    overlay.style.display = "none";
    return;
  }

  Object.assign(overlay.style, {
    display: "block",
    left: `${rect.left}px`,
    top: `${rect.top}px`,
    width: `${rect.width}px`,
    height: `${rect.height}px`
  });
}

function applyTranslatedView(img) {
  const { translatedDataUrl } = getState(img);

  if (!SETTINGS.enabled) {
    ensureOriginalVisible(img);
    hideClone(img);
    hideOverlay(img);
    hideButton(img);
    return;
  }

  if (!translatedDataUrl) {
    ensureOriginalVisible(img);
    hideClone(img);
    hideOverlay(img);
    return;
  }

  if (SETTINGS.renderMode === "overlay") {
    ensureOriginalVisible(img);
    hideClone(img);

    const overlay = getOrCreateOverlay(img);
    overlay.src = translatedDataUrl;
    updateOverlayGeometry(img);
    return;
  }

  hideOverlay(img);

  const clone = getOrCreateClone(img);
  clone.src = translatedDataUrl;
  syncCloneGeometry(img, clone);
  clone.style.display = "";

  hideOriginalForReplace(img);
}

function refreshTranslatedViews() {
  for (const [img] of STATES.entries()) {
    if (!img.isConnected) {
      cleanupState(img);
      continue;
    }

    applyTranslatedView(img);
    updateButtonPosition(img);
  }
}

function disableEverything() {
  for (const [img] of STATES.entries()) {
    if (!img.isConnected) {
      cleanupState(img);
      continue;
    }

    ensureOriginalVisible(img);
    hideClone(img);
    hideOverlay(img);
    hideButton(img);
  }
}

function applyGlobalState() {
  if (!SETTINGS.enabled) {
    disableEverything();
    return;
  }

  scanImages();
  refreshTranslatedViews();
}

function createButton(img) {
  const state = getState(img);
  if (state.button && state.button.isConnected) return state.button;

  const layer = ensureLayer();
  const button = document.createElement("button");
  try {
    button.textContent = chrome.i18n.getMessage("button_label") || "IBF";
    button.title = chrome.i18n.getMessage("button_translate") || "Translate image";
  } catch {
    button.textContent = "IBF";
    button.title = "Translate image";
  }

  Object.assign(button.style, {
    position: "fixed",
    minWidth: "44px",
    height: "28px",
    borderRadius: "10px",
    border: "1px solid rgba(60, 113, 122, 0.22)",
    background: "rgba(35, 42, 114, 0.28)",
    color: "rgba(255,255,255,0.92)",
    boxShadow: "0 2px 10px rgba(0,0,0,0.18)",
    cursor: "pointer",
    pointerEvents: "auto",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "13px",
    fontWeight: "700",
    letterSpacing: "0.6px",
    lineHeight: "1",
    padding: "0 10px",
    backdropFilter: "blur(2px)",
    transition: "transform 0s"
  });
  button.dataset.comicTranslatorButton = "1";

  button.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!SETTINGS.enabled) return;

    const imageUrl = getImageUrl(img);
    if (!imageUrl) return;
    const state = getState(img);
    const translationKey = makeTranslationKey(imageUrl);

    if (state.translatedDataUrl && state.translationKey === translationKey) {
      toggleTranslatedView(img, button);
      return;
    }

    const originalDisplay = button.style.display;
    const originalText = button.textContent;
    button.disabled = true;
    button.classList.add("is-loading");

    try {
      const domImage = await captureImageDataFromDom(img);
      const visibleCapture = !domImage ? getVisibleCaptureRect(img) : null;
      if (visibleCapture) button.style.display = "none";

      const response = await requestTranslationViaPort(img, imageUrl, domImage, visibleCapture);
      if (!response) return; // ошибка уже показана

      const dataUrl = await resolveResultDataUrl(response);
      if (!dataUrl) return; // ошибка уже показана

      state.translatedDataUrl = dataUrl;
      state.translationKey = translationKey;
      applyTranslatedView(img);
      img.dataset.comicTranslatorTranslated = "1";
      button.textContent = t("button_revert", "Original");
    } catch (error) {
      console.error("[Comic Translator] click error:", error);
      showToast(`${t("error_generic", "Translation error")}: ${String(error)}`, "error");
    } finally {
      button.style.display = originalDisplay;
      button.disabled = false;
      button.classList.remove("is-loading");
      if (img.dataset.comicTranslatorTranslated === "1") {
        button.textContent = t("button_revert", "Original");
      } else {
        button.textContent = originalText;
      }
      updateButtonPosition(img);
    }
  });

  layer.appendChild(button);
  state.button = button;
  return button;
}

function toggleTranslatedView(img, button) {
  const wasTranslated = img.dataset.comicTranslatorTranslated === "1";
  if (wasTranslated) {
    ensureOriginalVisible(img);
    hideClone(img);
    hideOverlay(img);
    img.dataset.comicTranslatorTranslated = "0";
    button.textContent = t("button_label") || "IBF";
  } else {
    applyTranslatedView(img);
    img.dataset.comicTranslatorTranslated = "1";
    button.textContent = t("button_revert", "Original");
  }
}

function requestTranslationViaPort(img, imageUrl, domImage, visibleCapture) {
  return new Promise((resolve, reject) => {
    let port;
    try {
      port = chrome.runtime.connect({ name: "comic-translator" });
    } catch (err) {
      reject(err);
      return;
    }

    const timeout = setTimeout(() => {
      try { port.disconnect(); } catch {}
      reject(new Error("Translation timed out (no response in 5 minutes)"));
    }, 5 * 60 * 1000);

    port.onMessage.addListener((msg) => {
      if (msg.type === "keep-alive") return;
      if (msg.type === "translate-result") {
        clearTimeout(timeout);
        port.disconnect();
        resolve(msg.data);
      }
    });

    port.onDisconnect.addListener(() => {
      clearTimeout(timeout);
      const err = chrome.runtime.lastError;
      reject(new Error(err?.message || "Port disconnected before response"));
    });

    const message = {
      type: "translate-image",
      imageUrl,
      pageUrl: window.location.href,
      sourceOcrLang: SETTINGS.sourceOcrLang,
      targetLang: SETTINGS.targetLang,
      domImageInfo: domImage
        ? {
            mime: domImage.mime,
            width: domImage.width,
            height: domImage.height,
            bytes: domImage.bytes,
            elapsedMs: domImage.elapsedMs
          }
        : null,
      visibleCapture
    };

    const transfer = [];
    if (domImage?.arrayBuffer) {
      message.domImageBuffer = domImage.arrayBuffer;
      transfer.push(domImage.arrayBuffer);
    }
    port.postMessage(message, transfer);
  }).then(
    (response) => {
      if (!response?.ok || !response.result_url) {
        console.error("[Comic Translator] translate error:", response?.error || response);
        showToast(`${t("error_translate", "Translation error")}: ${response?.error || "unknown error"}`, "error");
        return null;
      }
      return response;
    },
    (error) => {
      console.error("[Comic Translator] port request failed:", error);
      showToast(`${t("error_generic", "Translation error")}: ${String(error)}`, "error");
      return null;
    }
  );
}

async function resolveResultDataUrl(response) {
  if (response.dataUrl) return response.dataUrl;
  const fetchResult = await chrome.runtime.sendMessage({
    type: "fetch-image-as-data-url",
    url: response.result_url
  });
  if (!fetchResult?.ok || !fetchResult?.dataUrl) {
    console.error("[Comic Translator] fetch result error:", fetchResult?.error || fetchResult);
    showToast(`${t("error_fetch", "Failed to load result")}: ${fetchResult?.error || "unknown error"}`, "error");
    return null;
  }
  return fetchResult.dataUrl;
}

function updateButtonPosition(img) {
  const state = getState(img);
  const button = state.button;
  if (!button) return;

  if (!SETTINGS.enabled) {
    button.style.display = "none";
    hideOverlay(img);
    return;
  }

  if (!img.isConnected || !looksLikeImage(img)) {
    button.style.display = "none";
    hideOverlay(img);
    return;
  }

  const rect = img.getBoundingClientRect();
  const hidden =
    rect.width < MIN_SIDE ||
    rect.height < MIN_SIDE ||
    rect.bottom < 0 ||
    rect.top > window.innerHeight ||
    rect.right < 0 ||
    rect.left > window.innerWidth;

  if (hidden) {
    button.style.display = "none";
    hideOverlay(img);
    return;
  }

  button.style.display = "flex";
  button.style.top = `${Math.max(8, rect.top + 8)}px`;
  button.style.left = `${Math.max(8, rect.right - 52)}px`;

  if (SETTINGS.renderMode === "overlay" && state.translatedDataUrl) {
    updateOverlayGeometry(img);
  }
}

function bindImage(img) {
  if (!SETTINGS.enabled) return;
  if (img.getAttribute(PROCESSED_ATTR) === "1") return;
  if (!looksLikeImage(img)) return;

  img.setAttribute(PROCESSED_ATTR, "1");
  createButton(img);
  updateButtonPosition(img);
  attachResizeObserver(img);

  // Auto-translate: запускаем перевод сразу, не дожидаясь клика.
  // Пропускаем, если img уже за пределами viewport — иначе каждое
  // прохождение через скролл будет триггерить OCR-цикл.
  if (SETTINGS.autoTranslate && isInViewport(img) && !getState(img).translatedDataUrl) {
    getState(img).button?.click();
  }

  console.log("[Comic Translator] button attached:", getImageUrl(img));
}

function isInViewport(img) {
  const rect = img.getBoundingClientRect();
  return rect.top < window.innerHeight &&
         rect.bottom > 0 &&
         rect.left < window.innerWidth &&
         rect.right > 0;
}

function attachResizeObserver(img) {
  const state = getState(img);
  if (state.resizeObserver || typeof ResizeObserver === "undefined") return;
  let timer = null;
  const observer = new ResizeObserver(() => {
    // debounce — ResizeObserver стреляет на каждый кадр анимации
    if (timer !== null) return;
    timer = setTimeout(() => {
      timer = null;
      if (!img.isConnected) return;
      const { translatedDataUrl } = state;
      updateButtonPosition(img);
      if (SETTINGS.renderMode === "overlay" && translatedDataUrl) {
        updateOverlayGeometry(img);
      }
    }, 50);
  });
  observer.observe(img);
  state.resizeObserver = observer;
}

function scanImages() {
  if (SETTINGS.enabled) {
    const images = document.querySelectorAll("img");
    images.forEach(bindImage);
  }

  for (const [img] of STATES.entries()) {
    if (!img.isConnected) {
      cleanupState(img);
      continue;
    }

    updateButtonPosition(img);
    applyTranslatedView(img);
  }
}

function debounce(fn, ms) {
  let timer = null;
  return function debounced(...args) {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn(...args);
    }, ms);
  };
}

const debouncedScanImages = debounce(scanImages, 100);

const observer = new MutationObserver(() => {
  window.requestAnimationFrame(scanImages);
});

observer.observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ["src", "srcset", "class"]
});

window.addEventListener("scroll", debouncedScanImages, { passive: true });
window.addEventListener("resize", debouncedScanImages, { passive: true });
window.addEventListener("load", scanImages);
document.addEventListener("readystatechange", scanImages);

// Обработчик контекстного меню: находим img с этим src и запускаем перевод.
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "translate-image-by-url" || !message.imageUrl) return;
  try {
    const images = document.querySelectorAll("img");
    for (const img of images) {
      if (getImageUrl(img) === message.imageUrl && looksLikeImage(img)) {
        bindImage(img);
        const state = getState(img);
        if (state.button) {
          state.button.click();
        }
        break;
      }
    }
    sendResponse({ ok: true });
  } catch (error) {
    sendResponse({ ok: false, error: String(error) });
  }
  return true;
});

(async () => {
  await loadSettings();
  applyGlobalState();
})();
