const STATES = new Map();
const PROCESSED_ATTR = "data-comic-translator-bound";
const MIN_SIDE = 80;

const SETTINGS = {
  enabled: true,
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

async function loadSettings() {
  try {
    const stored = await chrome.storage.local.get({
      enabled: true,
      renderMode: "replace",
      sourceOcrLang: "en",
      targetLang: "Russian"
    });

    SETTINGS.enabled = stored.enabled !== false;
    SETTINGS.renderMode = stored.renderMode === "overlay" ? "overlay" : "replace";
    SETTINGS.sourceOcrLang = stored.sourceOcrLang || "en";
    SETTINGS.targetLang = stored.targetLang || "Russian";

  } catch (error) {
    console.error("[Comic Translator] loadSettings error:", error);
  }
}

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "local") return;

  if (changes.enabled) {
    SETTINGS.enabled = changes.enabled.newValue !== false;
  }

  if (changes.renderMode) {
    SETTINGS.renderMode = changes.renderMode.newValue === "overlay" ? "overlay" : "replace";
  }

  if (changes.sourceOcrLang) {
    SETTINGS.sourceOcrLang = changes.sourceOcrLang.newValue || "en";
  }

  if (changes.targetLang) {
    SETTINGS.targetLang = changes.targetLang.newValue || "Russian";
  }

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
      originalDisplay: null
    };
    STATES.set(img, state);
  }
  return state;
}

function cleanupState(img) {
  const state = STATES.get(img);
  if (!state) return;

  if (state.button?.isConnected) state.button.remove();
  if (state.overlay?.isConnected) state.overlay.remove();
  if (state.clone?.isConnected) state.clone.remove();

  STATES.delete(img);
}

function getOrCreateClone(img) {
  const state = getState(img);

  if (!state.clone || !state.clone.isConnected) {
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

  if (!state.overlay || !state.overlay.isConnected) {
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
  const state = getState(img);
  img.style.display = state.originalDisplay ?? "";
}

function hideOriginalForReplace(img) {
  const state = getState(img);
  if (state.originalDisplay === null) {
    state.originalDisplay = img.style.display || "";
  }
  img.style.display = "none";
}

function hideClone(img) {
  const state = getState(img);
  if (state.clone) {
    state.clone.style.display = "none";
  }
}

function hideOverlay(img) {
  const state = getState(img);
  if (state.overlay) {
    state.overlay.style.display = "none";
  }
}

function hideButton(img) {
  const state = getState(img);
  if (state.button) {
    state.button.style.display = "none";
  }
}

function syncCloneGeometry(img, clone) {
  clone.style.width = `${img.clientWidth || img.naturalWidth}px`;
  clone.style.height = "auto";
}

function updateOverlayGeometry(img) {
  const state = getState(img);
  if (!state.overlay || !state.translatedDataUrl) return;

  const rect = img.getBoundingClientRect();
  const hidden =
    rect.width < MIN_SIDE ||
    rect.height < MIN_SIDE ||
    rect.bottom < 0 ||
    rect.top > window.innerHeight ||
    rect.right < 0 ||
    rect.left > window.innerWidth;

  if (hidden) {
    state.overlay.style.display = "none";
    return;
  }

  state.overlay.style.display = "block";
  state.overlay.style.left = `${rect.left}px`;
  state.overlay.style.top = `${rect.top}px`;
  state.overlay.style.width = `${rect.width}px`;
  state.overlay.style.height = `${rect.height}px`;
}

function applyTranslatedView(img) {
  const state = getState(img);

  if (!SETTINGS.enabled) {
    ensureOriginalVisible(img);
    hideClone(img);
    hideOverlay(img);
    hideButton(img);
    return;
  }

  if (!state.translatedDataUrl) {
    ensureOriginalVisible(img);
    hideClone(img);
    hideOverlay(img);
    return;
  }

  if (SETTINGS.renderMode === "overlay") {
    ensureOriginalVisible(img);
    hideClone(img);

    const overlay = getOrCreateOverlay(img);
    overlay.src = state.translatedDataUrl;
    updateOverlayGeometry(img);
    return;
  }

  hideOverlay(img);

  const clone = getOrCreateClone(img);
  clone.src = state.translatedDataUrl;
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
  button.textContent = "IBF";
  button.title = "Перевести изображение";

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
    backdropFilter: "blur(2px)"
  });

  button.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (!SETTINGS.enabled) return;

    const imageUrl = getImageUrl(img);
    if (!imageUrl) return;

    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = "...";

    try {
      const response = await chrome.runtime.sendMessage({
        type: "translate-image",
        imageUrl,
        pageUrl: window.location.href,
        sourceOcrLang: SETTINGS.sourceOcrLang,
        targetLang: SETTINGS.targetLang
      });

      if (!response?.ok || !response.result_url) {
        console.error("[Comic Translator] translate error:", response?.error || response);
        alert(`Ошибка перевода изображения: ${response?.error || "неизвестная ошибка"}`);
        return;
      }

      const fetchResult = await chrome.runtime.sendMessage({
        type: "fetch-image-as-data-url",
        url: response.result_url
      });

      if (!fetchResult?.ok || !fetchResult?.dataUrl) {
        console.error("[Comic Translator] fetch result error:", fetchResult?.error || fetchResult);
        alert(`Ошибка загрузки переведённого изображения: ${fetchResult?.error || "неизвестная ошибка"}`);
        return;
      }

      const state = getState(img);
      state.translatedDataUrl = fetchResult.dataUrl;

      if (!img.dataset.originalSrc) {
        img.dataset.originalSrc = img.src;
      }

      applyTranslatedView(img);
      img.dataset.comicTranslatorTranslated = "1";
    } catch (error) {
      console.error("[Comic Translator] click error:", error);
      alert(`Ошибка перевода изображения: ${String(error)}`);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
      updateButtonPosition(img);
    }
  });

  layer.appendChild(button);
  state.button = button;
  return button;
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

  console.log("[Comic Translator] button attached:", getImageUrl(img));
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

const observer = new MutationObserver(() => {
  window.requestAnimationFrame(scanImages);
});

observer.observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ["src", "srcset", "style", "class"]
});

window.addEventListener("scroll", scanImages, { passive: true });
window.addEventListener("resize", scanImages, { passive: true });
window.addEventListener("load", scanImages);
document.addEventListener("readystatechange", scanImages);

(async () => {
  await loadSettings();
  applyGlobalState();
})();