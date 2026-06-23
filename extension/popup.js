const DEFAULTS = {
  enabled: true,
  autoTranslate: false,
  renderMode: "replace",
  sourceOcrLang: "en",
  targetLang: "Russian",
  apiBase: "http://127.0.0.1:8000"
};

function t(key, fallback) {
  try {
    return chrome.i18n.getMessage(key) ?? fallback ?? key;
  } catch {
    return fallback ?? key;
  }
}

function isValidApiBase(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) return false;
  try {
    const url = new URL(trimmed);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

const I18N_TAGS = new Set(["LABEL", "SPAN", "DIV"]);

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const msg = t(el.getAttribute("data-i18n"));
    if (el.tagName === "TITLE") {
      document.title = msg;
      return;
    }
    if (I18N_TAGS.has(el.tagName)) {
      el.textContent = msg;
    }
  });
}

function updateModeSection(enabled) {
  const modeSection = document.getElementById("modeSection");
  if (modeSection) modeSection.classList.toggle("muted", !enabled);

  const autoSection = document.getElementById("autoSection");
  if (autoSection) autoSection.classList.toggle("muted", !enabled);

  document.querySelectorAll('input[name="renderMode"]').forEach((input) => {
    input.disabled = !enabled;
  });

  for (const id of ["sourceOcrLang", "targetLang", "autoTranslate"]) {
    const el = document.getElementById(id);
    if (el) el.disabled = !enabled;
  }
}

async function init() {
  applyI18n();

  const { enabled, autoTranslate, renderMode, sourceOcrLang, targetLang, apiBase } = await chrome.storage.local.get(DEFAULTS);
  const isEnabled = enabled !== false;
  const isAuto = autoTranslate === true;
  const mode = renderMode === "overlay" ? "overlay" : "replace";
  const ocrLang = sourceOcrLang ?? "en";
  const tgtLang = targetLang ?? "Russian";
  const apiUrl = apiBase ?? "http://127.0.0.1:8000";

  const enabledCheckbox = document.getElementById("enabled");
  const autoCheckbox = document.getElementById("autoTranslate");
  enabledCheckbox.checked = isEnabled;
  if (autoCheckbox) autoCheckbox.checked = isAuto;
  updateModeSection(isEnabled);

  document.querySelector(`input[name="renderMode"][value="${mode}"]`)?.setAttribute("checked", "");

  const sourceSelect = document.getElementById("sourceOcrLang");
  const targetSelect = document.getElementById("targetLang");
  if (sourceSelect) sourceSelect.value = ocrLang;
  if (targetSelect) targetSelect.value = tgtLang;

  const apiInput = document.getElementById("apiBase");
  if (apiInput) apiInput.value = apiUrl;

  enabledCheckbox.addEventListener("change", async () => {
    await chrome.storage.local.set({ enabled: enabledCheckbox.checked });
    updateModeSection(enabledCheckbox.checked);
  });

  autoCheckbox?.addEventListener("change", () =>
    chrome.storage.local.set({ autoTranslate: autoCheckbox.checked })
  );

  document.querySelectorAll('input[name="renderMode"]').forEach((input) => {
    input.addEventListener("change", async () => {
      if (input.checked) await chrome.storage.local.set({ renderMode: input.value });
    });
  });

  sourceSelect?.addEventListener("change", () =>
    chrome.storage.local.set({ sourceOcrLang: sourceSelect.value })
  );
  targetSelect?.addEventListener("change", () =>
    chrome.storage.local.set({ targetLang: targetSelect.value })
  );

  // API base URL: сохраняем на change (по Enter или потере фокуса).
  // На input — подсвечиваем красным, если URL невалидный.
  apiInput?.addEventListener("input", () => {
    apiInput.classList.toggle("is-invalid", !isValidApiBase(apiInput.value));
  });
  const commitApiBase = async () => {
    if (!apiInput) return;
    if (isValidApiBase(apiInput.value)) {
      apiInput.classList.remove("is-invalid");
      await chrome.storage.local.set({ apiBase: apiInput.value.trim() });
      checkServerStatus();
    } else {
      apiInput.classList.add("is-invalid");
    }
  };
  apiInput?.addEventListener("change", commitApiBase);
  apiInput?.addEventListener("blur", commitApiBase);

  await checkServerStatus();
  document.getElementById("serverStatus")?.addEventListener("click", checkServerStatus);
}

async function checkServerStatus() {
  const status = document.getElementById("serverStatus");
  const text = document.getElementById("serverStatusText");
  if (!status || !text) return;

  const setState = (cls, msg) => {
    status.classList.remove("is-ok", "is-error");
    if (cls) status.classList.add(cls);
    text.textContent = msg;
  };

  setState(null, t("server_checking", "Checking…"));

  try {
    const response = await chrome.runtime.sendMessage({ type: "ping-server" });
    if (response?.ok) {
      setState("is-ok", t("server_online", "Server online"));
    } else {
      setState("is-error", t("server_offline", "Server offline"));
    }
  } catch (error) {
    setState("is-error", `${t("server_offline", "Server offline")} (${String(error)})`);
  }
}

init().catch(console.error);
