const DEFAULTS = {
  enabled: true,
  autoTranslate: false,
  renderMode: "replace",
  sourceOcrLang: "en",
  targetLang: "Russian",
  apiBase: "http://127.0.0.1:8000",
  hotkeyEnabled: false,
  hotkeyCode: "KeyT",
  hotkeyMode: "cursor"
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

function keyCharToCode(char) {
  const upper = String(char ?? "").toUpperCase();
  if (/^[A-Z]$/.test(upper)) return `Key${upper}`;
  if (/^[0-9]$/.test(upper)) return `Digit${upper}`;
  return null;
}

function codeToKeyChar(code) {
  const key = /^Key([A-Z])$/.exec(String(code ?? ""));
  if (key) return key[1];
  const digit = /^Digit([0-9])$/.exec(String(code ?? ""));
  if (digit) return digit[1];
  return null;
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

  const hotkeySection = document.getElementById("hotkeySection");
  if (hotkeySection) hotkeySection.classList.toggle("muted", !enabled);

  document.querySelectorAll('input[name="renderMode"]').forEach((input) => {
    input.disabled = !enabled;
  });

  document.querySelectorAll('input[name="hotkeyMode"]').forEach((input) => {
    input.disabled = !enabled;
  });

  for (const id of ["sourceOcrLang", "targetLang", "autoTranslate", "hotkeyEnabled", "hotkeyKey"]) {
    const el = document.getElementById(id);
    if (el) el.disabled = !enabled;
  }
}

async function init() {
  applyI18n();

  const { enabled, autoTranslate, renderMode, sourceOcrLang, targetLang, apiBase, hotkeyEnabled, hotkeyCode, hotkeyMode } = await chrome.storage.local.get(DEFAULTS);
  const isEnabled = enabled !== false;
  const isAuto = autoTranslate === true;
  const mode = renderMode === "overlay" ? "overlay" : "replace";
  const ocrLang = sourceOcrLang ?? "en";
  const tgtLang = targetLang ?? "Russian";
  const apiUrl = apiBase ?? "http://127.0.0.1:8000";
  const isHotkey = hotkeyEnabled === true;
  const keyChar = codeToKeyChar(hotkeyCode) ?? "T";
  const hkMode = hotkeyMode === "visible" ? "visible" : "cursor";

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

  const hotkeyCheckbox = document.getElementById("hotkeyEnabled");
  const hotkeyInput = document.getElementById("hotkeyKey");
  if (hotkeyCheckbox) hotkeyCheckbox.checked = isHotkey;
  if (hotkeyInput) hotkeyInput.value = keyChar;
  document.querySelectorAll('input[name="hotkeyMode"]').forEach((input) => {
    input.checked = input.value === hkMode;
  });

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

  hotkeyCheckbox?.addEventListener("change", () =>
    chrome.storage.local.set({ hotkeyEnabled: hotkeyCheckbox.checked })
  );

  hotkeyInput?.addEventListener("input", () => {
    const valid = keyCharToCode(hotkeyInput.value) !== null;
    hotkeyInput.classList.toggle("is-invalid", !valid);
  });

  const commitHotkey = async () => {
    if (!hotkeyInput) return;
    const code = keyCharToCode(hotkeyInput.value);
    if (code) {
      hotkeyInput.classList.remove("is-invalid");
      hotkeyInput.value = codeToKeyChar(code);
      await chrome.storage.local.set({ hotkeyCode: code });
      return;
    }
    const { hotkeyCode: current } = await chrome.storage.local.get({ hotkeyCode: DEFAULTS.hotkeyCode });
    hotkeyInput.value = codeToKeyChar(current) ?? "T";
    hotkeyInput.classList.remove("is-invalid");
  };
  hotkeyInput?.addEventListener("change", commitHotkey);
  hotkeyInput?.addEventListener("blur", commitHotkey);

  document.querySelectorAll('input[name="hotkeyMode"]').forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) chrome.storage.local.set({ hotkeyMode: input.value });
    });
  });

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
