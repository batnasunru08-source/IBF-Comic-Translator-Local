const DEFAULTS = {
  enabled: true,
  renderMode: "replace",
  sourceOcrLang: "en",
  targetLang: "Russian"
};

function updateModeSection(enabled) {
  const section = document.getElementById("modeSection");
  if (!section) return;

  section.classList.toggle("muted", !enabled);

  document.querySelectorAll('input[name="renderMode"]').forEach((input) => {
    input.disabled = !enabled;
  });

  const sourceOcrLang = document.getElementById("sourceOcrLang");
  const targetLang = document.getElementById("targetLang");
  if (sourceOcrLang) sourceOcrLang.disabled = !enabled;
  if (targetLang) targetLang.disabled = !enabled;
}

async function init() {
  const stored = await chrome.storage.local.get(DEFAULTS);

  const enabledCheckbox = document.getElementById("enabled");
  const enabled = stored.enabled !== false;
  const renderMode = stored.renderMode === "overlay" ? "overlay" : "replace";
  const sourceOcrLang = stored.sourceOcrLang || "en";
  const targetLang = stored.targetLang || "Russian";

  enabledCheckbox.checked = enabled;
  updateModeSection(enabled);

  const radio = document.querySelector(`input[name="renderMode"][value="${renderMode}"]`);
  if (radio) radio.checked = true;

  const sourceSelect = document.getElementById("sourceOcrLang");
  const targetSelect = document.getElementById("targetLang");
  if (sourceSelect) sourceSelect.value = sourceOcrLang;
  if (targetSelect) targetSelect.value = targetLang;

  enabledCheckbox.addEventListener("change", async () => {
    await chrome.storage.local.set({ enabled: enabledCheckbox.checked });
    updateModeSection(enabledCheckbox.checked);
  });

  document.querySelectorAll('input[name="renderMode"]').forEach((input) => {
    input.addEventListener("change", async () => {
      if (!input.checked) return;
      await chrome.storage.local.set({ renderMode: input.value });
    });
  });

  sourceSelect?.addEventListener("change", async () => {
    await chrome.storage.local.set({ sourceOcrLang: sourceSelect.value });
  });

  targetSelect?.addEventListener("change", async () => {
    await chrome.storage.local.set({ targetLang: targetSelect.value });
  });
}

init().catch(console.error);