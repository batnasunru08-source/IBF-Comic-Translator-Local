const API_BASE = "http://127.0.0.1:8000";

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

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "translate-image") {
    (async () => {
      try {
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

        const data = await response.json();
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
        const response = await fetch(message.url, { method: "GET" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const blob = await response.blob();
        const dataUrl = await blobToDataUrl(blob);

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