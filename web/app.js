const fileInput = document.getElementById("audio-file");
const fileHint = document.getElementById("file-hint");
const sourceRadios = document.querySelectorAll("input[name='source']");
const uploadArea = document.getElementById("upload-area");
const remoteArea = document.getElementById("remote-area");
const languageSelect = document.getElementById("language");
const apiKeyInput = document.getElementById("api-key");
const form = document.getElementById("detect-form");
const resultBody = document.getElementById("result-body");
const statusPill = document.getElementById("status-pill");
const logList = document.getElementById("log-list");
const clearLogBtn = document.getElementById("clear-log");
const remoteInput = document.getElementById("audio-url");

let selectedFile = null;

const toBase64 = (file) =>
    new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result;
            const base64 = result.split(",").pop();
            resolve(base64);
        };
        reader.onerror = () => reject(new Error("Failed to read file"));
        reader.readAsDataURL(file);
    });

const setStatus = (state, text) => {
    statusPill.textContent = text;
    statusPill.className = `pill ${state}`;
};

const renderResult = (payload) => {
    const { classification, confidence, explanation } = payload;
    const confidencePct = `${(confidence * 100).toFixed(2)}%`;
    resultBody.innerHTML = `
        <div class="label">${classification.replace("_", " → ")}</div>
        <div class="score">${confidencePct}</div>
        <ul class="explanation">
            ${explanation.map((note) => `<li>${note}</li>`).join("")}
        </ul>
    `;
};

const renderError = (message) => {
    resultBody.innerHTML = `<p class="placeholder">${message}</p>`;
};

const appendLog = (entry) => {
    const item = document.createElement("article");
    item.className = `log-entry ${entry.kind}`;
    item.innerHTML = `
        <strong>${entry.title}</strong>
        <span>${new Date().toLocaleTimeString()}</span>
        <code>${entry.detail}</code>
    `;
    logList.prepend(item);
};

fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) {
        selectedFile = null;
        fileHint.textContent = "No file selected yet.";
        return;
    }
    selectedFile = file;
    const sizeLabel = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    fileHint.textContent = `${file.name} · ${sizeLabel}`;
});

sourceRadios.forEach((radio) =>
    radio.addEventListener("change", () => {
        const mode = document.querySelector("input[name='source']:checked").value;
        if (mode === "upload") {
            uploadArea.classList.remove("hidden");
            remoteArea.classList.add("hidden");
        } else {
            uploadArea.classList.add("hidden");
            remoteArea.classList.remove("hidden");
        }
    })
);

clearLogBtn.addEventListener("click", () => {
    logList.innerHTML = "";
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) {
        renderError("API key required.");
        setStatus("pill-error", "Missing key");
        return;
    }

    const mode = document.querySelector("input[name='source']:checked").value;
    const payload = {};

    try {
        if (mode === "upload") {
            if (!selectedFile) {
                throw new Error("Please select an MP3 file.");
            }
            payload.audio_base64 = await toBase64(selectedFile);
        } else {
            const url = remoteInput.value.trim();
            if (!url) {
                throw new Error("Provide a remote audio URL.");
            }
            payload.audio_url = url;
        }
    } catch (error) {
        renderError(error.message);
        setStatus("pill-error", "Missing media");
        return;
    }

    const language = languageSelect.value;
    if (language) {
        payload.language = language;
    }

    setStatus("pill-loading", "Running...");
    renderError("Waiting for response...");

    try {
        const response = await fetch("/detect", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-API-KEY": apiKey,
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorText = await response.text();
            appendLog({
                kind: "error",
                title: `HTTP ${response.status}`,
                detail: errorText,
            });
            throw new Error("Request failed.");
        }

        const result = await response.json();
        renderResult(result);
        setStatus("pill-success", result.classification.toLowerCase());
        appendLog({
            kind: "success",
            title: `Result: ${result.classification}`,
            detail: JSON.stringify(result),
        });
    } catch (error) {
        renderError(error.message || "Unexpected failure.");
        setStatus("pill-error", "Error");
    }
});
