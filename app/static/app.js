const state = {
  documents: [],
  messages: [],
  health: null,
  sending: false,
};

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  bindEvents();
  renderMessages();
  refreshAll();
  refreshIcons();
});

function bindElements() {
  els.statusText = document.querySelector("#statusText");
  els.modelText = document.querySelector("#modelText");
  els.documentList = document.querySelector("#documentList");
  els.uploadState = document.querySelector("#uploadState");
  els.fileInput = document.querySelector("#fileInput");
  els.dropzone = document.querySelector("#dropzone");
  els.refreshDocsBtn = document.querySelector("#refreshDocsBtn");
  els.ragToggle = document.querySelector("#ragToggle");
  els.topKInput = document.querySelector("#topKInput");
  els.messages = document.querySelector("#messages");
  els.chatForm = document.querySelector("#chatForm");
  els.messageInput = document.querySelector("#messageInput");
  els.sendBtn = document.querySelector("#sendBtn");
  els.clearChatBtn = document.querySelector("#clearChatBtn");
  els.errorBanner = document.querySelector("#errorBanner");
}

function bindEvents() {
  els.refreshDocsBtn.addEventListener("click", refreshAll);
  els.fileInput.addEventListener("change", () => {
    const [file] = els.fileInput.files;
    if (file) uploadFile(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("drag-over");
    });
  });

  els.dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (file) uploadFile(file);
  });

  els.documentList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-delete-doc]");
    if (!button) return;
    await deleteDocument(button.dataset.deleteDoc);
  });

  els.chatForm.addEventListener("submit", sendMessage);
  els.messageInput.addEventListener("input", resizeComposer);
  els.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.chatForm.requestSubmit();
    }
  });

  els.clearChatBtn.addEventListener("click", () => {
    state.messages = [];
    hideError();
    renderMessages();
  });
}

async function refreshAll() {
  await Promise.all([loadHealth(), loadDocuments()]);
  refreshIcons();
}

async function loadHealth() {
  try {
    state.health = await requestJson("/api/health");
    els.statusText.textContent = state.health.api_key_configured
      ? "API 已配置"
      : "等待配置 API Key";
    els.modelText.textContent = `${state.health.chat_model} / ${state.health.embedding_model}`;
  } catch (error) {
    els.statusText.textContent = "服务未连接";
    els.modelText.textContent = "后端不可用";
    showError(error.message);
  }
}

async function loadDocuments() {
  try {
    state.documents = await requestJson("/api/documents");
    renderDocuments();
  } catch (error) {
    showError(error.message);
  }
}

async function uploadFile(file) {
  hideError();
  els.uploadState.textContent = `正在上传 ${file.name}`;
  els.fileInput.value = "";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const result = await requestJson("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
    els.uploadState.textContent = `已索引 ${result.document.name}，${result.chunks} 个片段`;
    await refreshAll();
  } catch (error) {
    els.uploadState.textContent = "上传失败";
    showError(error.message);
  }
}

async function deleteDocument(documentId) {
  hideError();
  try {
    await requestJson(`/api/documents/${documentId}`, { method: "DELETE" });
    await refreshAll();
  } catch (error) {
    showError(error.message);
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const message = els.messageInput.value.trim();
  if (!message || state.sending) return;

  const history = state.messages
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item) => ({ role: item.role, content: item.content }))
    .slice(-10);

  state.messages.push({ role: "user", content: message });
  state.messages.push({ role: "assistant", content: "", loading: true });
  state.sending = true;
  els.messageInput.value = "";
  resizeComposer();
  hideError();
  renderMessages();

  try {
    const response = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history,
        use_rag: els.ragToggle.checked,
        top_k: Number(els.topKInput.value || 0),
      }),
    });

    state.messages.pop();
    state.messages.push({
      role: "assistant",
      content: response.answer,
      sources: response.sources,
    });
    await loadHealth();
  } catch (error) {
    state.messages.pop();
    showError(error.message);
  } finally {
    state.sending = false;
    renderMessages();
    refreshIcons();
    els.messageInput.focus();
  }
}

function renderDocuments() {
  if (!state.documents.length) {
    els.documentList.innerHTML = `<div class="muted-line">还没有知识库文档</div>`;
    return;
  }

  els.documentList.innerHTML = state.documents
    .map(
      (document) => `
        <article class="document-row">
          <div>
            <div class="document-name" title="${escapeHtml(document.name)}">${escapeHtml(document.name)}</div>
            <div class="document-meta">${document.chunk_count} 个片段</div>
          </div>
          <button class="icon-button" type="button" title="删除文档" data-delete-doc="${document.id}">
            <i data-lucide="trash-2"></i>
          </button>
        </article>
      `,
    )
    .join("");
  refreshIcons();
}

function renderMessages() {
  if (!state.messages.length) {
    els.messages.innerHTML = `
      <div class="empty-state">
        <div>
          <strong>开始一次 RAG 对话</strong>
          <span>上传资料后提问，回答会附带命中的来源片段。</span>
        </div>
      </div>
    `;
    return;
  }

  els.messages.innerHTML = state.messages
    .map((message) => {
      const content = message.loading
        ? `<span class="typing"><span></span><span></span><span></span></span>`
        : renderContent(message.content);
      const sources = message.sources?.length ? renderSources(message.sources) : "";
      return `
        <article class="message ${message.role}">
          <div class="bubble">${content}</div>
          ${sources}
        </article>
      `;
    })
    .join("");
  els.messages.scrollTop = els.messages.scrollHeight;
}

function renderSources(sources) {
  return `
    <div class="source-list">
      ${sources
        .map(
          (source, index) => `
            <section class="source-item">
              <div class="source-title">
                <span>[${index + 1}] ${escapeHtml(source.document_name)}</span>
                <span>${Math.round(source.score * 100)}%</span>
              </div>
              <div class="source-text">${escapeHtml(source.text)}</div>
            </section>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderContent(content) {
  return escapeHtml(content)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br />");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }

  if (!response.ok) {
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return payload;
}

function resizeComposer() {
  els.messageInput.style.height = "auto";
  els.messageInput.style.height = `${Math.min(els.messageInput.scrollHeight, 170)}px`;
}

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.hidden = false;
}

function hideError() {
  els.errorBanner.hidden = true;
  els.errorBanner.textContent = "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

