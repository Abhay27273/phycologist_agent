/* ============================================================
   Mindful — frontend logic
   Same-origin static page served by FastAPI (see app/api/server.py).
   No build step, no dependencies.
   ============================================================ */

(() => {
  "use strict";

  // ---------------------------------------------------------
  // State
  // ---------------------------------------------------------
  const state = {
    token: localStorage.getItem("mindful_token") || null,
    userId: localStorage.getItem("mindful_user_id") || null,
    sessionId: null,
    ws: null,
    authMode: "login",
    ttsEnabled: localStorage.getItem("mindful_tts") === "1",
    recognizing: false,
    currentAssistantBubble: null,
    typingEl: null,
    reconnectAttempted: false,
  };

  // ---------------------------------------------------------
  // DOM refs
  // ---------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const authScreen = $("auth-screen");
  const chatScreen = $("chat-screen");
  const authForm = $("auth-form");
  const authEmail = $("auth-email");
  const authPassword = $("auth-password");
  const authError = $("auth-error");
  const authSubmit = $("auth-submit");
  const tabs = document.querySelectorAll(".tab");
  const messageList = $("message-list");
  const emptyState = $("empty-state");
  const composer = $("composer");
  const composerInput = $("composer-input");
  const sendBtn = $("send-btn");
  const micBtn = $("mic-btn");
  const listeningHint = $("listening-hint");
  const riskBadge = $("risk-badge");
  const ttsToggle = $("tts-toggle");
  const logoutBtn = $("logout-btn");
  const connectionBanner = $("connection-banner");

  // ---------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------
  function apiUrl(path) {
    return path; // same-origin
  }

  function wsUrl(sessionId, token) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/api/v1/ws/chat/${sessionId}?token=${encodeURIComponent(token)}`;
  }

  function sessionKeyFor(userId) {
    return `mindful_session_${userId}`;
  }

  function getOrCreateSessionId(userId) {
    const key = sessionKeyFor(userId);
    let sid = localStorage.getItem(key);
    if (!sid) {
      sid = (crypto.randomUUID && crypto.randomUUID()) ||
        `sess-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      localStorage.setItem(key, sid);
    }
    return sid;
  }

  function scrollToBottom() {
    messageList.scrollTop = messageList.scrollHeight;
  }

  function appendMessage(role, text, opts = {}) {
    if (emptyState) emptyState.hidden = true;
    const el = document.createElement("div");
    el.className = `msg ${role}${opts.crisis ? " crisis" : ""}`;
    el.textContent = text;
    messageList.appendChild(el);
    scrollToBottom();
    return el;
  }

  function showTyping() {
    if (state.typingEl) return;
    const el = document.createElement("div");
    el.className = "typing-dots";
    el.innerHTML = "<span></span><span></span><span></span>";
    messageList.appendChild(el);
    state.typingEl = el;
    scrollToBottom();
  }

  function hideTyping() {
    if (state.typingEl) {
      state.typingEl.remove();
      state.typingEl = null;
    }
  }

  function setRiskBadge(level) {
    const labels = { LOW: "● steady", MEDIUM: "● elevated", HIGH: "● please reach out" };
    riskBadge.dataset.level = level;
    riskBadge.textContent = labels[level] || labels.LOW;
  }

  function setBusy(busy) {
    sendBtn.disabled = busy;
    composerInput.disabled = busy;
  }

  function showConnectionBanner(text) {
    if (!text) {
      connectionBanner.hidden = true;
      return;
    }
    connectionBanner.textContent = text;
    connectionBanner.hidden = false;
  }

  // ---------------------------------------------------------
  // Auth screen
  // ---------------------------------------------------------
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      state.authMode = tab.dataset.mode;
      authSubmit.textContent = state.authMode === "login" ? "Log in" : "Create account";
      authPassword.setAttribute(
        "autocomplete",
        state.authMode === "login" ? "current-password" : "new-password"
      );
      authError.hidden = true;
    });
  });

  authForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    authError.hidden = true;
    authSubmit.disabled = true;
    authSubmit.textContent = "Please wait…";

    const email = authEmail.value.trim();
    const password = authPassword.value;
    const path = state.authMode === "login" ? "/api/v1/auth/login" : "/api/v1/auth/register";

    try {
      const res = await fetch(apiUrl(path), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Something went wrong. Please try again.");
      }

      state.token = data.access_token;
      state.userId = data.user_id;
      localStorage.setItem("mindful_token", state.token);
      localStorage.setItem("mindful_user_id", state.userId);

      enterChat();
    } catch (err) {
      authError.textContent = err.message;
      authError.hidden = false;
    } finally {
      authSubmit.disabled = false;
      authSubmit.textContent = state.authMode === "login" ? "Log in" : "Create account";
    }
  });

  // ---------------------------------------------------------
  // Chat screen bootstrap
  // ---------------------------------------------------------
  function enterChat() {
    authScreen.hidden = true;
    chatScreen.hidden = false;
    state.sessionId = getOrCreateSessionId(state.userId);
    connectWebSocket();
    composerInput.focus();
  }

  function connectWebSocket() {
    showConnectionBanner("Connecting…");
    const ws = new WebSocket(wsUrl(state.sessionId, state.token));
    state.ws = ws;

    ws.onopen = () => {
      state.reconnectAttempted = false;
      showConnectionBanner("");
    };

    ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      handleServerEvent(data);
    };

    ws.onerror = () => {
      showConnectionBanner("Connection trouble…");
    };

    ws.onclose = (event) => {
      hideTyping();
      setBusy(false);
      if (event.code === 4401) {
        // Token invalid/expired — force re-auth
        logout();
        return;
      }
      if (!state.reconnectAttempted) {
        state.reconnectAttempted = true;
        showConnectionBanner("Reconnecting…");
        setTimeout(connectWebSocket, 1500);
      } else {
        showConnectionBanner("Disconnected. Reload the page to reconnect.");
      }
    };
  }

  function handleServerEvent(data) {
    switch (data.type) {
      case "meta":
        // Preemptive risk hint; final risk_level arrives on "done".
        if (typeof data.risk_score === "number" && data.risk_score >= 8) {
          setRiskBadge("HIGH");
        }
        hideTyping();
        break;

      case "sentence":
        if (!state.currentAssistantBubble) {
          state.currentAssistantBubble = appendMessage("assistant", data.content);
        } else {
          state.currentAssistantBubble.textContent += " " + data.content;
        }
        scrollToBottom();
        speak(data.content);
        break;

      case "done":
        setRiskBadge(data.risk_level || "LOW");
        if (data.risk_level === "HIGH" && state.currentAssistantBubble) {
          state.currentAssistantBubble.classList.add("crisis");
        }
        state.currentAssistantBubble = null;
        setBusy(false);
        composerInput.focus();
        break;

      case "error":
        hideTyping();
        appendMessage("system", data.message || "Something went wrong.");
        state.currentAssistantBubble = null;
        setBusy(false);
        break;
    }
  }

  // ---------------------------------------------------------
  // Sending messages
  // ---------------------------------------------------------
  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    sendCurrentMessage();
  });

  composerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendCurrentMessage();
    }
  });

  composerInput.addEventListener("input", () => {
    composerInput.style.height = "auto";
    composerInput.style.height = Math.min(composerInput.scrollHeight, 120) + "px";
  });

  function sendCurrentMessage() {
    const text = composerInput.value.trim();
    if (!text || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    appendMessage("user", text);
    composerInput.value = "";
    composerInput.style.height = "auto";
    setBusy(true);
    showTyping();

    state.ws.send(JSON.stringify({ user_id: state.userId, message: text }));
  }

  // ---------------------------------------------------------
  // Voice input (Web Speech API — browser-native, no server round-trip)
  // ---------------------------------------------------------
  const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognizer = null;

  if (!SpeechRecognitionImpl) {
    micBtn.disabled = true;
    micBtn.title = "Voice input isn't supported in this browser";
  } else {
    recognizer = new SpeechRecognitionImpl();
    recognizer.continuous = false;
    recognizer.interimResults = false;
    recognizer.lang = "en-US";

    recognizer.onstart = () => {
      state.recognizing = true;
      micBtn.classList.add("listening");
      listeningHint.hidden = false;
    };

    recognizer.onend = () => {
      state.recognizing = false;
      micBtn.classList.remove("listening");
      listeningHint.hidden = true;
    };

    recognizer.onerror = (e) => {
      state.recognizing = false;
      micBtn.classList.remove("listening");
      listeningHint.hidden = true;
      if (e.error !== "aborted" && e.error !== "no-speech") {
        appendMessage("system", "Voice input couldn't access your microphone.");
      }
    };

    recognizer.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      composerInput.value = transcript;
      composerInput.dispatchEvent(new Event("input"));
      composerInput.focus();
    };
  }

  micBtn.addEventListener("click", () => {
    if (!recognizer) return;
    if (state.recognizing) {
      recognizer.stop();
    } else {
      try {
        recognizer.start();
      } catch {
        /* already started — ignore */
      }
    }
  });

  // ---------------------------------------------------------
  // Text-to-speech playback of assistant sentences
  // ---------------------------------------------------------
  function updateTtsButton() {
    ttsToggle.setAttribute("aria-pressed", String(state.ttsEnabled));
    ttsToggle.textContent = state.ttsEnabled ? "🔊" : "🔈";
  }

  ttsToggle.addEventListener("click", () => {
    state.ttsEnabled = !state.ttsEnabled;
    localStorage.setItem("mindful_tts", state.ttsEnabled ? "1" : "0");
    if (!state.ttsEnabled && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    updateTtsButton();
  });

  function speak(text) {
    if (!state.ttsEnabled || !("speechSynthesis" in window)) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
  }

  // ---------------------------------------------------------
  // Logout
  // ---------------------------------------------------------
  function logout() {
    if (state.ws) {
      state.ws.onclose = null;
      state.ws.close();
      state.ws = null;
    }
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    localStorage.removeItem("mindful_token");
    localStorage.removeItem("mindful_user_id");
    state.token = null;
    state.userId = null;
    messageList.querySelectorAll(".msg, .typing-dots").forEach((el) => el.remove());
    if (emptyState) emptyState.hidden = false;
    setRiskBadge("LOW");
    chatScreen.hidden = true;
    authScreen.hidden = false;
    authEmail.value = "";
    authPassword.value = "";
  }

  logoutBtn.addEventListener("click", logout);

  // ---------------------------------------------------------
  // Boot
  // ---------------------------------------------------------
  updateTtsButton();
  if (state.token && state.userId) {
    enterChat();
  }
})();
