/* Intelligent OSINT Dashboard Controller (App Shell, Run State Machine, Tooltips, Toasts, Shortcuts) */

(function () {
  "use strict";

  const state = {
    activeView: "launch",
    theme: localStorage.getItem("osint_theme") || "dark",
    runState: "idle", // idle | running | completed | stopped | partial | error
    jobId: null,
    nodes: [],
    edges: [],
    auditEvents: [],
    selectedNodeId: null,
    searchQuery: "",
  };

  window.OSINT_STATE = state;

  const RUN_STATE_LABELS = {
    idle: "Idle",
    running: "Scanning",
    completed: "Completed",
    stopped: "Stopped",
    partial: "Partial",
    error: "Error",
  };

  function setRunState(nextState, customLabel) {
    const valid = ["idle", "running", "completed", "stopped", "partial", "error"].includes(nextState)
      ? nextState
      : "idle";
    state.runState = valid;
    const dot = document.getElementById("system-status-dot");
    const txt = document.getElementById("system-status-text");
    const pill = document.getElementById("status-pill");
    if (dot) dot.className = "status-dot " + valid;
    if (txt) txt.textContent = customLabel || RUN_STATE_LABELS[valid] || "Idle";
    if (pill) {
      pill.className = "badge " + (
        valid === "running" ? "badge-running" :
        valid === "completed" ? "badge-done" :
        valid === "partial" ? "badge-unavailable" :
        (valid === "stopped" || valid === "error") ? "badge-error" : "badge-waiting"
      );
    }
    const stopBtn = document.getElementById("btn-stop-scan");
    if (stopBtn) stopBtn.disabled = valid !== "running";
  }

  function decodeEntities(str) {
    if (str == null) return "";
    const s = String(str);
    if (!s.includes("&")) return s;
    const el = document.createElement("textarea");
    el.innerHTML = s;
    return el.value;
  }

  function esc(str) {
    return decodeEntities(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function applyTheme(theme) {
    state.theme = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", state.theme);
    localStorage.setItem("osint_theme", state.theme);
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.textContent = state.theme === "dark" ? "Light" : "Dark";
    }
  }

  function switchView(viewName) {
    state.activeView = viewName;
    document.querySelectorAll(".nav-item").forEach((btn) => {
      const isAct = btn.dataset.view === viewName;
      btn.classList.toggle("active", isAct);
      btn.setAttribute("aria-current", isAct ? "page" : "false");
    });
    document.querySelectorAll(".view-section").forEach((sec) => {
      sec.classList.toggle("active", sec.id === "view-" + viewName);
    });
    if (typeof window.onViewActivated === "function") {
      window.onViewActivated(viewName);
    }
  }

  let lastFocusedEl = null;

  function openInspector(nodeId) {
    lastFocusedEl = document.activeElement;
    state.selectedNodeId = nodeId || null;
    const drawer = document.getElementById("inspector-drawer");
    const backdrop = document.getElementById("inspector-backdrop");
    if (backdrop) backdrop.classList.add("open");
    if (drawer) {
      drawer.classList.add("open");
      drawer.setAttribute("aria-hidden", "false");
    }
    if (typeof window.renderInspector === "function") {
      window.renderInspector(nodeId);
    }
    requestAnimationFrame(() => {
      document.getElementById("inspector-close")?.focus({ preventScroll: true });
    });
  }

  function closeInspector() {
    const drawer = document.getElementById("inspector-drawer");
    const backdrop = document.getElementById("inspector-backdrop");
    if (backdrop) backdrop.classList.remove("open");
    if (drawer) {
      drawer.classList.remove("open");
      drawer.setAttribute("aria-hidden", "true");
    }
    if (lastFocusedEl && typeof lastFocusedEl.focus === "function") {
      lastFocusedEl.focus({ preventScroll: true });
      lastFocusedEl = null;
    }
  }

  function toast(message, variant = "info", opts = {}) {
    const stack = document.getElementById("toast-stack");
    if (!stack) return;
    while (stack.children.length >= 4) {
      stack.firstElementChild?.remove();
    }
    const el = document.createElement("div");
    el.className = `toast ${variant === "success" ? "toast-success" : variant === "error" ? "toast-error" : ""}`;
    el.setAttribute("role", "status");
    const span = document.createElement("span");
    span.textContent = String(message || "");
    span.style.flex = "1";
    el.appendChild(span);

    const dismiss = () => {
      el.classList.remove("open");
      setTimeout(() => el.remove(), 220);
    };

    if (opts && opts.actionLabel && typeof opts.onAction === "function") {
      const actBtn = document.createElement("button");
      actBtn.type = "button";
      actBtn.className = "btn btn-xs btn-primary";
      actBtn.textContent = opts.actionLabel;
      actBtn.addEventListener("click", () => {
        opts.onAction();
        dismiss();
      });
      el.appendChild(actBtn);
    }

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "btn btn-xs";
    closeBtn.textContent = "x";
    closeBtn.setAttribute("aria-label", "Dismiss notification");
    el.appendChild(closeBtn);
    stack.appendChild(el);
    requestAnimationFrame(() => el.classList.add("open"));

    const duration = Number(opts?.duration) || (opts?.actionLabel ? 6000 : 3500);
    let timer = setTimeout(dismiss, duration);
    el.addEventListener("mouseenter", () => clearTimeout(timer));
    el.addEventListener("mouseleave", () => { timer = setTimeout(dismiss, 2000); });
    closeBtn.addEventListener("click", dismiss);

    let startX = 0;
    let startTime = 0;
    el.addEventListener("pointerdown", (e) => {
      startX = e.clientX;
      startTime = Date.now();
    });
    el.addEventListener("pointerup", (e) => {
      const dx = e.clientX - startX;
      const dt = Math.max(1, Date.now() - startTime);
      if (Math.abs(dx) / dt > 0.11 || Math.abs(dx) > 90) dismiss();
    });
  }

  /* Shared Origin-Aware Tooltip Provider (skips delay & animation on subsequent hovers) */
  function initTooltipProvider() {
    let tipEl = document.getElementById("osint-tooltip-popover");
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.id = "osint-tooltip-popover";
      tipEl.className = "tooltip-popover";
      tipEl.setAttribute("role", "tooltip");
      document.body.appendChild(tipEl);
    }
    let openTimer = null;
    let warmUntil = 0;
    let activeTarget = null;

    const positionTip = (target) => {
      const r = target.getBoundingClientRect();
      const pad = 8;
      tipEl.style.left = "0px";
      tipEl.style.top = "0px";
      const tw = tipEl.offsetWidth || 180;
      const th = tipEl.offsetHeight || 32;
      let top = r.bottom + pad;
      let originY = "top";
      if (top + th > window.innerHeight - 10) {
        top = Math.max(8, r.top - th - pad);
        originY = "bottom";
      }
      let left = r.left + r.width / 2 - tw / 2;
      left = Math.max(8, Math.min(window.innerWidth - tw - 8, left));
      tipEl.style.setProperty("--transform-origin", `${originY} center`);
      tipEl.style.left = `${Math.round(left)}px`;
      tipEl.style.top = `${Math.round(top)}px`;
    };

    const showTip = (target) => {
      const text = target.getAttribute("data-tooltip");
      if (!text) return;
      const isInstant = Date.now() < warmUntil;
      tipEl.setAttribute("data-instant", isInstant ? "true" : "false");
      tipEl.textContent = text;
      positionTip(target);
      tipEl.classList.add("open");
      warmUntil = Date.now() + 650;
    };

    const hideTip = () => {
      clearTimeout(openTimer);
      activeTarget = null;
      tipEl.classList.remove("open");
      if (tipEl.classList.contains("open") || warmUntil > Date.now()) {
        warmUntil = Date.now() + 450;
      }
    };

    document.addEventListener("pointerover", (e) => {
      const t = e.target?.closest?.("[data-tooltip]");
      if (!t || t === activeTarget) return;
      activeTarget = t;
      clearTimeout(openTimer);
      if (Date.now() < warmUntil) {
        showTip(t);
      } else {
        openTimer = setTimeout(() => showTip(t), 260);
      }
    });
    document.addEventListener("pointerout", (e) => {
      if (!activeTarget) return;
      if (!e.relatedTarget || !activeTarget.contains(e.relatedTarget)) {
        hideTip();
      }
    });
    window.addEventListener("scroll", hideTip, true);
  }

  function openShortcutsModal() {
    if (document.getElementById("osint-shortcuts-modal")) return;
    const prev = document.activeElement;
    const modal = document.createElement("div");
    modal.id = "osint-shortcuts-modal";
    modal.className = "modal-backdrop";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-label", "Keyboard Shortcuts");
    modal.innerHTML = `
      <div class="card modal-dialog" style="width:min(480px, 100%); gap:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div class="field-label" style="margin-bottom:2px;">Navigation &amp; Controls</div>
            <strong style="font-size:16px;">Keyboard Shortcuts</strong>
          </div>
          <button type="button" class="btn btn-sm" id="btn-close-shortcuts">Esc</button>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px; font-size:13px;">
          <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border-subtle);"><span>Focus global node search</span><kbd>/</kbd></div>
          <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border-subtle);"><span>Switch views (Launch .. Report)</span><span><kbd>1</kbd> – <kbd>7</kbd></span></div>
          <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border-subtle);"><span>Navigate result cards</span><span><kbd>&larr;</kbd> <kbd>&rarr;</kbd></span></div>
          <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border-subtle);"><span>Close drawer or modal</span><kbd>Esc</kbd></div>
          <div style="display:flex; justify-content:space-between; padding:6px 0;"><span>Show this shortcuts dialog</span><kbd>?</kbd></div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    requestAnimationFrame(() => modal.classList.add("open"));
    const close = () => {
      modal.classList.remove("open");
      document.removeEventListener("keydown", onKey);
      setTimeout(() => modal.remove(), 200);
      if (prev && typeof prev.focus === "function") prev.focus({ preventScroll: true });
    };
    const onKey = (e) => {
      if (e.key === "Escape") close();
    };
    modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    document.getElementById("btn-close-shortcuts")?.addEventListener("click", close);
    document.getElementById("btn-close-shortcuts")?.focus({ preventScroll: true });
    document.addEventListener("keydown", onKey);
  }

  window.OSINT_UI = {
    switchView,
    openInspector,
    closeInspector,
    applyTheme,
    setRunState,
    toast,
    decodeEntities,
    esc,
    openShortcutsModal,
  };

  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(state.theme);
    setRunState("idle");
    initTooltipProvider();

    document.getElementById("theme-toggle")?.addEventListener("click", () => {
      applyTheme(state.theme === "dark" ? "light" : "dark");
    });

    document.getElementById("btn-shortcuts")?.addEventListener("click", openShortcutsModal);

    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.view) switchView(btn.dataset.view);
      });
    });

    document.getElementById("inspector-close")?.addEventListener("click", closeInspector);
    document.getElementById("inspector-backdrop")?.addEventListener("click", closeInspector);

    const drawerEl = document.getElementById("inspector-drawer");
    if (drawerEl) {
      let dragStartX = 0;
      let dragStartTime = 0;
      drawerEl.addEventListener("pointerdown", (e) => {
        dragStartX = e.clientX;
        dragStartTime = Date.now();
      });
      drawerEl.addEventListener("pointerup", (e) => {
        const dx = e.clientX - dragStartX;
        const dt = Math.max(1, Date.now() - dragStartTime);
        if (dx > 24 && (dx / dt > 0.11 || dx > 120)) closeInspector();
      });
      drawerEl.addEventListener("keydown", (e) => {
        if (e.key !== "Tab" || !drawerEl.classList.contains("open")) return;
        const focusables = drawerEl.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (!focusables.length) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      });
    }

    const searchInput = document.getElementById("global-search");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        state.searchQuery = e.target.value.trim().toLowerCase();
        if (typeof window.onSearchChanged === "function") {
          window.onSearchChanged(state.searchQuery);
        }
      });
      searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && state.activeView === "graph" && typeof window.focusGraphSearchNode === "function") {
          window.focusGraphSearchNode(state.searchQuery);
        }
      });
    }

    const VIEW_KEYS = ["launch", "scan", "results", "graph", "timeline", "audit", "report"];
    document.addEventListener("keydown", (e) => {
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
      const isTyping = tag === "input" || tag === "textarea" || tag === "select";

      if (e.key === "/" && !isTyping) {
        e.preventDefault();
        searchInput?.focus();
      } else if (e.key === "?" && !isTyping) {
        e.preventDefault();
        openShortcutsModal();
      } else if (!isTyping && e.key >= "1" && e.key <= "7") {
        const idx = Number(e.key) - 1;
        if (VIEW_KEYS[idx]) switchView(VIEW_KEYS[idx]);
      } else if (e.key === "Escape") {
        closeInspector();
      }
    });
  });
})();
