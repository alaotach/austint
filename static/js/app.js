/* Intelligent OSINT Dashboard Controller (M1 App Shell + Navigation + Theme + Keyboard) */

(function () {
  "use strict";

  const state = {
    activeView: "launch",
    theme: localStorage.getItem("osint_theme") || "dark",
    jobId: null,
    nodes: [],
    edges: [],
    auditEvents: [],
    selectedNodeId: null,
    searchQuery: "",
  };

  window.OSINT_STATE = state;

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
      btn.classList.toggle("active", btn.dataset.view === viewName);
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

  function toast(message, variant = "info") {
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
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "btn btn-sm";
    closeBtn.textContent = "x";
    closeBtn.setAttribute("aria-label", "Dismiss notification");
    el.append(span, closeBtn);
    stack.appendChild(el);
    requestAnimationFrame(() => el.classList.add("open"));

    const dismiss = () => {
      el.classList.remove("open");
      setTimeout(() => el.remove(), 220);
    };
    let timer = setTimeout(dismiss, 3500);
    el.addEventListener("mouseenter", () => clearTimeout(timer));
    el.addEventListener("mouseleave", () => { timer = setTimeout(dismiss, 1800); });
    closeBtn.addEventListener("click", dismiss);

    // Swipe-to-dismiss with momentum (emil-design-eng)
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

  window.OSINT_UI = {
    switchView,
    openInspector,
    closeInspector,
    applyTheme,
    toast,
  };

  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(state.theme);

    const themeBtn = document.getElementById("theme-toggle");
    if (themeBtn) {
      themeBtn.addEventListener("click", () => {
        applyTheme(state.theme === "dark" ? "light" : "dark");
      });
    }

    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.view) {
          switchView(btn.dataset.view);
        }
      });
    });

    const closeBtn = document.getElementById("inspector-close");
    if (closeBtn) {
      closeBtn.addEventListener("click", closeInspector);
    }
    document.getElementById("inspector-backdrop")?.addEventListener("click", closeInspector);

    // Drawer swipe-right-to-dismiss with velocity check & focus trap
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
    }

    document.addEventListener("keydown", (e) => {
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
      const isTyping = tag === "input" || tag === "textarea" || tag === "select";

      if (e.key === "/" && !isTyping) {
        e.preventDefault();
        if (searchInput) searchInput.focus();
      } else if (e.key === "Escape") {
        closeInspector();
      }
    });
  });
})();
