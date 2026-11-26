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

  function openInspector(nodeId) {
    state.selectedNodeId = nodeId || null;
    const drawer = document.getElementById("inspector-drawer");
    if (drawer) {
      drawer.classList.add("open");
      drawer.setAttribute("aria-hidden", "false");
    }
    if (typeof window.renderInspector === "function") {
      window.renderInspector(nodeId);
    }
  }

  function closeInspector() {
    const drawer = document.getElementById("inspector-drawer");
    if (drawer) {
      drawer.classList.remove("open");
      drawer.setAttribute("aria-hidden", "true");
    }
  }

  window.OSINT_UI = {
    switchView,
    openInspector,
    closeInspector,
    applyTheme,
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
