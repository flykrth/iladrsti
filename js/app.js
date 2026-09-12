/**
 * Ilādṛṣṭi Web Dashboard - Main Application Controller
 * Manages scene switching, telemetry HUD updates, KaTeX rendering,
 * navigation scrollspy, and interactive UI states.
 */

document.addEventListener("DOMContentLoaded", () => {
  const data = window.ILADRISTI_DATA;
  if (!data) {
    console.error("Ilādṛṣṭi dataset not found in window.ILADRISTI_DATA");
    return;
  }

  // 1. Initialize Before/After Slider
  const slider = new BeforeAfterSlider({
    containerId: "comparisonBox",
    afterImgId: "sliderAfterImg",
    beforeImgId: "sliderBeforeImg",
    handleLineId: "sliderHandleLine",
    handleBtnId: "sliderHandleBtn",
    leftBadgeId: "sliderLeftBadge",
    rightBadgeId: "sliderRightBadge",
    legendId: "confusionLegend",
    scanBtnId: "btnAutoScan",
  });

  // 2. Initialize Charts
  const charts = new DashboardCharts(data);

  // 3. Current State
  let activeSceneIndex = 0;
  const scenes = data.scenes;

  // 4. Setup Telemetry HUD Elements
  const hudSceneName = document.getElementById("hudSceneName");
  const hudSceneDesc = document.getElementById("hudSceneDesc");
  const hudDice = document.getElementById("hudDice");
  const hudIoU = document.getElementById("hudIoU");
  const hudPrecision = document.getElementById("hudPrecision");
  const hudRecall = document.getElementById("hudRecall");
  const hudArea = document.getElementById("hudArea");

  // Function to update HUD
  function updateHUD(scene) {
    if (!scene) return;
    if (hudSceneName) hudSceneName.textContent = `${scene.name} (${scene.tile})`;
    if (hudSceneDesc) hudSceneDesc.textContent = `${scene.location} • ${scene.description}`;
    if (hudDice) hudDice.textContent = scene.metrics.dice.toFixed(4);
    if (hudIoU) hudIoU.textContent = scene.metrics.iou.toFixed(4);
    if (hudPrecision) hudPrecision.textContent = `${(scene.metrics.precision * 100).toFixed(1)}%`;
    if (hudRecall) hudRecall.textContent = `${(scene.metrics.recall * 100).toFixed(1)}%`;
    if (hudArea) hudArea.textContent = `${scene.metrics.scarAreaPct.toFixed(1)}%`;
  }

  // Function to select a scene
  function selectScene(index) {
    if (index < 0 || index >= scenes.length) return;
    activeSceneIndex = index;
    const scene = scenes[index];

    // Update buttons active state
    document.querySelectorAll(".scene-btn").forEach((btn, idx) => {
      if (idx === index) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    // Update slider and HUD
    slider.setScene(scene);
    updateHUD(scene);
  }

  // Bind Scene Selector Buttons
  const sceneButtons = document.querySelectorAll(".scene-btn");
  sceneButtons.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(btn.getAttribute("data-scene-idx"), 10);
      if (!isNaN(idx)) {
        selectScene(idx);
      }
    });
  });

  // Bind Layer Mode Selector Buttons
  const modeButtons = document.querySelectorAll(".mode-btn");
  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.getAttribute("data-mode");
      if (mode) {
        slider.setMode(mode);
      }
    });
  });

  // Select first scene initially
  selectScene(0);

  // 5. Render Math with KaTeX
  renderMathematicalFormulas();

  // 6. Navigation Scrollspy and Navbar Styling
  setupNavigation();

  // 7. Tab Switching in Metrics Dashboard
  setupDashboardTabs();
});

/**
 * Render Mathematical Formulations using KaTeX if available
 */
function renderMathematicalFormulas() {
  if (typeof katex !== "undefined") {
    try {
      // 1. Focal-Tversky Loss
      const ftlElem = document.getElementById("katexFTL");
      if (ftlElem) {
        katex.render(
          "FTL = (1 - TI)^\\gamma = \\left(1 - \\frac{TP + \\epsilon}{TP + \\alpha FN + \\beta FP + \\epsilon}\\right)^{1.333}",
          ftlElem,
          { displayMode: true, throwOnError: false }
        );
      }

      // 2. Tversky Index
      const tiElem = document.getElementById("katexTI");
      if (tiElem) {
        katex.render(
          "TI = \\frac{\\sum_{i} p_i y_i + \\epsilon}{\\sum_{i} p_i y_i + 0.7 \\sum_{i} (1 - p_i) y_i + 0.3 \\sum_{i} p_i (1 - y_i) + \\epsilon}",
          tiElem,
          { displayMode: true, throwOnError: false }
        );
      }

      // 3. Conv1 Weight Adaptation
      const convElem = document.getElementById("katexConv");
      if (convElem) {
        katex.render(
          "\\mathbf{W}_{\\text{conv1}}[:, 3:4, :, :] \\leftarrow \\frac{1}{3} \\sum_{c=0}^{2} \\mathbf{W}_{\\text{ImageNet}}[:, c:c+1, :, :]",
          convElem,
          { displayMode: true, throwOnError: false }
        );
      }
    } catch (e) {
      console.warn("KaTeX render error:", e);
    }
  }
}

/**
 * Setup navigation scrollspy, glass blur on scroll, and mobile hamburger menu
 */
function setupNavigation() {
  const navbar = document.querySelector(".navbar");
  const navLinks = document.querySelectorAll(".nav-link");
  const sections = document.querySelectorAll("section[id]");
  const mobileToggle = document.getElementById("mobileNavToggle");
  const navList = document.getElementById("navLinksList");

  // Scroll effect on navbar
  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }

    // Scrollspy
    let current = "";
    sections.forEach((section) => {
      const sectionTop = section.offsetTop - 140;
      const sectionHeight = section.offsetHeight;
      if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
        current = section.getAttribute("id");
      }
    });

    navLinks.forEach((link) => {
      link.classList.remove("active");
      if (link.getAttribute("href") === `#${current}`) {
        link.classList.add("active");
      }
    });
  });

  // Mobile menu toggle
  if (mobileToggle && navList) {
    mobileToggle.addEventListener("click", () => {
      navList.classList.toggle("open");
    });

    // Close when clicking a link
    navLinks.forEach((link) => {
      link.addEventListener("click", () => {
        navList.classList.remove("open");
      });
    });
  }
}

/**
 * Setup Tab Switching for Metrics & Ablation Dashboard
 */
function setupDashboardTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => (p.style.display = "none"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab-target");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.style.display = "block";
      }
    });
  });
}
