/**
 * Ilādṛṣṭi Interactive Before/After Split Image Slider Engine
 * Supports fluid mouse/touch dragging, keyboard arrows, and auto-sweep animation.
 */

class BeforeAfterSlider {
  constructor(options = {}) {
    this.container = document.getElementById(options.containerId || "comparisonBox");
    this.afterImg = document.getElementById(options.afterImgId || "sliderAfterImg");
    this.beforeImg = document.getElementById(options.beforeImgId || "sliderBeforeImg");
    this.handleLine = document.getElementById(options.handleLineId || "sliderHandleLine");
    this.handleBtn = document.getElementById(options.handleBtnId || "sliderHandleBtn");
    this.leftBadge = document.getElementById(options.leftBadgeId || "sliderLeftBadge");
    this.rightBadge = document.getElementById(options.rightBadgeId || "sliderRightBadge");
    this.confusionLegend = document.getElementById(options.legendId || "confusionLegend");
    this.scanBtn = document.getElementById(options.scanBtnId || "btnAutoScan");

    this.position = 50.0; // Percentage 0 - 100
    this.isDragging = false;
    this.isScanning = false;
    this.scanDirection = 1; // 1 = right, -1 = left
    this.scanSpeed = 0.35; // % per frame
    this.animationFrame = null;

    this.currentScene = null;
    this.currentMode = "rgb_vs_pred"; // 'rgb_vs_pred', 'nir_vs_pred', 'pred_vs_gt', 'rgb_vs_confusion'

    this.init();
  }

  init() {
    if (!this.container || !this.afterImg || !this.handleLine) {
      console.warn("Slider DOM elements not found.");
      return;
    }

    // Pointer event listeners on container for fluid dragging
    this.container.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    window.addEventListener("pointermove", (e) => this.onPointerMove(e));
    window.addEventListener("pointerup", () => this.onPointerUp());
    window.addEventListener("pointercancel", () => this.onPointerUp());

    // Keyboard accessibility
    this.container.setAttribute("tabindex", "0");
    this.container.addEventListener("keydown", (e) => this.onKeyDown(e));

    // Auto-scan button
    if (this.scanBtn) {
      this.scanBtn.addEventListener("click", () => this.toggleAutoScan());
    }

    // Set initial position
    this.setPosition(50.0);
  }

  onPointerDown(e) {
    this.isDragging = true;
    if (this.isScanning) {
      this.stopAutoScan();
    }
    this.updatePositionFromPointer(e);
  }

  onPointerMove(e) {
    if (!this.isDragging) return;
    this.updatePositionFromPointer(e);
  }

  onPointerUp() {
    this.isDragging = false;
  }

  updatePositionFromPointer(e) {
    const rect = this.container.getBoundingClientRect();
    const clientX = e.clientX;
    let newPos = ((clientX - rect.left) / rect.width) * 100.0;
    this.setPosition(newPos);
  }

  onKeyDown(e) {
    if (e.key === "ArrowLeft") {
      this.setPosition(this.position - 5);
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      this.setPosition(this.position + 5);
      e.preventDefault();
    } else if (e.key === "Home") {
      this.setPosition(0);
      e.preventDefault();
    } else if (e.key === "End") {
      this.setPosition(100);
      e.preventDefault();
    }
  }

  setPosition(percent) {
    this.position = Math.max(0.0, Math.min(100.0, percent));
    const pStr = this.position.toFixed(2);

    // Apply clip-path to after image (shows from left edge to position)
    this.afterImg.style.clipPath = `polygon(0 0, ${pStr}% 0, ${pStr}% 100%, 0 100%)`;

    // Move handle line
    this.handleLine.style.left = `${pStr}%`;
  }

  toggleAutoScan() {
    if (this.isScanning) {
      this.stopAutoScan();
    } else {
      this.startAutoScan();
    }
  }

  startAutoScan() {
    this.isScanning = true;
    if (this.scanBtn) {
      this.scanBtn.classList.add("scanning");
      this.scanBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="4" width="4" height="16"/>
          <rect x="14" y="4" width="4" height="16"/>
        </svg>
        Pause Scan
      `;
    }

    const animate = () => {
      if (!this.isScanning) return;

      this.position += this.scanDirection * this.scanSpeed;
      if (this.position >= 92) {
        this.position = 92;
        this.scanDirection = -1;
      } else if (this.position <= 8) {
        this.position = 8;
        this.scanDirection = 1;
      }

      this.setPosition(this.position);
      this.animationFrame = requestAnimationFrame(animate);
    };

    this.animationFrame = requestAnimationFrame(animate);
  }

  stopAutoScan() {
    this.isScanning = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    if (this.scanBtn) {
      this.scanBtn.classList.remove("scanning");
      this.scanBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Auto Scan
      `;
    }
  }

  setScene(scene) {
    this.currentScene = scene;
    this.updateLayers();
  }

  setMode(mode) {
    this.currentMode = mode;
    this.updateLayers();
  }

  updateLayers() {
    if (!this.currentScene) return;

    const assets = this.currentScene.assets;
    let leftImgSrc = "";
    let rightImgSrc = "";
    let leftLabel = "";
    let rightLabel = "";
    let showConfusionLegend = false;

    switch (this.currentMode) {
      case "rgb_vs_pred":
        leftImgSrc = assets.rgb;
        rightImgSrc = assets.overlay;
        leftLabel = "Raw Sentinel-2 RGB";
        rightLabel = "Predicted Wildfire Mask Overlay";
        break;

      case "nir_vs_pred":
        leftImgSrc = assets.nir;
        rightImgSrc = assets.overlay;
        leftLabel = "False-Color Infrared (NIR-R-G)";
        rightLabel = "Predicted Wildfire Mask Overlay";
        break;

      case "pred_vs_gt":
        leftImgSrc = assets.pred;
        rightImgSrc = assets.gt;
        leftLabel = "Model Prediction (Focal-Tversky)";
        rightLabel = "Ground Truth Scar (Reference)";
        break;

      case "rgb_vs_confusion":
        leftImgSrc = assets.rgb;
        rightImgSrc = assets.confusion;
        leftLabel = "Raw Satellite RGB";
        rightLabel = "Spatial Confusion Map (TP / FP / FN)";
        showConfusionLegend = true;
        break;

      default:
        leftImgSrc = assets.rgb;
        rightImgSrc = assets.overlay;
        leftLabel = "Raw RGB";
        rightLabel = "Prediction Overlay";
    }

    // Set images
    this.beforeImg.src = leftImgSrc;
    this.afterImg.src = rightImgSrc;

    // Set badges
    if (this.leftBadge) this.leftBadge.textContent = leftLabel;
    if (this.rightBadge) this.rightBadge.textContent = rightLabel;

    // Toggle confusion map legend
    if (this.confusionLegend) {
      if (showConfusionLegend) {
        this.confusionLegend.classList.add("show");
      } else {
        this.confusionLegend.classList.remove("show");
      }
    }
  }
}

// Export
window.BeforeAfterSlider = BeforeAfterSlider;
