/**
 * Ilādṛṣṭi Metrics & Ablation Dashboard Charts
 * Powered by Chart.js with glowing dark-glass theme customization
 */

class DashboardCharts {
  constructor(data) {
    this.data = data || window.ILADRISTI_DATA;
    this.benchmarkChart = null;
    this.trainingChart = null;
    this.spectralChart = null;

    this.init();
  }

  init() {
    if (typeof Chart === "undefined") {
      console.warn("Chart.js not loaded. Retrying in 500ms...");
      setTimeout(() => this.init(), 500);
      return;
    }

    // Configure global Chart.js defaults
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(15, 12, 28, 0.95)";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(255, 255, 255, 0.2)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.titleFont = { size: 13, weight: "700" };
    Chart.defaults.plugins.tooltip.bodyFont = { size: 12, weight: "500" };

    this.renderBenchmarkChart();
    this.renderTrainingChart();
    this.renderSpectralChart();
  }

  renderBenchmarkChart() {
    const ctx = document.getElementById("benchmarkChart");
    if (!ctx) return;

    const benchmarks = this.data.benchmarks;
    const labels = benchmarks.map((b) => b.name);
    const diceScores = benchmarks.map((b) => b.testDice);
    const iouScores = benchmarks.map((b) => b.testIoU);
    const losses = benchmarks.map((b) => b.testLoss);

    this.benchmarkChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Test Dice (F1)",
            data: diceScores,
            backgroundColor: "rgba(255, 75, 114, 0.8)",
            borderColor: "#ff4b72",
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: "Test Mean IoU",
            data: iouScores,
            backgroundColor: "rgba(139, 92, 246, 0.8)",
            borderColor: "#8b5cf6",
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: "Test Loss",
            data: losses,
            backgroundColor: "rgba(6, 182, 212, 0.7)",
            borderColor: "#06b6d4",
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.06)" },
            ticks: { color: "#e2e8f0", font: { weight: "600" } },
          },
          y: {
            beginAtZero: true,
            max: 1.0,
            grid: { color: "rgba(255, 255, 255, 0.08)" },
            ticks: {
              stepSize: 0.2,
              callback: (val) => val.toFixed(2),
            },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: "#f8fafc",
              usePointStyle: true,
              pointStyle: "circle",
              padding: 16,
            },
          },
          tooltip: {
            callbacks: {
              label: (context) => `${context.dataset.label}: ${context.raw.toFixed(4)}`,
            },
          },
        },
      },
    });
  }

  renderTrainingChart() {
    const ctx = document.getElementById("trainingChart");
    if (!ctx) return;

    const history = this.data.trainingHistory;
    const epochs = history.map((h) => `Epoch ${h.epoch}`);
    const trainLoss = history.map((h) => h.trainLoss);
    const valLoss = history.map((h) => h.valLoss);
    const valDice = history.map((h) => h.valDice);
    const valIoU = history.map((h) => h.valIoU);

    this.trainingChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          {
            label: "Validation Dice (F1)",
            data: valDice,
            borderColor: "#ff4b72",
            backgroundColor: "rgba(255, 75, 114, 0.15)",
            borderWidth: 2.5,
            tension: 0.35,
            fill: false,
            pointBackgroundColor: history.map((h) => (h.isBest ? "#facc15" : "#ff4b72")),
            pointBorderColor: "#ffffff",
            pointRadius: history.map((h) => (h.isBest ? 7 : 3.5)),
            pointHoverRadius: 8,
          },
          {
            label: "Validation IoU",
            data: valIoU,
            borderColor: "#8b5cf6",
            borderWidth: 2,
            tension: 0.35,
            fill: false,
            pointBackgroundColor: history.map((h) => (h.isBest ? "#facc15" : "#8b5cf6")),
            pointRadius: history.map((h) => (h.isBest ? 6 : 3)),
          },
          {
            label: "Training Loss",
            data: trainLoss,
            borderColor: "rgba(255, 255, 255, 0.35)",
            borderWidth: 1.5,
            borderDash: [4, 4],
            tension: 0.3,
            pointRadius: 0,
            fill: false,
          },
          {
            label: "Validation Loss",
            data: valLoss,
            borderColor: "#06b6d4",
            borderWidth: 1.8,
            tension: 0.35,
            pointRadius: 2.5,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#cbd5e1", maxRotation: 0 },
          },
          y: {
            beginAtZero: true,
            max: 0.8,
            grid: { color: "rgba(255, 255, 255, 0.08)" },
            ticks: {
              stepSize: 0.1,
              callback: (val) => val.toFixed(1),
            },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: "#f8fafc",
              usePointStyle: true,
              pointStyle: "circle",
              padding: 16,
            },
          },
          tooltip: {
            callbacks: {
              afterBody: (tooltipItems) => {
                const idx = tooltipItems[0].dataIndex;
                if (history[idx].isBest) {
                  return "★ Peak Model Saved (Val Dice: 0.6924, IoU: 0.6100)";
                }
                return "";
              },
            },
          },
        },
      },
    });
  }

  renderSpectralChart() {
    const ctx = document.getElementById("spectralChart");
    if (!ctx) return;

    this.spectralChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Test Dice (F1)", "Test Mean IoU"],
        datasets: [
          {
            label: "3-Band RGB Only",
            data: [0.5027, 0.4245],
            backgroundColor: "rgba(148, 163, 184, 0.4)",
            borderColor: "rgba(148, 163, 184, 0.8)",
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: "4-Band (RGB + NIR)",
            data: [0.5796, 0.5013],
            backgroundColor: "rgba(16, 185, 129, 0.8)",
            borderColor: "#10b981",
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        scales: {
          x: {
            beginAtZero: true,
            max: 0.7,
            grid: { color: "rgba(255, 255, 255, 0.08)" },
            ticks: { color: "#e2e8f0" },
          },
          y: {
            grid: { display: false },
            ticks: { color: "#f8fafc", font: { weight: "700", size: 13 } },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#f8fafc", usePointStyle: true, padding: 14 },
          },
          tooltip: {
            callbacks: {
              afterLabel: (context) => {
                if (context.datasetIndex === 1) {
                  return context.dataIndex === 0
                    ? "Δ Gain: +0.0769 (+15.3% relative boost)"
                    : "Δ Gain: +0.0768 (+18.1% relative boost)";
                }
                return "";
              },
            },
          },
        },
      },
    });
  }
}

// Export
window.DashboardCharts = DashboardCharts;
