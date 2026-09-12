"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";
import { BarChart3, TrendingUp, Table, Award, CheckCircle2 } from "lucide-react";
import metricsData from "../public/data/metrics.json";
import { MandalaWatermark, MandalaCorner } from "./Mandala";

export default function MetricsDashboard() {
  const [activeTab, setActiveTab] = useState("charts"); // 'charts' | 'table'
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const benchmarks = metricsData.benchmarks;
  const history = metricsData.trainingHistory;

  // Transform data for Benchmark Bar Chart
  const benchmarkChartData = benchmarks.map((b) => ({
    name: b.name.replace("Ablation Run ", "Run "),
    dice: b.testDice,
    iou: b.testIoU,
    loss: b.testLoss,
  }));

  // Transform data for Spectral Gain Breakdown Chart
  const spectralGainData = [
    {
      metric: "Test Dice (F1)",
      "3-Band RGB": 0.5027,
      "4-Band (RGB + NIR)": 0.5796,
    },
    {
      metric: "Test Mean IoU",
      "3-Band RGB": 0.4245,
      "4-Band (RGB + NIR)": 0.5013,
    },
  ];

  // Custom Dark Glass Tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="p-3 rounded-xl bg-[#0b0817]/95 border border-white/20 shadow-2xl backdrop-blur-xl text-xs">
          <p className="font-bold text-white mb-1.5">{label}</p>
          {payload.map((entry, index) => (
            <div key={`item-${index}`} className="flex items-center gap-2 py-0.5" style={{ color: entry.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="text-slate-300">{entry.name}:</span>
              <span className="font-mono font-bold text-white">
                {typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <section id="metrics" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="text-center max-w-3xl mx-auto mb-10">
        <span className="pill-badge track mb-3">Empirical Results & Spectral Ablation</span>
        <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight mt-2 mb-4">
          Benchmark Evaluation Dashboard
        </h2>
        <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
          Strict deterministic evaluation across an isolated held-out test split (Seed locked at 42).
          Comparing Mandatory Baseline (BCE) against Focal-Tversky across spectral configurations.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex items-center justify-center gap-3 mb-8">
        <button
          onClick={() => setActiveTab("charts")}
          className={`px-5 py-2.5 rounded-full text-sm font-bold transition-all flex items-center gap-2 ${
            activeTab === "charts"
              ? "bg-white/20 text-white border border-white/30 shadow-lg shadow-white/10"
              : "bg-white/[0.06] text-slate-400 hover:text-white border border-white/10"
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          Visual Charts Grid
        </button>
        <button
          onClick={() => setActiveTab("table")}
          className={`px-5 py-2.5 rounded-full text-sm font-bold transition-all flex items-center gap-2 ${
            activeTab === "table"
              ? "bg-white/20 text-white border border-white/30 shadow-lg shadow-white/10"
              : "bg-white/[0.06] text-slate-400 hover:text-white border border-white/10"
          }`}
        >
          <Table className="w-4 h-4" />
          Detailed Data Table
        </button>
      </div>

      {/* Tab 1: Charts Grid */}
      {activeTab === "charts" && (
        <div className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Chart 1: Model Architecture Benchmark */}
            <div className="glass-card p-6 relative overflow-hidden">
              <MandalaCorner position="tl" />
              <MandalaWatermark position="bottom-right" size={220} opacity={0.04} />
              <div className="flex items-center justify-between mb-4 relative z-10">
                <div className="flex items-center gap-2">
                  <Award className="w-5 h-5 text-brand-pink" />
                  <h3 className="text-base sm:text-lg font-bold text-white tracking-tight">
                    Model Architecture Benchmark (Held-out Test N=40)
                  </h3>
                </div>
                <span className="pill-badge track">Dice &amp; IoU</span>
              </div>
              <p className="text-xs text-slate-300 mb-6 relative z-10">
                Comparative test metrics comparing 4-Band BCE Baseline vs 3-Band and 4-Band Focal-Tversky configurations.
              </p>

              <div className="h-72 w-full relative z-10">
                {mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={benchmarkChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="name" stroke="#94a3b8" tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                      <YAxis stroke="#94a3b8" domain={[0, 1]} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend
                        wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }}
                        iconType="circle"
                      />
                      <Bar dataKey="dice" name="Test Dice (F1)" fill="#ff4b72" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="iou" name="Test Mean IoU" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="loss" name="Test Loss" fill="#06b6d4" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            {/* Chart 2: Spectral Gain Breakdown */}
            <div className="glass-card p-6 relative overflow-hidden">
              <MandalaCorner position="tr" />
              <MandalaWatermark position="bottom-left" size={220} opacity={0.04} />
              <div className="flex items-center justify-between mb-4 relative z-10">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-brand-emerald" />
                  <h3 className="text-base sm:text-lg font-bold text-white tracking-tight">
                    NIR Spectral Gain (3-Band vs 4-Band)
                  </h3>
                </div>
                <span className="pill-badge success">+7.69% Absolute Gain</span>
              </div>
              <p className="text-xs text-slate-300 mb-6">
                Quantifying the physical penetration gain enabled by the Sentinel-2 Band 8 (NIR, 842nm) channel.
              </p>

              <div className="h-72 w-full">
                {mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={spectralGainData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="metric" stroke="#94a3b8" tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                      <YAxis stroke="#94a3b8" domain={[0, 0.8]} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend
                        wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }}
                        iconType="circle"
                      />
                      <Bar dataKey="3-Band RGB" name="3-Band RGB Only" fill="rgba(148, 163, 184, 0.5)" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="4-Band (RGB + NIR)" name="4-Band (RGB + NIR)" fill="#10b981" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>

          {/* Chart 3: 15-Epoch Training Progression (Full Width) */}
          <div className="glass-card p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-brand-purple" />
                <h3 className="text-base sm:text-lg font-bold text-white tracking-tight">
                  15-Epoch Training Progression &amp; Validation Convergence
                </h3>
              </div>
              <span className="pill-badge seed self-start sm:self-auto">
                &star; Peak Model Checkpoint at Epoch 5
              </span>
            </div>
            <p className="text-xs text-slate-300 mb-6">
              Evaluation metrics across 15 full training epochs. Early stopping checkpoint preserved at Epoch 5 (Val Dice: 0.6924, Val IoU: 0.6100).
            </p>

            <div className="h-80 w-full">
              {mounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis
                      dataKey="epoch"
                      stroke="#94a3b8"
                      tickFormatter={(e) => `Ep ${e}`}
                      tick={{ fill: "#cbd5e1", fontSize: 11 }}
                    />
                    <YAxis stroke="#94a3b8" domain={[0, 0.8]} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                      wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }}
                      iconType="circle"
                    />
                    <Line
                      type="monotone"
                      dataKey="valDice"
                      name="Validation Dice (F1)"
                      stroke="#ff4b72"
                      strokeWidth={2.5}
                      dot={{ r: 3.5, fill: "#ff4b72", stroke: "#ffffff", strokeWidth: 1 }}
                      activeDot={{ r: 7 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="valIoU"
                      name="Validation IoU"
                      stroke="#8b5cf6"
                      strokeWidth={2}
                      dot={{ r: 3, fill: "#8b5cf6" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="trainLoss"
                      name="Training Loss"
                      stroke="rgba(255,255,255,0.4)"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="valLoss"
                      name="Validation Loss"
                      stroke="#06b6d4"
                      strokeWidth={1.8}
                      dot={{ r: 2.5, fill: "#06b6d4" }}
                    />
                    {/* Mark peak checkpoint */}
                    <ReferenceDot
                      x={5}
                      y={0.6924}
                      r={8}
                      fill="#facc15"
                      stroke="#ffffff"
                      strokeWidth={2}
                      label={{
                        value: "Peak (Ep 5)",
                        position: "top",
                        fill: "#facc15",
                        fontSize: 11,
                        fontWeight: 700,
                      }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Structured Benchmark Table */}
      {activeTab === "table" && (
        <div className="glass-panel overflow-hidden border border-white/20 shadow-2xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm border-collapse">
              <thead>
                <tr className="border-b border-white/20 bg-white/[0.04] text-slate-300">
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Experiment Name</th>
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Input Bands</th>
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Objective Loss</th>
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Val Dice</th>
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Val Mean IoU</th>
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Test Dice</th>
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Test Mean IoU</th>
                  <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Test Loss</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/10 font-mono text-slate-200">
                {/* Mandatory Baseline Winner */}
                <tr className="bg-brand-pink/15 hover:bg-brand-pink/20 transition-colors">
                  <td className="py-3.5 px-4 font-sans font-extrabold text-white flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-brand-pink" />
                    Mandatory Baseline
                    <span className="pill-badge track text-[9px] py-0.5 px-1.5 ml-1">Winner</span>
                  </td>
                  <td className="py-3.5 px-4">4 (RGB + NIR)</td>
                  <td className="py-3.5 px-4 font-sans">BCEWithLogitsLoss</td>
                  <td className="py-3.5 px-4">0.6924</td>
                  <td className="py-3.5 px-4">0.6100</td>
                  <td className="py-3.5 px-4 font-bold text-[#ff758c] text-sm">0.7834</td>
                  <td className="py-3.5 px-4 font-bold text-[#34d399] text-sm">0.7044</td>
                  <td className="py-3.5 px-4">0.3781</td>
                </tr>

                {/* Ablation Run 1 */}
                <tr className="hover:bg-white/[0.04] transition-colors">
                  <td className="py-3.5 px-4 font-sans font-bold text-slate-200">Ablation Run 1 (3-Band)</td>
                  <td className="py-3.5 px-4">3 (RGB-Only)</td>
                  <td className="py-3.5 px-4 font-sans">Focal-Tversky (&alpha;=0.7, &beta;=0.3, &gamma;=1.333)</td>
                  <td className="py-3.5 px-4">0.4414</td>
                  <td className="py-3.5 px-4">0.3653</td>
                  <td className="py-3.5 px-4">0.5027</td>
                  <td className="py-3.5 px-4">0.4245</td>
                  <td className="py-3.5 px-4">0.5154</td>
                </tr>

                {/* Ablation Run 2 */}
                <tr className="hover:bg-white/[0.04] transition-colors">
                  <td className="py-3.5 px-4 font-sans font-bold text-slate-200">Ablation Run 2 (4-Band)</td>
                  <td className="py-3.5 px-4">4 (RGB + NIR)</td>
                  <td className="py-3.5 px-4 font-sans">Focal-Tversky (&alpha;=0.7, &beta;=0.3, &gamma;=1.333)</td>
                  <td className="py-3.5 px-4">0.4840</td>
                  <td className="py-3.5 px-4">0.4034</td>
                  <td className="py-3.5 px-4">0.5796</td>
                  <td className="py-3.5 px-4">0.5013</td>
                  <td className="py-3.5 px-4">0.5102</td>
                </tr>

                {/* Empirical Physical Delta */}
                <tr className="bg-emerald-500/10 font-bold text-emerald-300">
                  <td className="py-3.5 px-4 font-sans">NIR Band Contribution (&Delta;)</td>
                  <td className="py-3.5 px-4">+1 Band (NIR)</td>
                  <td className="py-3.5 px-4 font-sans">Empirical Physical Delta</td>
                  <td className="py-3.5 px-4">+0.0426</td>
                  <td className="py-3.5 px-4">+0.0381</td>
                  <td className="py-3.5 px-4 text-emerald-300 text-sm">+0.0769 (+15.3%)</td>
                  <td className="py-3.5 px-4 text-emerald-300 text-sm">+0.0768 (+18.1%)</td>
                  <td className="py-3.5 px-4">-0.0052</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
