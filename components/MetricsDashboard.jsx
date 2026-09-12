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

  // Custom White & Gold Glass Tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="p-3 rounded-xl bg-white/95 border border-[#d4af37]/45 shadow-xl backdrop-blur-xl text-xs text-slate-900">
          <p className="font-bold text-slate-900 mb-1.5">{label}</p>
          {payload.map((entry, index) => (
            <div key={`item-${index}`} className="flex items-center gap-2 py-0.5" style={{ color: entry.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="text-slate-600">{entry.name}:</span>
              <span className="font-mono font-bold text-slate-900">
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
    <section id="metrics" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto relative">
      {/* Ambient Section Background Mandala Watermark */}
      <MandalaWatermark position="center" size={680} opacity={0.10} />

      {/* Section Header */}
      <div className="text-center max-w-3xl mx-auto mb-10 relative z-10">
        <span className="pill-badge track mb-3">Empirical Results &amp; Spectral Ablation</span>
        <h2 className="text-3xl sm:text-4xl font-black text-slate-900 tracking-tight mt-2 mb-4">
          Benchmark Evaluation Dashboard
        </h2>
        <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
          Strict deterministic evaluation across an isolated held-out test split (Seed locked at 42).
          Comparing Mandatory Baseline (BCE) against Focal-Tversky across spectral configurations.
        </p>
      </div>

      {/* Tabs in White & Gold */}
      <div className="flex items-center justify-center gap-3 mb-8 relative z-10">
        <button
          onClick={() => setActiveTab("charts")}
          className={`px-5 py-2.5 rounded-full text-sm font-bold transition-all flex items-center gap-2 ${
            activeTab === "charts"
              ? "bg-[#d4af37] text-white border border-[#d4af37] shadow-md shadow-[#d4af37]/25"
              : "bg-amber-50/50 text-slate-600 hover:text-slate-900 border border-[#d4af37]/35 hover:bg-amber-50/80"
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          Visual Charts Grid
        </button>
        <button
          onClick={() => setActiveTab("table")}
          className={`px-5 py-2.5 rounded-full text-sm font-bold transition-all flex items-center gap-2 ${
            activeTab === "table"
              ? "bg-[#d4af37] text-white border border-[#d4af37] shadow-md shadow-[#d4af37]/25"
              : "bg-amber-50/50 text-slate-600 hover:text-slate-900 border border-[#d4af37]/35 hover:bg-amber-50/80"
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
            <div className="glass-card p-6 relative overflow-hidden bg-white/92 border border-[#d4af37]/35 shadow-xl">
              <MandalaCorner position="tl" />
              <MandalaWatermark position="bottom-right" size={240} opacity={0.12} />
              <div className="flex items-center justify-between mb-4 relative z-10">
                <div className="flex items-center gap-2">
                  <Award className="w-5 h-5 text-[#b45309]" />
                  <h3 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                    Model Architecture Benchmark (Held-out Test N=40)
                  </h3>
                </div>
                <span className="pill-badge track">Dice &amp; IoU</span>
              </div>
              <p className="text-xs text-slate-600 mb-6 relative z-10">
                Comparative test metrics comparing 4-Band BCE Baseline vs 3-Band and 4-Band Focal-Tversky configurations.
              </p>

              <div className="h-72 w-full relative z-10">
                {mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={benchmarkChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(212, 175, 55, 0.18)" />
                      <XAxis dataKey="name" stroke="#64748b" tick={{ fill: "#334155", fontSize: 11 }} />
                      <YAxis stroke="#64748b" domain={[0, 1]} tick={{ fill: "#334155", fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend
                        wrapperStyle={{ fontSize: "12px", paddingTop: "10px", color: "#334155" }}
                        iconType="circle"
                      />
                      <Bar dataKey="dice" name="Test Dice (F1)" fill="#b45309" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="iou" name="Test Mean IoU" fill="#d4af37" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="loss" name="Test Loss" fill="#0284c7" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            {/* Chart 2: Spectral Gain Breakdown */}
            <div className="glass-card p-6 relative overflow-hidden bg-white/92 border border-[#d4af37]/35 shadow-xl">
              <MandalaCorner position="tr" />
              <MandalaWatermark position="bottom-left" size={240} opacity={0.12} />
              <div className="flex items-center justify-between mb-4 relative z-10">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-[#047857]" />
                  <h3 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                    NIR Spectral Gain (3-Band vs 4-Band)
                  </h3>
                </div>
                <span className="pill-badge success">+7.69% Absolute Gain</span>
              </div>
              <p className="text-xs text-slate-600 mb-6 relative z-10">
                Quantifying the physical penetration gain enabled by the Sentinel-2 Band 8 (NIR, 842nm) channel.
              </p>

              <div className="h-72 w-full relative z-10">
                {mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={spectralGainData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(212, 175, 55, 0.18)" />
                      <XAxis dataKey="metric" stroke="#64748b" tick={{ fill: "#334155", fontSize: 11 }} />
                      <YAxis stroke="#64748b" domain={[0, 0.8]} tick={{ fill: "#334155", fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend
                        wrapperStyle={{ fontSize: "12px", paddingTop: "10px", color: "#334155" }}
                        iconType="circle"
                      />
                      <Bar dataKey="3-Band RGB" name="3-Band RGB Only" fill="rgba(180, 83, 9, 0.35)" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="4-Band (RGB + NIR)" name="4-Band (RGB + NIR)" fill="#b45309" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>

          {/* Chart 3: 15-Epoch Training Progression */}
          <div className="glass-card p-6 sm:p-8 bg-white/92 border border-[#d4af37]/35 shadow-xl relative overflow-hidden">
            <MandalaCorner position="tl" />
            <MandalaCorner position="br" />
            <MandalaWatermark position="center" size={380} opacity={0.08} />

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 relative z-10">
              <div>
                <h3 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                  15-Epoch Training Convergence Progression
                </h3>
                <p className="text-xs text-slate-600 mt-0.5">
                  AdamW Optimizer &bull; Cosine Annealing LR (1e-4 &rarr; 1e-6) &bull; Checkpointed at Peak Val Dice
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="pill-badge track">Best Checkpoint: Epoch 5</span>
              </div>
            </div>

            <div className="h-80 w-full relative z-10">
              {mounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 15, right: 15, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(212, 175, 55, 0.18)" />
                    <XAxis dataKey="epoch" stroke="#64748b" tick={{ fill: "#334155", fontSize: 11 }} label={{ value: "Training Epoch", position: "insideBottomRight", offset: -5, fill: "#64748b", fontSize: 11 }} />
                    <YAxis stroke="#64748b" domain={[0, 1]} tick={{ fill: "#334155", fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "12px", color: "#334155" }} iconType="circle" />
                    <Line type="monotone" dataKey="valDice" name="Validation Dice (F1)" stroke="#b45309" strokeWidth={2.5} dot={{ r: 3.5, fill: "#b45309" }} activeDot={{ r: 6 }} />
                    <Line type="monotone" dataKey="valIoU" name="Validation IoU" stroke="#d4af37" strokeWidth={2} dot={{ r: 3, fill: "#d4af37" }} />
                    <Line type="monotone" dataKey="trainLoss" name="Training Loss" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                    <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke="#0284c7" strokeWidth={1.8} dot={{ r: 2.5, fill: "#0284c7" }} />
                    <ReferenceDot x={5} y={0.6924} r={7} fill="#b45309" stroke="#ffffff" strokeWidth={2} label={{ value: "Peak (Ep 5)", position: "top", fill: "#b45309", fontSize: 11, fontWeight: 700 }} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Structured Benchmark Table (White & Gold) */}
      {activeTab === "table" && (
        <div className="glass-panel overflow-hidden border border-[#d4af37]/35 shadow-xl bg-white/95 relative z-10">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm border-collapse">
              <thead>
                <tr className="border-b border-[#d4af37]/30 bg-amber-50/70 text-[#854d0e]">
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
              <tbody className="divide-y divide-[#d4af37]/15 font-mono text-slate-700">
                {/* Mandatory Baseline Winner */}
                <tr className="bg-amber-100/40 hover:bg-amber-100/60 transition-colors">
                  <td className="py-3.5 px-4 font-sans font-extrabold text-slate-900 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-[#b45309]" />
                    Mandatory Baseline
                    <span className="pill-badge track text-[9px] py-0.5 px-1.5 ml-1">Winner</span>
                  </td>
                  <td className="py-3.5 px-4">4 (RGB + NIR)</td>
                  <td className="py-3.5 px-4 font-sans">BCEWithLogitsLoss</td>
                  <td className="py-3.5 px-4">0.6924</td>
                  <td className="py-3.5 px-4">0.6100</td>
                  <td className="py-3.5 px-4 font-bold text-[#b45309] text-sm">0.7834</td>
                  <td className="py-3.5 px-4 font-bold text-[#047857] text-sm">0.7044</td>
                  <td className="py-3.5 px-4">0.3781</td>
                </tr>

                {/* Ablation Run 1 */}
                <tr className="hover:bg-amber-50/40 transition-colors">
                  <td className="py-3.5 px-4 font-sans font-bold text-slate-800">Ablation Run 1 (3-Band)</td>
                  <td className="py-3.5 px-4">3 (RGB-Only)</td>
                  <td className="py-3.5 px-4 font-sans">Focal-Tversky (&alpha;=0.7, &beta;=0.3, &gamma;=1.333)</td>
                  <td className="py-3.5 px-4">0.4414</td>
                  <td className="py-3.5 px-4">0.3653</td>
                  <td className="py-3.5 px-4">0.5027</td>
                  <td className="py-3.5 px-4">0.4245</td>
                  <td className="py-3.5 px-4">0.5154</td>
                </tr>

                {/* Ablation Run 2 */}
                <tr className="hover:bg-amber-50/40 transition-colors">
                  <td className="py-3.5 px-4 font-sans font-bold text-slate-800">Ablation Run 2 (4-Band)</td>
                  <td className="py-3.5 px-4">4 (RGB + NIR)</td>
                  <td className="py-3.5 px-4 font-sans">Focal-Tversky (&alpha;=0.7, &beta;=0.3, &gamma;=1.333)</td>
                  <td className="py-3.5 px-4">0.4840</td>
                  <td className="py-3.5 px-4">0.4034</td>
                  <td className="py-3.5 px-4">0.5796</td>
                  <td className="py-3.5 px-4">0.5013</td>
                  <td className="py-3.5 px-4">0.5102</td>
                </tr>

                {/* Empirical Physical Delta */}
                <tr className="bg-emerald-50 font-bold text-emerald-800">
                  <td className="py-3.5 px-4 font-sans">NIR Band Contribution (&Delta;)</td>
                  <td className="py-3.5 px-4">+1 Band (NIR)</td>
                  <td className="py-3.5 px-4 font-sans">Empirical Physical Delta</td>
                  <td className="py-3.5 px-4">+0.0426</td>
                  <td className="py-3.5 px-4">+0.0381</td>
                  <td className="py-3.5 px-4 text-emerald-800 text-sm">+0.0769 (+15.3%)</td>
                  <td className="py-3.5 px-4 text-emerald-800 text-sm">+0.0768 (+18.1%)</td>
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
