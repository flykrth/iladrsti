"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Play, Code2, Sparkles, Activity, Cpu, Layers } from "lucide-react";
import { MandalaWatermark, MandalaCorner } from "./Mandala";

export default function Hero() {
  const kpis = [
    {
      label: "Test Dice Score (F1)",
      value: "0.7834",
      sub: "Held-out test split (N=40)",
      accentClass: "text-[#b45309]",
      icon: Activity,
    },
    {
      label: "Test Mean IoU",
      value: "0.7044",
      sub: "Strict spatial Jaccard index",
      accentClass: "text-[#047857]",
      icon: Sparkles,
    },
    {
      label: "NIR Spectral Gain",
      value: "+7.69%",
      sub: "+15.3% relative boost",
      accentClass: "text-[#0284c7]",
      icon: Layers,
    },
    {
      label: "Edge GPU Latency",
      value: "36.4 FPS",
      sub: "< 50s per full Sentinel tile",
      accentClass: "text-[#7c3aed]",
      icon: Cpu,
    },
    {
      label: "Model Precision",
      value: "ResNet-50",
      sub: "4-Band Transfer Adapted U-Net",
      accentClass: "text-[#b45309]",
      icon: Code2,
    },
  ];

  return (
    <section id="hero" className="pt-32 sm:pt-36 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto relative">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="glass-panel p-6 sm:p-10 lg:p-14 relative text-center overflow-hidden border border-[#d4af37]/35 shadow-xl bg-white/92"
      >
        {/* Sacred Indian Mandala Corner Flourishes & Background Halo in Royal Gold */}
        <MandalaCorner position="tl" />
        <MandalaCorner position="tr" />
        <MandalaCorner position="bl" />
        <MandalaCorner position="br" />
        <MandalaWatermark position="center" size={620} opacity={0.16} />

        {/* Subtle golden accent border line on top */}
        <div className="absolute top-0 left-1/4 right-1/4 h-[1.5px] bg-gradient-to-r from-transparent via-[#d4af37] to-transparent z-10" />

        {/* Badges row */}
        <div className="flex items-center justify-center gap-2 sm:gap-3 flex-wrap mb-6 relative z-10">
          <span className="pill-badge track">
            <span className="pulse-dot fire" /> Deep Learning Hackathon @ Amrita
          </span>
          <span className="pill-badge success">European Space Agency Sentinel-2 L2A</span>
          <span className="pill-badge seed">Deterministic Evaluation (N=40 Test)</span>
        </div>

        {/* Title in White & Gold Luxury Typography */}
        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight mb-4 leading-tight relative z-10">
          <span className="bg-gradient-to-r from-slate-950 via-[#854d0e] to-[#b45309] bg-clip-text text-transparent">
            Ilādṛṣṭi: Multispectral Wildfire
          </span>
          <br />
          <span className="text-slate-900">Semantic Segmentation</span>
        </h1>

        {/* Sanskrit Etymology */}
        <div className="text-sm sm:text-base font-semibold text-[#854d0e] mb-6 tracking-wide relative z-10">
          Sanskrit: <span className="font-bold text-[#b45309]">इला</span> (Earth) +{" "}
          <span className="font-bold text-[#b45309]">दृष्टि</span> (Vision) — Autonomous Earth Observation Intelligence
        </div>

        {/* Mission Statement */}
        <p className="max-w-3xl mx-auto text-base sm:text-lg text-slate-700 leading-relaxed mb-8 relative z-10">
          Engineered for high-precision autonomous wildfire perimeter delineation and burned area scar mapping.
          Combines <strong className="text-slate-900 font-semibold">4-Band Transfer Learning</strong> with{" "}
          <strong className="text-slate-900 font-semibold">Focal-Tversky Loss Optimization</strong> to penetrate dense smoke hazes,
          resolve fine perimeter boundaries, and accelerate tactical disaster response and aerial retardant drops.
        </p>

        {/* CTA Actions */}
        <div className="flex items-center justify-center gap-4 flex-wrap mb-12 relative z-10">
          <Link href="#interactive-demo" className="btn-primary">
            <Play className="w-4 h-4 fill-current" />
            Launch Interactive Visual Demo
          </Link>
          <Link href="#methodology" className="glass-button">
            <Code2 className="w-4 h-4 text-[#b45309]" />
            Explore Focal-Tversky Math
          </Link>
        </div>

        {/* Live KPI Telemetry Grid with Golden Highlights */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4 text-left relative z-10">
          {kpis.map((kpi) => {
            const Icon = kpi.icon;
            return (
              <motion.div
                key={kpi.label}
                whileHover={{ y: -4, scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="p-4 rounded-xl bg-amber-50/40 border border-[#d4af37]/35 hover:border-[#d4af37]/70 hover:bg-amber-50/70 transition-all shadow-sm"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600">
                    {kpi.label}
                  </span>
                  <Icon className="w-4 h-4 text-[#854d0e]" />
                </div>
                <div className={`text-2xl sm:text-3xl font-extrabold tracking-tight ${kpi.accentClass}`}>
                  {kpi.value}
                </div>
                <div className="text-[11px] text-slate-500 mt-1">{kpi.sub}</div>
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </section>
  );
}
