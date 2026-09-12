"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Image from "next/image";
import { Play, Pause, ChevronLeft, ChevronRight, Eye, ShieldAlert, Mountain, CloudRain, Flame, Activity } from "lucide-react";
import metricsData from "../public/data/metrics.json";

export default function ImageSlider() {
  const scenes = metricsData.scenes;
  const [selectedSceneIndex, setSelectedSceneIndex] = useState(0);
  const [mode, setMode] = useState("rgb_vs_pred"); // 'rgb_vs_pred' | 'nir_vs_pred' | 'pred_vs_gt' | 'rgb_vs_confusion'
  const [position, setPosition] = useState(50.0);
  const [isDragging, setIsDragging] = useState(false);
  const [isScanning, setIsScanning] = useState(false);

  const containerRef = useRef(null);
  const scanDirectionRef = useRef(1);
  const scanAnimFrameRef = useRef(null);

  const currentScene = scenes[selectedSceneIndex];

  // Derive left and right images and labels based on active mode
  let leftSrc = currentScene.assets.rgb;
  let rightSrc = currentScene.assets.overlay;
  let leftLabel = "Raw Sentinel-2 RGB";
  let rightLabel = "Predicted Wildfire Mask Overlay";
  let showConfusionLegend = false;

  switch (mode) {
    case "rgb_vs_pred":
      leftSrc = currentScene.assets.rgb;
      rightSrc = currentScene.assets.overlay;
      leftLabel = "Raw Sentinel-2 RGB";
      rightLabel = "Predicted Wildfire Mask Overlay";
      break;
    case "nir_vs_pred":
      leftSrc = currentScene.assets.nir;
      rightSrc = currentScene.assets.overlay;
      leftLabel = "False-Color Infrared (NIR-R-G)";
      rightLabel = "Predicted Wildfire Mask Overlay";
      break;
    case "pred_vs_gt":
      leftSrc = currentScene.assets.pred;
      rightSrc = currentScene.assets.gt;
      leftLabel = "Model Prediction (Focal-Tversky)";
      rightLabel = "Ground Truth Scar (Reference)";
      break;
    case "rgb_vs_confusion":
      leftSrc = currentScene.assets.rgb;
      rightSrc = currentScene.assets.confusion;
      leftLabel = "Raw Satellite RGB";
      rightLabel = "Spatial Confusion Map (TP / FP / FN)";
      showConfusionLegend = true;
      break;
    default:
      break;
  }

  // Pointer & Drag handling
  const updatePosition = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const newPos = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.max(0, Math.min(100, newPos)));
  }, []);

  const handlePointerDown = (e) => {
    setIsDragging(true);
    if (isScanning) {
      setIsScanning(false);
    }
    updatePosition(e.clientX);
  };

  useEffect(() => {
    const handlePointerMove = (e) => {
      if (!isDragging) return;
      updatePosition(e.clientX);
    };

    const handlePointerUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", handlePointerUp);
      window.addEventListener("pointercancel", handlePointerUp);
    }

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [isDragging, updatePosition]);

  // Keyboard navigation
  const handleKeyDown = (e) => {
    if (e.key === "ArrowLeft") {
      setPosition((p) => Math.max(0, p - 5));
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      setPosition((p) => Math.min(100, p + 5));
      e.preventDefault();
    } else if (e.key === "Home") {
      setPosition(0);
      e.preventDefault();
    } else if (e.key === "End") {
      setPosition(100);
      e.preventDefault();
    }
  };

  // Auto-scan animation
  useEffect(() => {
    if (!isScanning) {
      if (scanAnimFrameRef.current) {
        cancelAnimationFrame(scanAnimFrameRef.current);
        scanAnimFrameRef.current = null;
      }
      return;
    }

    const scanSpeed = 0.35;
    const animate = () => {
      setPosition((prev) => {
        let next = prev + scanDirectionRef.current * scanSpeed;
        if (next >= 92) {
          next = 92;
          scanDirectionRef.current = -1;
        } else if (next <= 8) {
          next = 8;
          scanDirectionRef.current = 1;
        }
        return next;
      });
      scanAnimFrameRef.current = requestAnimationFrame(animate);
    };

    scanAnimFrameRef.current = requestAnimationFrame(animate);
    return () => {
      if (scanAnimFrameRef.current) {
        cancelAnimationFrame(scanAnimFrameRef.current);
      }
    };
  }, [isScanning]);

  const sceneIcons = [
    Flame,
    Activity,
    Mountain,
    Eye,
    CloudRain,
    ShieldAlert,
  ];

  return (
    <section id="interactive-demo" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="text-center max-w-3xl mx-auto mb-10">
        <span className="pill-badge track mb-3">Interactive Demonstration</span>
        <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight mt-2 mb-4">
          Multispectral Inference & Comparison Slider
        </h2>
        <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
          Drag the interactive divider to compare raw Sentinel-2 satellite inputs directly with Ilādṛṣṭi&apos;s predicted wildfire perimeter masks.
        </p>
      </div>

      <div className="glass-panel p-4 sm:p-6 lg:p-8 relative overflow-hidden shadow-2xl">
        {/* Controls Toolbar */}
        <div className="flex flex-col xl:flex-row items-stretch xl:items-center justify-between gap-4 mb-6">
          {/* Scene Selector */}
          <div className="flex items-center gap-2 overflow-x-auto pb-2 xl:pb-0 scrollbar-none">
            {scenes.map((scene, idx) => {
              const Icon = sceneIcons[idx] || Flame;
              const isSelected = selectedSceneIndex === idx;
              const isFailure = scene.category === "failure";

              return (
                <button
                  key={scene.id}
                  onClick={() => setSelectedSceneIndex(idx)}
                  className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap shrink-0 ${
                    isSelected
                      ? isFailure
                        ? "bg-red-500/25 border border-red-500/60 text-red-300 shadow-lg shadow-red-500/20"
                        : "bg-brand-pink/25 border border-brand-pink/60 text-white shadow-lg shadow-brand-pink/20"
                      : "bg-white/[0.06] border border-white/15 text-slate-300 hover:bg-white/[0.12] hover:text-white"
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isSelected ? (isFailure ? "text-red-400" : "text-brand-pink") : "text-slate-400"}`} />
                  {scene.name.replace("Failure Case: ", "Failure: ")}
                </button>
              );
            })}
          </div>

          {/* Layer Mode & Auto Scan Actions */}
          <div className="flex items-center gap-2.5 flex-wrap justify-between xl:justify-end">
            <div className="flex items-center gap-1.5 bg-black/40 p-1 rounded-xl border border-white/15 flex-wrap">
              {[
                { id: "rgb_vs_pred", label: "RGB vs Prediction" },
                { id: "nir_vs_pred", label: "NIR (CIR) vs Pred" },
                { id: "pred_vs_gt", label: "Pred vs GT" },
                { id: "rgb_vs_confusion", label: "Spatial Confusion" },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    mode === m.id
                      ? "bg-white/20 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <button
              onClick={() => setIsScanning((prev) => !prev)}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 border shadow-sm ${
                isScanning
                  ? "bg-amber-500/25 border-amber-400/50 text-amber-300"
                  : "bg-white/[0.08] border-white/20 text-slate-200 hover:bg-white/[0.15] hover:text-white"
              }`}
            >
              {isScanning ? (
                <>
                  <Pause className="w-3.5 h-3.5 fill-current" />
                  Pause Scan
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Auto Scan
                </>
              )}
            </button>
          </div>
        </div>

        {/* Before / After Split Slider Stage */}
        <div className="relative w-full aspect-[4/3] sm:aspect-[16/10] md:aspect-[16/9] max-h-[640px] rounded-2xl overflow-hidden select-none border border-white/20 shadow-2xl bg-black">
          <div
            ref={containerRef}
            tabIndex={0}
            onPointerDown={handlePointerDown}
            onKeyDown={handleKeyDown}
            className="relative w-full h-full cursor-ew-resize focus:outline-none focus:ring-2 focus:ring-brand-pink/50"
          >
            {/* Left & Right Floating Layer Badges */}
            <div className="absolute top-4 left-4 z-20 px-3 py-1.5 rounded-lg bg-black/75 backdrop-blur-md border border-white/20 text-[11px] font-bold text-white shadow-lg pointer-events-none">
              {leftLabel}
            </div>
            <div className="absolute top-4 right-4 z-20 px-3 py-1.5 rounded-lg bg-black/75 backdrop-blur-md border border-white/20 text-[11px] font-bold text-white shadow-lg pointer-events-none">
              {rightLabel}
            </div>

            {/* Before Image (Base Layer - Left/Full) */}
            <div className="absolute inset-0 w-full h-full pointer-events-none">
              <Image
                src={leftSrc}
                alt={leftLabel}
                fill
                priority
                sizes="(max-width: 1280px) 100vw, 1280px"
                className="object-cover"
              />
            </div>

            {/* After Image (Top Layer - Clipped) */}
            <div
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{
                clipPath: `polygon(0 0, ${position}% 0, ${position}% 100%, 0 100%)`,
              }}
            >
              <Image
                src={rightSrc}
                alt={rightLabel}
                fill
                priority
                sizes="(max-width: 1280px) 100vw, 1280px"
                className="object-cover"
              />
            </div>

            {/* Divider Line & Glowing Handle */}
            <div
              className="absolute top-0 bottom-0 w-[2px] bg-white shadow-[0_0_12px_rgba(255,75,114,0.9)] z-30 pointer-events-none"
              style={{ left: `${position}%` }}
            >
              <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-[#0d0a1a] border-2 border-white shadow-xl flex items-center justify-center text-white">
                <ChevronLeft className="w-3.5 h-3.5 -mr-1" />
                <ChevronRight className="w-3.5 h-3.5 -ml-1" />
              </div>
            </div>

            {/* Spatial Confusion Map Legend Overlay */}
            {showConfusionLegend && (
              <div className="absolute bottom-4 left-4 z-20 p-3 rounded-xl bg-black/85 backdrop-blur-md border border-white/20 shadow-2xl flex flex-col gap-1.5 text-[11px] font-semibold text-slate-200 pointer-events-none">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-sm bg-[#10b981]" />
                  <span>True Positive (Hit)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-sm bg-[#ef4444]" />
                  <span>False Positive (Commission)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-sm bg-[#3b82f6]" />
                  <span>False Negative (Omission)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-sm bg-slate-700" />
                  <span>Background (True Negative)</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Real-Time Telemetry HUD Overlay Bar */}
        <div className="mt-4 p-4 rounded-xl bg-white/[0.05] border border-white/15 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h5 className="text-base font-bold text-white tracking-tight">
              {currentScene.name} <span className="text-xs text-slate-400 font-mono">({currentScene.tile})</span>
            </h5>
            <p className="text-xs text-slate-300 mt-0.5">
              {currentScene.location} &bull; {currentScene.description}
            </p>
          </div>

          <div className="flex items-center gap-3 sm:gap-6 shrink-0 flex-wrap">
            <div className="text-left">
              <div className="text-[10px] uppercase font-bold text-slate-400">Scene Dice</div>
              <div className="text-lg font-black text-brand-pink font-mono">
                {currentScene.metrics.dice.toFixed(4)}
              </div>
            </div>
            <div className="text-left">
              <div className="text-[10px] uppercase font-bold text-slate-400">Mean IoU</div>
              <div className="text-lg font-black text-brand-emerald font-mono">
                {currentScene.metrics.iou.toFixed(4)}
              </div>
            </div>
            <div className="text-left">
              <div className="text-[10px] uppercase font-bold text-slate-400">Precision</div>
              <div className="text-lg font-black text-slate-100 font-mono">
                {(currentScene.metrics.precision * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-left">
              <div className="text-[10px] uppercase font-bold text-slate-400">Recall</div>
              <div className="text-lg font-black text-slate-100 font-mono">
                {(currentScene.metrics.recall * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-left">
              <div className="text-[10px] uppercase font-bold text-slate-400">Burned Scar</div>
              <div className="text-lg font-black text-brand-pink font-mono">
                {currentScene.metrics.scarAreaPct.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
