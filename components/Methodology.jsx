"use client";

import { useMemo } from "react";
import katex from "katex";
import { Layers, Network, Zap, BookOpen, AlertCircle } from "lucide-react";
import { MandalaWatermark, MandalaCorner } from "./Mandala";

export default function Methodology() {
  // Render KaTeX equations safely and compactly for light theme
  const conv1FormulaHtml = useMemo(() => {
    try {
      return katex.renderToString(
        "\\mathbf{W}_{\\text{conv1}}[:, 3:4, :, :] \\leftarrow \\frac{1}{3} \\sum_{c=0}^{2} \\mathbf{W}_{\\text{ImageNet}}[:, c:c+1, :, :]",
        { displayMode: true, throwOnError: false }
      );
    } catch {
      return "W_conv1[:, 3:4] <- 1/3 sum(W_ImageNet[:, 0:3])";
    }
  }, []);

  const ftlFormulaHtml = useMemo(() => {
    try {
      return katex.renderToString(
        "\\mathrm{FTL} = (1 - \\mathrm{TI})^\\gamma = (1 - \\mathrm{TI})^{1.333}",
        { displayMode: true, throwOnError: false }
      );
    } catch {
      return "FTL = (1 - TI)^γ";
    }
  }, []);

  const tiFormulaHtml = useMemo(() => {
    try {
      return katex.renderToString(
        "\\mathrm{TI} = \\frac{\\mathrm{TP} + \\epsilon}{\\mathrm{TP} + 0.7\\,\\mathrm{FN} + 0.3\\,\\mathrm{FP} + \\epsilon}",
        { displayMode: true, throwOnError: false }
      );
    } catch {
      return "TI = (TP + ε) / (TP + 0.7·FN + 0.3·FP + ε)";
    }
  }, []);

  return (
    <section id="methodology" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto relative">
      {/* Ambient Section Background Mandala Watermark in Temple Gold */}
      <MandalaWatermark position="center" size={700} opacity={0.12} />

      {/* Section Header */}
      <div className="text-center max-w-3xl mx-auto mb-14 relative z-10">
        <span className="pill-badge track mb-3">Methodology &amp; Deep Learning Architecture</span>
        <h2 className="text-3xl sm:text-4xl font-black text-slate-900 tracking-tight mt-2 mb-4">
          ResNet-50 U-Net &amp; Focal-Tversky Loss
        </h2>
        <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
          Resolving severe spatial class imbalance and smoke occlusion through multispectral weight transfer and gradient-focusing loss formulations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start relative z-10">
        {/* Left Column: Symmetrical 5-Stage U-Net Architecture (White & Gold) */}
        <div className="lg:col-span-7 glass-card p-6 sm:p-8 relative overflow-hidden bg-white/92 border border-[#d4af37]/35 shadow-xl">
          {/* Indian Mandala Cultural Elements */}
          <MandalaCorner position="tl" />
          <MandalaCorner position="br" />
          <MandalaWatermark position="bottom-right" size={320} opacity={0.14} />

          <div className="flex items-center gap-3 mb-4 relative z-10">
            <div className="p-2 rounded-lg bg-amber-50 text-[#b45309] border border-[#d4af37]/40 shadow-xs">
              <Network className="w-6 h-6" />
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              Symmetrical 5-Stage U-Net Architecture
            </h3>
          </div>

          <p className="text-slate-600 text-sm leading-relaxed mb-6 relative z-10">
            Standard optical segmentation models fail when wildfires occupy less than{" "}
            <strong className="text-slate-900 font-semibold">5%</strong> of a satellite tile. Ilādṛṣṭi couples a deep{" "}
            <strong className="text-slate-900 font-semibold">ResNet-50 encoder</strong> with a customized 5-stage expansive decoder to retain fine perimeter details.
          </p>

          <div className="space-y-4 mb-6 relative z-10">
            {/* Step 1 */}
            <div className="flex items-start gap-4 p-4 rounded-xl bg-amber-50/40 border border-[#d4af37]/25 hover:bg-amber-50/70 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#d4af37]/30 to-[#fef3c7] border border-[#d4af37]/60 text-[#854d0e] font-extrabold text-sm flex items-center justify-center shrink-0 shadow-xs">
                1
              </div>
              <div>
                <h4 className="text-sm sm:text-base font-bold text-slate-900 mb-1">
                  4-Band Initial Convolution Adaptation
                </h4>
                <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                  Adapts ImageNet-pretrained weights to 4-channel input (Red, Green, Blue, NIR). The 4th NIR channel kernel is initialized via channel-wise mean averaging, preserving feature transfer without cold-start penalties.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex items-start gap-4 p-4 rounded-xl bg-amber-50/40 border border-[#d4af37]/25 hover:bg-amber-50/70 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#d4af37]/30 to-[#fef3c7] border border-[#d4af37]/60 text-[#854d0e] font-extrabold text-sm flex items-center justify-center shrink-0 shadow-xs">
                2
              </div>
              <div>
                <h4 className="text-sm sm:text-base font-bold text-slate-900 mb-1">
                  Multi-Scale Residual Skip Connections
                </h4>
                <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                  High-resolution feature maps from ResNet stages (C1, C2, C3, C4) are routed directly into the decoder stages, restoring thin fire breaks and boundary perimeters lost during downsampling.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex items-start gap-4 p-4 rounded-xl bg-amber-50/40 border border-[#d4af37]/25 hover:bg-amber-50/70 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#d4af37]/30 to-[#fef3c7] border border-[#d4af37]/60 text-[#854d0e] font-extrabold text-sm flex items-center justify-center shrink-0 shadow-xs">
                3
              </div>
              <div>
                <h4 className="text-sm sm:text-base font-bold text-slate-900 mb-1">
                  Expansive Decoder with Double Convolutions
                </h4>
                <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                  Each decoder stage applies bilinear upsampling followed by double 3x3 convolutions, Batch Normalization, and ReLU activations, terminating in a 1x1 logit projection head.
                </p>
              </div>
            </div>
          </div>

          {/* 4-Band Conv1 Adaptation Formula Callout */}
          <div className="p-4 rounded-xl bg-amber-50/60 border border-[#d4af37]/35 shadow-sm relative z-10">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#854d0e] mb-2">
              <Zap className="w-4 h-4 text-[#b45309]" /> Multispectral Weight Transfer Formula:
            </div>
            <div
              className="text-xs sm:text-sm overflow-x-auto no-scrollbar text-slate-900 py-1"
              dangerouslySetInnerHTML={{ __html: conv1FormulaHtml }}
            />
          </div>
        </div>

        {/* Right Column: Dedicated Focal-Tversky Loss Mathematics Block (White & Gold) */}
        <div className="lg:col-span-5 glass-panel p-6 sm:p-8 bg-gradient-to-b from-[#ffffff] to-[#fefcf8] border border-[#d4af37]/45 shadow-xl relative overflow-hidden">
          {/* Indian Mandala Cultural Elements */}
          <MandalaCorner position="tr" />
          <MandalaCorner position="bl" />
          <MandalaWatermark position="top-right" size={300} opacity={0.15} />

          <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#d4af37]/25 relative z-10">
            <div className="flex items-center gap-2.5">
              <BookOpen className="w-5 h-5 text-[#b45309]" />
              <h3 className="text-lg sm:text-xl font-extrabold text-slate-900 tracking-tight">
                Focal-Tversky Loss
              </h3>
            </div>
            <span className="pill-badge track">Phase 2 Objective</span>
          </div>

          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed mb-6 relative z-10">
            Standard Binary Cross-Entropy (BCE) treats every pixel equally, allowing overwhelming background pixels (95%) to suppress fire gradient updates. The <strong className="text-slate-900 font-semibold">Focal-Tversky Loss (FTL)</strong> penalizes False Negatives and concentrates learning on ambiguous boundary pixels.
          </p>

          {/* Primary Formula Box - Clean, Compact, Zero Scrollbars */}
          <div className="bg-amber-50/60 border border-[#d4af37]/35 rounded-xl p-4 mb-4 text-center shadow-xs relative z-10">
            <div
              className="overflow-x-auto no-scrollbar text-sm sm:text-base text-slate-900 py-1 flex justify-center items-center"
              dangerouslySetInnerHTML={{ __html: ftlFormulaHtml }}
            />
            <div className="text-[11px] font-semibold text-[#854d0e] mt-1 uppercase tracking-wider">
              Non-Linear Boundary Focusing Objective
            </div>
          </div>

          {/* Tversky Index Formula Box - Clean, Compact, Zero Scrollbars */}
          <div className="bg-amber-50/60 border border-[#d4af37]/35 rounded-xl p-4 mb-6 text-center shadow-xs relative z-10">
            <div
              className="overflow-x-auto no-scrollbar text-xs sm:text-sm text-slate-900 py-1 flex justify-center items-center"
              dangerouslySetInnerHTML={{ __html: tiFormulaHtml }}
            />
            <div className="text-[11px] font-semibold text-[#854d0e] mt-1 uppercase tracking-wider">
              Asymmetric Tversky Index Formulation
            </div>
          </div>

          {/* Parameter Breakdown Grid */}
          <div className="grid grid-cols-3 gap-2.5 mb-6 relative z-10">
            <div className="p-3 rounded-lg bg-white border border-[#d4af37]/30 text-center shadow-xs">
              <div className="text-sm font-black text-[#b45309] font-mono">&alpha; = 0.7</div>
              <div className="text-[10px] text-slate-600 mt-1 leading-tight">
                Heavy penalty on False Negatives
              </div>
            </div>
            <div className="p-3 rounded-lg bg-white border border-[#d4af37]/30 text-center shadow-xs">
              <div className="text-sm font-black text-[#c59b27] font-mono">&beta; = 0.3</div>
              <div className="text-[10px] text-slate-600 mt-1 leading-tight">
                Moderate penalty on False Positives
              </div>
            </div>
            <div className="p-3 rounded-lg bg-white border border-[#d4af37]/30 text-center shadow-xs">
              <div className="text-sm font-black text-[#854d0e] font-mono">&gamma; = 1.333</div>
              <div className="text-[10px] text-slate-600 mt-1 leading-tight">
                Focuses gradient on hard borders
              </div>
            </div>
          </div>

          {/* Theoretical Guarantee Callout */}
          <div className="p-3.5 rounded-xl bg-amber-50/80 border border-[#d4af37]/40 flex items-start gap-2.5 relative z-10 shadow-xs">
            <AlertCircle className="w-5 h-5 text-[#b45309] shrink-0 mt-0.5" />
            <p className="text-xs text-slate-700 leading-relaxed">
              <strong className="text-slate-900 font-semibold">Theoretical Guarantee:</strong> When &gamma; &gt; 1, easy confident predictions (TI &rarr; 1) generate near-zero loss, preventing background saturation and driving backpropagation updates into elusive firebreak margins.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
