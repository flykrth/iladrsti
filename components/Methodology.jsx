"use client";

import { useMemo } from "react";
import katex from "katex";
import { Layers, Network, Zap, BookOpen, AlertCircle } from "lucide-react";

export default function Methodology() {
  // Render KaTeX equations safely
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
        "FTL = (1 - TI)^\\gamma = \\left(1 - \\frac{TP + \\epsilon}{TP + \\alpha FN + \\beta FP + \\epsilon}\\right)^{1.333}",
        { displayMode: true, throwOnError: false }
      );
    } catch {
      return "FTL = (1 - TI)^γ";
    }
  }, []);

  const tiFormulaHtml = useMemo(() => {
    try {
      return katex.renderToString(
        "TI = \\frac{\\sum_{i} p_i y_i + \\epsilon}{\\sum_{i} p_i y_i + 0.7 \\sum_{i} (1 - p_i) y_i + 0.3 \\sum_{i} p_i (1 - y_i) + \\epsilon}",
        { displayMode: true, throwOnError: false }
      );
    } catch {
      return "TI = (TP + ε) / (TP + α·FN + β·FP + ε)";
    }
  }, []);

  return (
    <section id="methodology" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="text-center max-w-3xl mx-auto mb-14">
        <span className="pill-badge track mb-3">Methodology & Deep Learning Architecture</span>
        <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight mt-2 mb-4">
          ResNet-50 U-Net & Focal-Tversky Loss
        </h2>
        <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
          Resolving severe spatial class imbalance and smoke occlusion through multispectral weight transfer and gradient-focusing loss formulations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Symmetrical 5-Stage U-Net Architecture */}
        <div className="lg:col-span-7 glass-card p-6 sm:p-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-brand-pink/20 text-brand-pink border border-brand-pink/30">
              <Network className="w-6 h-6" />
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Symmetrical 5-Stage U-Net Architecture
            </h3>
          </div>

          <p className="text-slate-300 text-sm leading-relaxed mb-6">
            Standard optical segmentation models fail when wildfires occupy less than{" "}
            <strong className="text-white font-semibold">5%</strong> of a satellite tile. Ilādṛṣṭi couples a deep{" "}
            <strong className="text-white font-semibold">ResNet-50 encoder</strong> with a customized 5-stage expansive decoder to retain fine perimeter details.
          </p>

          <div className="space-y-4 mb-6">
            {/* Step 1 */}
            <div className="flex items-start gap-4 p-4 rounded-xl bg-white/[0.04] border border-white/10 hover:bg-white/[0.08] transition-colors">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-pink/30 to-brand-purple/30 border border-brand-pink/50 text-brand-pink font-extrabold text-sm flex items-center justify-center shrink-0">
                1
              </div>
              <div>
                <h4 className="text-sm sm:text-base font-bold text-white mb-1">
                  4-Band Initial Convolution Adaptation
                </h4>
                <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                  Adapts ImageNet-pretrained weights to 4-channel input (Red, Green, Blue, NIR). The 4th NIR channel kernel is initialized via channel-wise mean averaging, preserving feature transfer without cold-start penalties.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex items-start gap-4 p-4 rounded-xl bg-white/[0.04] border border-white/10 hover:bg-white/[0.08] transition-colors">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-pink/30 to-brand-purple/30 border border-brand-pink/50 text-brand-pink font-extrabold text-sm flex items-center justify-center shrink-0">
                2
              </div>
              <div>
                <h4 className="text-sm sm:text-base font-bold text-white mb-1">
                  Multi-Scale Residual Skip Connections
                </h4>
                <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                  High-resolution feature maps from ResNet stages (C1, C2, C3, C4) are routed directly into the decoder stages, restoring thin fire breaks and boundary perimeters lost during downsampling.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex items-start gap-4 p-4 rounded-xl bg-white/[0.04] border border-white/10 hover:bg-white/[0.08] transition-colors">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-pink/30 to-brand-purple/30 border border-brand-pink/50 text-brand-pink font-extrabold text-sm flex items-center justify-center shrink-0">
                3
              </div>
              <div>
                <h4 className="text-sm sm:text-base font-bold text-white mb-1">
                  Expansive Decoder with Double Convolutions
                </h4>
                <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                  Each decoder stage applies bilinear upsampling followed by double 3x3 convolutions, Batch Normalization, and ReLU activations, terminating in a 1x1 logit projection head.
                </p>
              </div>
            </div>
          </div>

          {/* 4-Band Conv1 Adaptation Formula Callout */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-brand-pink/30 shadow-inner">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-brand-pink mb-2">
              <Zap className="w-4 h-4" /> Multispectral Weight Transfer Formula:
            </div>
            <div
              className="text-xs sm:text-sm overflow-x-auto text-slate-100 py-1"
              dangerouslySetInnerHTML={{ __html: conv1FormulaHtml }}
            />
          </div>
        </div>

        {/* Right Column: Dedicated Focal-Tversky Loss Mathematics Block */}
        <div className="lg:col-span-5 glass-panel p-6 sm:p-8 bg-gradient-to-b from-[#140e26]/80 to-[#231034]/70 border border-brand-pink/30 shadow-2xl relative">
          <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/15">
            <div className="flex items-center gap-2.5">
              <BookOpen className="w-5 h-5 text-brand-pink" />
              <h3 className="text-lg sm:text-xl font-extrabold text-white tracking-tight">
                Focal-Tversky Loss
              </h3>
            </div>
            <span className="pill-badge track">Phase 2 Objective</span>
          </div>

          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed mb-6">
            Standard Binary Cross-Entropy (BCE) treats every pixel equally, allowing overwhelming background pixels (95%) to suppress fire gradient updates. The <strong className="text-white font-semibold">Focal-Tversky Loss (FTL)</strong> penalizes False Negatives and concentrates learning on ambiguous boundary pixels.
          </p>

          {/* Primary Formula Box */}
          <div className="bg-black/60 border border-white/15 rounded-xl p-4 mb-4 text-center shadow-inner">
            <div
              className="overflow-x-auto text-sm sm:text-base text-white py-1"
              dangerouslySetInnerHTML={{ __html: ftlFormulaHtml }}
            />
            <div className="text-[11px] font-semibold text-slate-400 mt-1 uppercase tracking-wider">
              Non-Linear Boundary Focusing Objective
            </div>
          </div>

          {/* Tversky Index Formula Box */}
          <div className="bg-black/60 border border-white/15 rounded-xl p-4 mb-6 text-center shadow-inner">
            <div
              className="overflow-x-auto text-xs sm:text-sm text-white py-1"
              dangerouslySetInnerHTML={{ __html: tiFormulaHtml }}
            />
            <div className="text-[11px] font-semibold text-slate-400 mt-1 uppercase tracking-wider">
              Asymmetric Tversky Index Formulation
            </div>
          </div>

          {/* Parameter Breakdown Grid */}
          <div className="grid grid-cols-3 gap-2.5 mb-6">
            <div className="p-3 rounded-lg bg-white/[0.05] border border-white/10 text-center">
              <div className="text-sm font-black text-brand-pink font-mono">&alpha; = 0.7</div>
              <div className="text-[10px] text-slate-400 mt-1 leading-tight">
                Heavy penalty on False Negatives
              </div>
            </div>
            <div className="p-3 rounded-lg bg-white/[0.05] border border-white/10 text-center">
              <div className="text-sm font-black text-brand-purple font-mono">&beta; = 0.3</div>
              <div className="text-[10px] text-slate-400 mt-1 leading-tight">
                Moderate penalty on False Positives
              </div>
            </div>
            <div className="p-3 rounded-lg bg-white/[0.05] border border-white/10 text-center">
              <div className="text-sm font-black text-brand-cyan font-mono">&gamma; = 1.333</div>
              <div className="text-[10px] text-slate-400 mt-1 leading-tight">
                Focuses gradient on hard borders
              </div>
            </div>
          </div>

          {/* Theoretical Guarantee Callout */}
          <div className="p-3.5 rounded-xl bg-brand-pink/10 border border-brand-pink/25 flex items-start gap-2.5">
            <AlertCircle className="w-5 h-5 text-brand-pink shrink-0 mt-0.5" />
            <p className="text-xs text-slate-200 leading-relaxed">
              <strong className="text-white font-semibold">Theoretical Guarantee:</strong> When &gamma; &gt; 1, easy confident predictions (TI &rarr; 1) generate near-zero loss, preventing background saturation and driving backpropagation updates into elusive firebreak margins.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
