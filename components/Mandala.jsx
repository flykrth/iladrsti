"use client";

import React from "react";

/**
 * Pure Mathematical Inline SVG Indian Mandala Components
 * Intricate sacred geometry: 16-point yantra star, 8-petal classical lotus,
 * 32 sunburst rays, concentric bead rings, sacred shatkona triangles, and golden bindu.
 * Zero external assets rule compliant.
 */

export function MandalaSvg({
  size = 400,
  className = "",
  primaryColor = "#d4af37", // Temple Gold
  accentColor = "#ff8800",  // Indian Saffron
  cyanColor = "#00f3ff",    // High-tech Neon Cyan
  terracottaColor = "#e04b2b", // Terracotta Red
}) {
  return (
    <svg
      viewBox="0 0 400 400"
      width={size}
      height={size}
      className={className}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id="mandalaCenterGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={primaryColor} stopOpacity="0.4" />
          <stop offset="70%" stopColor={accentColor} stopOpacity="0.1" />
          <stop offset="100%" stopColor="transparent" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Subtle Central Radial Glow */}
      <circle cx="200" cy="200" r="180" fill="url(#mandalaCenterGlow)" />

      {/* Concentric Outer Rings */}
      <circle cx="200" cy="200" r="192" stroke={primaryColor} strokeWidth="1.2" strokeOpacity="0.75" />
      <circle cx="200" cy="200" r="185" stroke={cyanColor} strokeWidth="1" strokeDasharray="3 3" strokeOpacity="0.8" />
      <circle cx="200" cy="200" r="172" stroke={accentColor} strokeWidth="1.5" strokeOpacity="0.8" />

      {/* 32 Sunburst Perimeter Rays */}
      <g stroke={primaryColor} strokeWidth="1.2" strokeOpacity="0.6">
        <line x1="200" y1="8" x2="200" y2="28" />
        <line x1="200" y1="372" x2="200" y2="392" />
        <line x1="8" y1="200" x2="28" y2="200" />
        <line x1="372" y1="200" x2="392" y2="200" />
        <line x1="64" y1="64" x2="78" y2="78" />
        <line x1="322" y1="322" x2="336" y2="336" />
        <line x1="64" y1="336" x2="78" y2="322" />
        <line x1="322" y1="78" x2="336" y2="64" />
        <line x1="200" y1="18" x2="200" y2="28" transform="rotate(22.5 200 200)" />
        <line x1="200" y1="372" x2="200" y2="382" transform="rotate(22.5 200 200)" />
        <line x1="18" y1="200" x2="28" y2="200" transform="rotate(22.5 200 200)" />
        <line x1="372" y1="200" x2="382" y2="200" transform="rotate(22.5 200 200)" />
        <line x1="200" y1="18" x2="200" y2="28" transform="rotate(67.5 200 200)" />
        <line x1="200" y1="372" x2="200" y2="382" transform="rotate(67.5 200 200)" />
        <line x1="18" y1="200" x2="28" y2="200" transform="rotate(67.5 200 200)" />
        <line x1="372" y1="200" x2="382" y2="200" transform="rotate(67.5 200 200)" />
        <line x1="200" y1="18" x2="200" y2="28" transform="rotate(112.5 200 200)" />
        <line x1="200" y1="372" x2="200" y2="382" transform="rotate(112.5 200 200)" />
        <line x1="18" y1="200" x2="28" y2="200" transform="rotate(112.5 200 200)" />
        <line x1="372" y1="200" x2="382" y2="200" transform="rotate(112.5 200 200)" />
        <line x1="200" y1="18" x2="200" y2="28" transform="rotate(157.5 200 200)" />
        <line x1="200" y1="372" x2="200" y2="382" transform="rotate(157.5 200 200)" />
        <line x1="18" y1="200" x2="28" y2="200" transform="rotate(157.5 200 200)" />
        <line x1="372" y1="200" x2="382" y2="200" transform="rotate(157.5 200 200)" />
      </g>

      {/* 16-Point Geometric Star / Yantra Outer Layer */}
      <polygon
        points="200,35 215,85 265,55 245,105 305,95 270,140 335,150 285,185 345,200 285,215 335,250 270,260 305,305 245,295 265,345 215,315 200,365 185,315 135,345 155,295 95,305 130,260 65,250 115,215 55,200 115,185 65,150 130,140 95,95 155,105 135,55 185,85"
        stroke={accentColor}
        strokeWidth="1.2"
        strokeOpacity="0.75"
      />

      {/* Concentric Bead Ring */}
      <circle cx="200" cy="200" r="140" stroke={primaryColor} strokeWidth="1" strokeOpacity="0.6" />
      {[...Array(16)].map((_, i) => {
        const angle = (i * 360) / 16;
        const rad = (angle * Math.PI) / 180;
        const cx = 200 + 140 * Math.cos(rad);
        const cy = 200 + 140 * Math.sin(rad);
        return (
          <circle key={`bead-${i}`} cx={cx} cy={cy} r="2.5" fill={primaryColor} opacity="0.8" />
        );
      })}

      {/* 8-Petal Classical Indian Lotus */}
      <g stroke={terracottaColor} strokeWidth="1.5" strokeOpacity="0.85">
        <path d="M200,90 Q225,145 200,165 Q175,145 200,90 Z" />
        <path d="M200,310 Q225,255 200,235 Q175,255 200,310 Z" />
        <path d="M90,200 Q145,225 165,200 Q145,175 90,200 Z" />
        <path d="M310,200 Q255,225 235,200 Q255,175 310,200 Z" />
        <path d="M122,122 Q175,150 175,175 Q150,175 122,122 Z" />
        <path d="M278,278 Q225,250 225,225 Q250,225 278,278 Z" />
        <path d="M122,278 Q175,250 175,225 Q150,225 122,278 Z" />
        <path d="M278,122 Q225,150 225,175 Q250,175 278,122 Z" />
      </g>

      {/* Sacred Shatkona (Interlocking Yantra Triangles) */}
      <polygon
        points="200,115 273,242 127,242"
        stroke={cyanColor}
        strokeWidth="1.2"
        strokeOpacity="0.85"
      />
      <polygon
        points="200,285 273,158 127,158"
        stroke={cyanColor}
        strokeWidth="1.2"
        strokeOpacity="0.85"
      />

      {/* Inner Lotus Core & Golden Bindu */}
      <circle cx="200" cy="200" r="50" stroke={primaryColor} strokeWidth="1.5" strokeOpacity="0.9" />
      <circle cx="200" cy="200" r="28" stroke={accentColor} strokeWidth="1" strokeDasharray="3 3" />
      <circle cx="200" cy="200" r="14" fill={primaryColor} fillOpacity="0.3" stroke={primaryColor} strokeWidth="1.5" />
      <circle cx="200" cy="200" r="4" fill={cyanColor} />
    </svg>
  );
}

/**
 * MandalaWatermark: Low-opacity background watermark for cards and sections
 */
export function MandalaWatermark({
  size = 320,
  position = "center", // 'center' | 'top-right' | 'bottom-left' | 'top-left' | 'bottom-right'
  opacity = 0.08,
  rotate = true,
  className = "",
}) {
  const positionClasses = {
    center: "top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2",
    "top-right": "-top-16 -right-16",
    "bottom-left": "-bottom-16 -left-16",
    "top-left": "-top-16 -left-16",
    "bottom-right": "-bottom-16 -right-16",
  };

  return (
    <div
      className={`absolute pointer-events-none select-none z-0 ${positionClasses[position] || ""} ${className}`}
      style={{ opacity }}
    >
      <div className={rotate ? "animate-spin-very-slow" : ""}>
        <MandalaSvg size={size} />
      </div>
    </div>
  );
}

/**
 * MandalaCorner: Corner brackets with sacred geometry filigree
 */
export function MandalaCorner({ position = "tl", className = "" }) {
  const cornerStyles = {
    tl: "top-0 left-0 border-t-2 border-l-2 border-[#d4af37] rounded-tl-sm",
    tr: "top-0 right-0 border-t-2 border-r-2 border-[#00f3ff] rounded-tr-sm",
    bl: "bottom-0 left-0 border-b-2 border-l-2 border-[#00f3ff] rounded-bl-sm",
    br: "bottom-0 right-0 border-b-2 border-r-2 border-[#ff8800] rounded-br-sm",
  };

  return (
    <div
      className={`absolute w-5 h-5 pointer-events-none z-10 transition-opacity ${cornerStyles[position] || ""} ${className}`}
    >
      {/* Decorative inner corner tick */}
      <span className="absolute w-1.5 h-1.5 bg-white/40 top-0.5 left-0.5" />
    </div>
  );
}

/**
 * MandalaBadge: Rotating header emblem for Brand & Section Badges
 */
export function MandalaBadge({ size = 36, className = "" }) {
  return (
    <div
      className={`relative rounded-xl bg-gradient-to-tr from-[#161226] to-[#251838] border border-[#d4af37]/40 flex items-center justify-center shadow-lg shadow-[#d4af37]/15 shrink-0 overflow-hidden ${className}`}
      style={{ width: size, height: size }}
    >
      <div className="animate-spin-very-slow w-full h-full flex items-center justify-center p-0.5">
        <MandalaSvg size={size - 2} />
      </div>
    </div>
  );
}

export default MandalaSvg;
