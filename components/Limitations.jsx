"use client";

import Image from "next/image";
import { AlertTriangle, CloudRain, Mountain, Wind } from "lucide-react";
import { MandalaWatermark, MandalaCorner } from "./Mandala";

export default function Limitations() {
  const failures = [
    {
      title: "1. Dense Clouds & Edge Diffraction",
      badge: "False Positive Commission",
      badgeColor: "bg-rose-50 text-rose-800 border-rose-300",
      icon: CloudRain,
      mechanism:
        "High-albedo cloud tops reflect >85% of visible and NIR solar radiation. At cloud edges, sharp dropoffs create extreme spatial gradients that convolutional filters occasionally mistake for fresh fire perimeters.",
      impact:
        "Fringe false alarm rings bordering thick stratocumulus banks (demonstrated in Scene a1__07_13 with 13,269 FP pixels).",
      mitigation:
        "Ingestion of Sentinel-2 Scene Classification Layer (SCL) to pre-mask cloud and cirrus pixels prior to inference.",
    },
    {
      title: "2. Deep Mountain Ravine Shadows",
      badge: "Ash Spectral Mimicry",
      badgeColor: "bg-purple-50 text-purple-800 border-purple-300",
      icon: Mountain,
      mechanism:
        "Steep north-facing alpine ravines receive minimal direct solar irradiance. Low reflectance values (<0.05 across RGB and NIR) deceptively mimic the flat, dark optical absorption profile of charcoal ash.",
      impact:
        "Localized false positive clusters in deep mountain gorges (demonstrated in Scene k__4_6 with 7,211 FP pixels).",
      mitigation:
        "Fusing Short-Wave Infrared (SWIR Band 12, 2190nm) and Digital Elevation Model (DEM) hillshading layers.",
    },
    {
      title: "3. Low-Biomass Grassland Scars",
      badge: "Transient Omission",
      badgeColor: "bg-amber-50 text-amber-800 border-amber-300",
      icon: Wind,
      mechanism:
        "Fast-moving savanna grass fires leave only ephemeral surface soot without deep canopy destruction. Wind rapidly scatters the light ash layer within 48 hours, causing spectral signatures to revert before satellite revisit.",
      impact:
        "Under-segmentation (False Negatives) for thin pasture fires compared to dense coniferous forest burns.",
      mitigation:
        "Bi-temporal Sentinel-2 pre/post difference mapping (dNBR) to detect transient vegetative biomass shifts.",
    },
  ];

  return (
    <section id="limitations" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="text-center max-w-3xl mx-auto mb-10">
        <span className="pill-badge track text-amber-800 border-[#d4af37]/40 bg-amber-500/10 mb-3 inline-flex">
          Scientific Integrity &amp; Edge Cases
        </span>
        <h2 className="text-3xl sm:text-4xl font-black text-slate-900 tracking-tight mt-2 mb-4">
          Limitations &amp; Optical Failure Analysis
        </h2>
        <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
          Honest accounting of physical optical ambiguities, false positive triggers under extreme atmospheric conditions, and proposed architectural mitigations.
        </p>
      </div>

      {/* Warning Callout Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-amber-50/95 via-orange-50/85 to-amber-100/75 border border-[#d4af37]/50 shadow-md flex items-start gap-4 mb-8 relative overflow-hidden">
        <MandalaCorner position="top-right" size={90} opacity={0.12} />
        <div className="p-3 rounded-xl bg-amber-500/15 text-amber-800 border border-[#d4af37]/50 shrink-0 relative z-10">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <div className="relative z-10">
          <h3 className="text-base sm:text-lg font-bold text-slate-900 mb-1">
            Transparent Negative Results Audit
          </h3>
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
            While Ilādṛṣṭi demonstrates high accuracy across standard vegetative wildfire regimes, optical satellite systems suffer from physical photon scattering ambiguities where non-fire phenomena exhibit spectral signatures resembling carbonized ash.
          </p>
        </div>
      </div>

      {/* Failure Cases Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        {failures.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.title}
              className="glass-card p-6 flex flex-col justify-between border-[#d4af37]/35 hover:border-[#b45309]/50 relative overflow-hidden bg-white/92 shadow-lg"
            >
              <MandalaWatermark position="bottom-right" size={180} opacity={0.08} />
              <div className="relative z-10">
                <div className="flex items-center justify-between gap-2 mb-3">
                  <div className="p-2 rounded-lg bg-amber-500/10 text-amber-800 border border-[#d4af37]/30">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className={`pill-badge text-[10px] ${item.badgeColor}`}>
                    {item.badge}
                  </span>
                </div>
                <h4 className="text-base font-bold text-slate-900 mb-3">{item.title}</h4>

                <div className="space-y-3 text-xs leading-relaxed text-slate-600">
                  <div>
                    <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">
                      Physical Mechanism:
                    </span>
                    <p>{item.mechanism}</p>
                  </div>

                  <div>
                    <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">
                      Observed Impact:
                    </span>
                    <p>{item.impact}</p>
                  </div>
                </div>
              </div>

              <div className="mt-5 pt-3.5 border-t border-[#d4af37]/20 text-[11px] text-slate-700 relative z-10">
                <strong className="text-amber-800 font-semibold">Phase 3 Mitigation:</strong> {item.mitigation}
              </div>
            </div>
          );
        })}
      </div>

      {/* Failure Case Visual Showcase */}
      <div className="glass-panel p-6 sm:p-8 border border-[#d4af37]/35 shadow-xl bg-white/92 relative overflow-hidden">
        <MandalaWatermark position="bottom-left" size={240} opacity={0.06} />
        <div className="relative z-10">
          <h4 className="text-lg font-bold text-slate-900 mb-1">
            Empirical Failure Showcase: Cloud Boundary &amp; Shadow False Positives
          </h4>
          <p className="text-xs sm:text-sm text-slate-600 mb-6">
            Direct inspection of actual test samples where optical limitations triggered false alarms (Scene <code className="text-amber-800 font-mono bg-amber-100/60 px-1.5 py-0.5 rounded border border-[#d4af37]/40">a1__07_13</code>):
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            {/* Tile 1 */}
            <div className="group rounded-xl overflow-hidden bg-slate-50 border border-[#d4af37]/35 p-2.5 shadow-md">
              <div className="relative aspect-square w-full rounded-lg overflow-hidden mb-2">
                <Image
                  src="/assets/demo/failure_cloud_rgb.png"
                  alt="Cloud Tile Sentinel-2 RGB"
                  fill
                  sizes="(max-width: 768px) 100vw, 33vw"
                  className="object-cover group-hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="text-[11px] font-semibold text-slate-700 text-center py-1">
                1. Sentinel-2 RGB (Cloud Scene a1__07_13)
              </div>
            </div>

            {/* Tile 2 */}
            <div className="group rounded-xl overflow-hidden bg-slate-50 border border-[#d4af37]/35 p-2.5 shadow-md">
              <div className="relative aspect-square w-full rounded-lg overflow-hidden mb-2">
                <Image
                  src="/assets/demo/failure_cloud_overlay.png"
                  alt="Prediction Overlay on Cloud Scene"
                  fill
                  sizes="(max-width: 768px) 100vw, 33vw"
                  className="object-cover group-hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="text-[11px] font-semibold text-rose-800 text-center py-1">
                2. False Alarms along Cloud Margin (Red Overlay)
              </div>
            </div>

            {/* Tile 3 */}
            <div className="group rounded-xl overflow-hidden bg-slate-50 border border-[#d4af37]/35 p-2.5 shadow-md">
              <div className="relative aspect-square w-full rounded-lg overflow-hidden mb-2">
                <Image
                  src="/assets/demo/failure_cloud_confusion.png"
                  alt="Spatial Confusion Map"
                  fill
                  sizes="(max-width: 768px) 100vw, 33vw"
                  className="object-cover group-hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="text-[11px] font-semibold text-amber-800 text-center py-1">
                3. Spatial Confusion Map (Red = Commission Error)
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
