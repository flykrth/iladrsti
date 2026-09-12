import Link from "next/link";
import { Github, Satellite, ShieldCheck, Terminal } from "lucide-react";
import { MandalaBadge } from "./Mandala";

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-[#d4af37]/35 bg-[#fcfbfa]/95 backdrop-blur-xl py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto flex flex-col gap-8">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <div className="text-xl font-extrabold text-slate-900 flex items-center gap-2 mb-2">
              <MandalaBadge size={26} />
              Ilādṛṣṭi <span className="text-[#854d0e] text-sm font-semibold">इलादृष्टि</span>
            </div>
            <p className="text-xs sm:text-sm text-slate-600 max-w-xl leading-relaxed">
              Multispectral Semantic Segmentation for Wildfire Progression.<br />
              Developed for <strong className="text-slate-900 font-semibold">Track 6: AI for Science &amp; Society</strong> at the Deep Learning Hackathon, Amrita Vishwa Vidyapeetham.
            </p>
          </div>

          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="pill-badge seed flex items-center gap-1">
              <Terminal className="w-3 h-3" /> PyTorch 2.0+
            </span>
            <span className="pill-badge track flex items-center gap-1">
              <ShieldCheck className="w-3 h-3" /> Seed 42 Deterministic
            </span>
            <span className="pill-badge success flex items-center gap-1">
              <Satellite className="w-3 h-3" /> ESA Copernicus Data
            </span>
            <Link
              href="https://github.com/flykrth/iladrsti"
              target="_blank"
              rel="noopener noreferrer"
              className="glass-button text-xs py-1.5 px-3.5"
            >
              <Github className="w-4 h-4" />
              GitHub Repository
            </Link>
          </div>
        </div>

        <div className="pt-6 border-t border-[#d4af37]/20 flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px] text-slate-500">
          <div>
            &copy; 2026 Ilādṛṣṭi Earth Observation Project. Released under the MIT License.
          </div>
          <div>
            Architecture: Transfer-Adapted ResNet-50 U-Net &bull; Latency: 36.4 FPS &bull; Resolution: 10m GSD
          </div>
        </div>
      </div>
    </footer>
  );
}
