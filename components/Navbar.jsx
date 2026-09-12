"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Menu, X } from "lucide-react";
import { MandalaBadge } from "./Mandala";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState("hero");

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 30);

      const sections = ["hero", "methodology", "interactive-demo", "metrics", "limitations"];
      const current = sections.find((id) => {
        const el = document.getElementById(id);
        if (el) {
          const rect = el.getBoundingClientRect();
          return rect.top <= 160 && rect.bottom >= 160;
        }
        return false;
      });

      if (current) {
        setActiveSection(current);
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { href: "#hero", label: "Overview", id: "hero" },
    { href: "#methodology", label: "Architecture & Loss", id: "methodology" },
    { href: "#interactive-demo", label: "Interactive Demo", id: "interactive-demo" },
    { href: "#metrics", label: "Metrics & Ablation", id: "metrics" },
    { href: "#limitations", label: "Failure Analysis", id: "limitations" },
  ];

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/90 backdrop-blur-xl border-b border-[#d4af37]/35 py-3 shadow-md shadow-[#d4af37]/10"
          : "bg-white/75 backdrop-blur-md border-b border-[#d4af37]/25 py-3.5"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-3">
        {/* Brand in White & Gold */}
        <Link href="#hero" className="flex items-center gap-3 group shrink-0">
          <MandalaBadge size={40} className="group-hover:scale-105 transition-transform" />
          <div className="flex flex-col">
            <div className="text-lg font-extrabold tracking-tight text-slate-900 flex items-center gap-1.5">
              Ilādṛṣṭi <span className="text-[#b45309] text-xs font-semibold">इलादृष्टि</span>
            </div>
            <span className="text-[10px] uppercase font-bold tracking-widest text-[#854d0e]">
              Multispectral Wildfire AI
            </span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <ul className="hidden md:flex items-center gap-1 xl:gap-2 shrink-0">
          {navLinks.map((link) => (
            <li key={link.id} className="shrink-0">
              <Link
                href={link.href}
                className={`px-3 py-1.5 rounded-full text-xs xl:text-sm font-semibold transition-all duration-200 whitespace-nowrap inline-block ${
                  activeSection === link.id
                    ? "bg-[#d4af37]/20 text-[#854d0e] border border-[#d4af37]/50 shadow-sm"
                    : "text-slate-700 hover:text-slate-950 hover:bg-[#d4af37]/10"
                }`}
              >
                {link.label}
              </Link>
            </li>
          ))}
        </ul>

        {/* Badges & Mobile Toggle */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="hidden sm:inline-flex pill-badge track">
            <span className="pulse-dot fire" /> Track 6: AI for Science
          </span>
          <span className="hidden xl:inline-flex pill-badge seed">
            Seed 42 Locked
          </span>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg bg-amber-50/80 border border-[#d4af37]/35 text-slate-700 hover:text-slate-950"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden px-4 pt-3 pb-5 mt-2 bg-white/95 backdrop-blur-2xl border-b border-[#d4af37]/30 shadow-2xl flex flex-col gap-2">
          {navLinks.map((link) => (
            <Link
              key={link.id}
              href={link.href}
              onClick={() => setMobileMenuOpen(false)}
              className={`px-4 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                activeSection === link.id
                  ? "bg-[#d4af37]/20 text-[#854d0e] border border-[#d4af37]/40"
                  : "text-slate-700 hover:bg-[#d4af37]/10 hover:text-slate-950"
              }`}
            >
              {link.label}
            </Link>
          ))}
          <div className="pt-2 flex items-center gap-2 flex-wrap">
            <span className="pill-badge track">
              <span className="pulse-dot fire" /> Track 6: AI for Science
            </span>
            <span className="pill-badge seed">Seed 42 Locked</span>
          </div>
        </div>
      )}
    </nav>
  );
}
