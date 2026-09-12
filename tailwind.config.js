/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          deep: "#0b0817",
          indigo: "#130c25",
          purple: "#2a0845",
          pink: "#6b114d",
        },
        brand: {
          pink: "#ff4b72",
          purple: "#8b5cf6",
          cyan: "#06b6d4",
          emerald: "#10b981",
          amber: "#f59e0b",
          red: "#ef4444",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
      backdropBlur: {
        xs: "2px",
        glass: "14px",
        heavy: "20px",
      },
      boxShadow: {
        glass: "0 8px 32px 0 rgba(0, 0, 0, 0.25)",
        "glass-hover": "0 16px 40px 0 rgba(0, 0, 0, 0.38), 0 0 24px rgba(255, 75, 114, 0.2)",
        glow: "0 0 20px rgba(255, 75, 114, 0.35)",
        "glow-cyan": "0 0 20px rgba(6, 182, 212, 0.35)",
      },
      keyframes: {
        auroraMove: {
          "0%": { transform: "translate(0, 0) rotate(0deg) scale(1)" },
          "50%": { transform: "translate(-5%, 8%) rotate(6deg) scale(1.08)" },
          "100%": { transform: "translate(6%, -5%) rotate(-5deg) scale(1.03)" },
        },
        auroraPulse: {
          "0%": { opacity: "0.5", transform: "scale(1)" },
          "50%": { opacity: "0.85", transform: "scale(1.12)" },
          "100%": { opacity: "0.6", transform: "scale(0.98)" },
        },
        pulseSlow: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.4", transform: "scale(0.9)" },
        },
      },
      animation: {
        "aurora-move": "auroraMove 24s ease-in-out infinite alternate",
        "aurora-pulse": "auroraPulse 14s ease-in-out infinite alternate",
        "pulse-slow": "pulseSlow 2.5s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
