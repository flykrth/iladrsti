import "./globals.css";
import { Inter, JetBrains_Mono } from "next/font/google";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata = {
  title: "Ilādṛṣṭi (इलादृष्टि) — Multispectral Wildfire Semantic Segmentation",
  description:
    "Earth Observation Deep Learning Dashboard for Autonomous Multispectral Wildfire Detection & Burned Area Delineation using Sentinel-2 L2A Satellite Imagery.",
  keywords: [
    "Wildfire Detection",
    "Sentinel-2",
    "Multispectral",
    "Focal-Tversky Loss",
    "ResNet-50",
    "Semantic Segmentation",
    "U-Net",
    "Earth Observation",
  ],
  authors: [{ name: "Ilādṛṣṭi Team" }],
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
          crossOrigin="anonymous"
        />
      </head>
      <body className="bg-background-deep font-sans text-slate-100 antialiased selection:bg-brand-pink selection:text-white">
        {/* Vibrant Animated Aurora Mesh Gradient Background */}
        <div className="aurora-bg">
          <div className="aurora-mesh" />
          <div className="aurora-noise" />
        </div>

        {children}
      </body>
    </html>
  );
}
