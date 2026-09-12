import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Methodology from "@/components/Methodology";
import ImageSlider from "@/components/ImageSlider";
import MetricsDashboard from "@/components/MetricsDashboard";
import Limitations from "@/components/Limitations";
import Footer from "@/components/Footer";

export default function HomePage() {
  return (
    <main className="min-h-screen relative flex flex-col justify-between">
      <Navbar />
      <div className="flex-1">
        <Hero />
        <Methodology />
        <ImageSlider />
        <MetricsDashboard />
        <Limitations />
      </div>
      <Footer />
    </main>
  );
}
