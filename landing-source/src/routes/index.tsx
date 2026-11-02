import { createFileRoute } from "@tanstack/react-router";
import { MotionConfig } from "framer-motion";

import { Navbar } from "@/components/landing/Navbar";
import { Hero } from "@/components/landing/Hero";
import { Features } from "@/components/landing/Features";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { Showcase } from "@/components/landing/Showcase";
import { WorkflowPreview } from "@/components/landing/WorkflowPreview";
import { Technology } from "@/components/landing/Technology";
import { About } from "@/components/landing/About";
import { FinalCta } from "@/components/landing/FinalCta";
import { Footer } from "@/components/landing/Footer";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "NammaRetina — See Beyond the Scan. Understand the Progression." },
      {
        name: "description",
        content:
          "An AI-assisted retinal screening platform helping clinicians assess diabetic retinopathy severity, visualize model insights, monitor patient progression, and generate structured reports.",
      },
      {
        property: "og:title",
        content: "NammaRetina — See Beyond the Scan. Understand the Progression.",
      },
      {
        property: "og:description",
        content:
          "An AI-assisted retinal screening platform for clinicians: DR severity assessment, explainable AI insights, progression tracking, and structured reporting.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: LandingPage,
});

function LandingPage() {
  return (
    <MotionConfig reducedMotion="user">
      <Navbar />
      <main>
        <Hero />
        <Features />
        <HowItWorks />
        <Showcase />
        <WorkflowPreview />
        <Technology />
        <About />
        <FinalCta />
      </main>
      <Footer />
    </MotionConfig>
  );
}
