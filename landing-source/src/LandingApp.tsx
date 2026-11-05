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

/**
 * Public marketing landing page — compiled to static assets and served by Flask at `/`.
 * Sign In uses a real document navigation to Flask `/signin` (not client routing).
 */
export function LandingApp() {
  return (
    <>
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
    </>
  );
}
