import { ArrowUpRight, Github } from "lucide-react";

import { site } from "@/lib/site";
import { DoctorSignInLink } from "./DoctorSignInLink";
import { Logo } from "./Logo";

export function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="bg-night pt-16 text-night-foreground" aria-labelledby="footer-heading">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <h2 id="footer-heading" className="sr-only">
          Footer
        </h2>
        <div className="grid gap-12 pb-12 lg:grid-cols-[1.4fr_1fr_1fr_1fr]">
          {/* brand */}
          <div>
            <a href="/" className="inline-flex items-center gap-2.5 rounded-md focus-visible:outline-2 focus-visible:outline-primary">
              <Logo className="h-8 w-8" />
              <span className="text-[1.05rem] font-bold tracking-tight">
                Namma<span className="text-primary">Retina</span>
              </span>
            </a>
            <p className="mt-4 max-w-sm text-sm leading-relaxed text-night-muted">
              Advancing accessible, explainable AI-assisted retinal screening—so clinicians can see
              beyond the scan and understand disease progression.
            </p>
            <a
              href={site.githubUrl}
              target="_blank"
              rel="noreferrer noopener"
              className="group mt-5 inline-flex items-center gap-2 rounded-full border border-white/15 px-4 py-2 text-sm font-medium text-night-foreground transition-colors hover:border-primary/50 hover:bg-white/5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            >
              <Github className="h-4 w-4" aria-hidden="true" />
              GitHub
              <ArrowUpRight className="h-3.5 w-3.5 opacity-60 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" aria-hidden="true" />
            </a>
          </div>

          {/* link columns */}
          <nav aria-label="Platform">
            <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-night-muted">Platform</h3>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li><a className="transition-colors hover:text-primary" href="#platform">Features</a></li>
              <li><a className="transition-colors hover:text-primary" href="#how-it-works">How It Works</a></li>
              <li><a className="transition-colors hover:text-primary" href="#showcase">Interactive Showcase</a></li>
              <li><DoctorSignInLink className="transition-colors hover:text-primary">Doctor Sign In</DoctorSignInLink></li>
            </ul>
          </nav>
          <nav aria-label="Technology">
            <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-night-muted">Technology</h3>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li><a className="transition-colors hover:text-primary" href="#technology">Technology Stack</a></li>
              <li><a className="transition-colors hover:text-primary" href="#workflow-preview">Workflow Preview</a></li>
              <li><a className="transition-colors hover:text-primary" href={site.githubUrl}>Source Code</a></li>
            </ul>
          </nav>
          <nav aria-label="About">
            <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-night-muted">About</h3>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li><a className="transition-colors hover:text-primary" href="#about">About the Project</a></li>
              <li><a className="transition-colors hover:text-primary" href="#get-started">Access the Portal</a></li>
            </ul>
          </nav>
        </div>

        {/* disclaimer */}
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
          <p className="text-xs leading-relaxed text-night-muted">
            NammaRetina is an academic, AI-assisted decision-support project. Its outputs are not a
            substitute for professional clinical judgment and are not presented as clinically
            validated diagnoses or prognoses.
          </p>
        </div>

        <div className="flex flex-col items-center justify-between gap-3 border-t border-white/10 py-6 text-xs text-night-muted sm:flex-row">
          <p>© {year} NammaRetina. An academic research project.</p>
          <p>All patient data shown on this site is fictional demonstration data.</p>
        </div>
      </div>
    </footer>
  );
}
