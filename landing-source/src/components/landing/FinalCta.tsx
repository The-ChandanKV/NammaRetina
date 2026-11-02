import { ArrowRight, ShieldCheck } from "lucide-react";

import { DoctorSignInLink } from "./DoctorSignInLink";
import { Reveal } from "./Reveal";

export function FinalCta() {
  return (
    <section
      id="get-started"
      className="relative overflow-hidden bg-night py-24 text-night-foreground sm:py-28"
      aria-labelledby="final-cta-heading"
    >
      <div className="pointer-events-none absolute inset-0" aria-hidden="true">
        <div className="absolute left-1/2 top-1/2 h-[42rem] w-[42rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-primary)_18%,transparent),transparent_65%)] blur-3xl" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgb(255_255_255_/_0.035)_1px,transparent_1px),linear-gradient(to_bottom,rgb(255_255_255_/_0.035)_1px,transparent_1px)] bg-[size:56px_56px] [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_70%)]" />
      </div>

      <div className="relative mx-auto max-w-3xl px-4 text-center sm:px-6 lg:px-8">
        <Reveal>
          <h2
            id="final-cta-heading"
            className="text-balance text-3xl font-bold leading-tight tracking-tight sm:text-4xl lg:text-[2.75rem]"
          >
            Bring Clarity to Every{" "}
            <span className="text-gradient-teal">Retinal Assessment.</span>
          </h2>
        </Reveal>
        <Reveal delay={0.08}>
          <p className="mx-auto mt-5 max-w-xl text-pretty text-base leading-relaxed text-night-muted sm:text-lg">
            Explore an integrated platform for AI-assisted screening, explainable insights, and
            patient progression tracking.
          </p>
        </Reveal>
        <Reveal delay={0.14}>
          <div className="mt-10 flex justify-center">
            <DoctorSignInLink
              className="group inline-flex items-center gap-2 rounded-full bg-primary px-8 py-4 text-sm font-semibold text-primary-foreground shadow-[0_24px_60px_-20px_color-mix(in_oklab,var(--color-primary)_80%,transparent)] transition-all hover:bg-primary/90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            >
              Access Doctor Portal
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" aria-hidden="true" />
            </DoctorSignInLink>
          </div>
        </Reveal>
        <Reveal delay={0.2}>
          <p className="mt-6 flex items-center justify-center gap-2 text-xs text-night-muted">
            <ShieldCheck className="h-4 w-4 text-primary" aria-hidden="true" />
            Access is intended for authorized clinical users.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
