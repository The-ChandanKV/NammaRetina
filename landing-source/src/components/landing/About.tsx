import { ArrowUpRight, GraduationCap } from "lucide-react";

import { Reveal } from "./Reveal";
import { site } from "@/lib/site";

export function About() {
  return (
    <section id="about" className="scroll-mt-24 py-20 sm:py-24 lg:py-28" aria-labelledby="about-heading">
      <div className="mx-auto grid max-w-7xl gap-12 px-4 sm:px-6 lg:grid-cols-2 lg:gap-16 lg:px-8">
        <div>
          <Reveal>
            <span className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/5 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
              About
            </span>
          </Reveal>
          <Reveal delay={0.05}>
            <h2
              id="about-heading"
              className="mt-5 text-balance text-3xl font-bold leading-tight tracking-tight text-foreground sm:text-4xl lg:text-[2.75rem]"
            >
              Technology in Service of Better Retinal Care.
            </h2>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="mt-6 space-y-4 text-pretty text-base leading-relaxed text-muted-foreground">
              <p>
                NammaRetina is an academic AI-assisted retinal screening project exploring how deep
                learning, explainable AI, and longitudinal analysis can support the early assessment
                and monitoring of diabetic retinopathy.
              </p>
              <p>
                The platform brings together image analysis, visual interpretation,
                patient-history tracking, and structured reporting in a clinician-oriented
                experience.
              </p>
            </div>
          </Reveal>
        </div>

        <div className="flex flex-col justify-center gap-4">
          <Reveal delay={0.1}>
            <div className="rounded-2xl border border-border bg-card p-6 sm:p-8">
              <span className="flex h-11 w-11 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary">
                <GraduationCap className="h-5 w-5" aria-hidden="true" />
              </span>
              <h3 className="mt-5 text-lg font-semibold tracking-tight text-foreground">
                An open academic project
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                NammaRetina is presented as a research and education effort. It has not undergone
                clinical certification or commercial deployment, and its outputs are intended to
                support—not replace—professional clinical judgment.
              </p>
              <a
                href={site.githubUrl}
                target="_blank"
                rel="noreferrer noopener"
                className="group mt-5 inline-flex items-center gap-2 rounded-full border border-border bg-background px-5 py-2.5 text-sm font-semibold text-foreground transition-colors hover:border-primary/40 hover:bg-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                View on GitHub
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" aria-hidden="true" />
              </a>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
