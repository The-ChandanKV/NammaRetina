import {
  Brain,
  ChartLine,
  FileDown,
  ScanEye,
  Sparkles,
  Waves,
  type LucideIcon,
} from "lucide-react";

import { Reveal, SectionHeading } from "./Reveal";

interface Feature {
  icon: LucideIcon;
  title: string;
  description: string;
  note?: string;
  span?: string;
}

const features: Feature[] = [
  {
    icon: ScanEye,
    title: "AI-Assisted DR Screening",
    description:
      "Analyze retinal fundus images using a deep-learning model that classifies diabetic retinopathy across five severity levels.",
    span: "lg:col-span-2",
  },
  {
    icon: Brain,
    title: "Explainable AI Insights",
    description:
      "Use Grad-CAM heatmaps and visual explanations to explore regions influencing AI-generated assessments.",
  },
  {
    icon: ChartLine,
    title: "Patient Progression Tracking",
    description:
      "Review historical screening results and visualize changes in recorded disease severity over time.",
  },
  {
    icon: Waves,
    title: "Retinal Progression Visualization",
    description:
      "Explore illustrative disease-progression visualizations that support educational and clinical discussion.",
    note: "Illustrative simulation — not a validated forecast.",
    span: "lg:col-span-2",
  },
  {
    icon: Sparkles,
    title: "AI-Assisted Report Interpretation",
    description:
      "Generate readable explanations and supporting clinical information using integrated language-model capabilities.",
  },
  {
    icon: FileDown,
    title: "Structured PDF Reporting",
    description:
      "Create downloadable reports containing screening results, explanatory images, and relevant clinical context.",
    span: "lg:col-span-2",
  },
];

export function Features() {
  return (
    <section id="platform" className="scroll-mt-24 py-20 sm:py-24 lg:py-28" aria-labelledby="platform-heading">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Platform"
          title="Built for the Way Clinicians Work."
          description="From retinal imaging to longitudinal insights, one connected workflow."
        />

        <div className="mt-14 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, i) => (
            <Reveal
              key={feature.title}
              delay={0.06 * i}
              className={feature.span ?? ""}
            >
              <article
                className={`group relative h-full overflow-hidden rounded-2xl border border-border bg-card p-6 transition-all duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-[0_24px_50px_-24px_color-mix(in_oklab,var(--color-primary)_55%,transparent)] sm:p-7 ${
                  feature.span ? "" : ""
                }`}
              >
                {/* hover accent glow */}
                <div
                  className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-primary)_12%,transparent),transparent_70%)] opacity-0 transition-opacity duration-500 group-hover:opacity-100"
                  aria-hidden="true"
                />
                <span className="flex h-11 w-11 items-center justify-center rounded-xl border border-primary/20 bg-primary/8 text-primary transition-colors group-hover:bg-primary/15">
                  <feature.icon className="h-5 w-5" aria-hidden="true" />
                </span>
                <h3 className="mt-5 text-lg font-semibold tracking-tight text-foreground">
                  {feature.title}
                </h3>
                <p className="mt-2.5 text-sm leading-relaxed text-muted-foreground">
                  {feature.description}
                </p>
                {feature.note ? (
                  <p className="mt-4 inline-flex rounded-full border border-border bg-muted px-3 py-1 text-[11px] font-medium text-muted-foreground">
                    {feature.note}
                  </p>
                ) : null}
              </article>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
