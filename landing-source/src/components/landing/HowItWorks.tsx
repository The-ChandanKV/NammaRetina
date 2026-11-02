import { Cpu, FileCheck2, ScanEye, Upload, type LucideIcon } from "lucide-react";

import { Reveal, SectionHeading } from "./Reveal";

interface Step {
  icon: LucideIcon;
  title: string;
  description: string;
}

const steps: Step[] = [
  {
    icon: Upload,
    title: "Upload Retinal Scan",
    description: "Upload a retinal fundus image and associate it with a patient record.",
  },
  {
    icon: Cpu,
    title: "AI-Assisted Assessment",
    description:
      "The EfficientNetB0-based model processes the image and estimates diabetic retinopathy severity.",
  },
  {
    icon: ScanEye,
    title: "Explain and Monitor",
    description:
      "Explore visual explanations, compare historical examinations, and review progression information.",
  },
  {
    icon: FileCheck2,
    title: "Generate Clinical Report",
    description:
      "Review the findings and export a structured report to support clinician decision-making.",
  },
];

export function HowItWorks() {
  return (
    <section
      id="how-it-works"
      className="scroll-mt-24 bg-secondary/40 py-20 sm:py-24 lg:py-28"
      aria-labelledby="how-it-works-heading"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Workflow"
          title="From Image to Insight."
          description="A four-step, clinician-oriented path from a retinal image to a structured clinical report."
        />

        <div className="relative mt-16">
          {/* connecting line — horizontal on desktop, vertical on mobile */}
          <div
            className="absolute left-[1.4rem] top-0 h-full w-px bg-gradient-to-b from-primary/40 via-primary/20 to-transparent md:left-0 md:top-[2.4rem] md:h-px md:w-full md:bg-gradient-to-r"
            aria-hidden="true"
          />

          <ol className="relative grid gap-10 md:grid-cols-4 md:gap-6">
            {steps.map((step, i) => (
              <Reveal key={step.title} delay={0.1 * i}>
                <li className="flex gap-5 md:flex-col md:gap-0">
                  <div className="relative z-10 flex flex-col items-center md:items-start">
                    <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-primary/30 bg-background text-sm font-bold text-primary shadow-[0_0_0_6px_var(--color-background)]">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                  </div>
                  <div className="pb-2 md:mt-6 md:pb-0">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                      <step.icon className="h-5 w-5" aria-hidden="true" />
                    </span>
                    <h3 className="mt-3 text-base font-semibold tracking-tight text-foreground">
                      {step.title}
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                      {step.description}
                    </p>
                  </div>
                </li>
              </Reveal>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
