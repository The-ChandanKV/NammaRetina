import {
  Aperture,
  Cpu,
  Database,
  Layers,
  Network,
  Server,
  type LucideIcon,
} from "lucide-react";

import { Reveal, SectionHeading } from "./Reveal";

interface Tech {
  icon: LucideIcon;
  name: string;
  role: string;
  description: string;
}

const techs: Tech[] = [
  {
    icon: Network,
    name: "EfficientNetB0",
    role: "Classification model",
    description:
      "A compact convolutional network that estimates diabetic retinopathy severity from retinal fundus images across five levels.",
  },
  {
    icon: Cpu,
    name: "TensorFlow / Keras",
    role: "Deep-learning inference",
    description:
      "Runs the trained model's inference pipeline that powers the platform's screening assessments.",
  },
  {
    icon: Aperture,
    name: "OpenCV",
    role: "Image preprocessing",
    description:
      "Prepares retinal photographs for analysis—cropping, illumination normalization, and quality checks.",
  },
  {
    icon: Layers,
    name: "Grad-CAM",
    role: "Visual interpretability",
    description:
      "Highlights the image regions that influenced each assessment, so clinicians can inspect the evidence behind a result.",
  },
  {
    icon: Server,
    name: "Python Flask",
    role: "Application backend",
    description:
      "Serves the clinical application itself: authentication, dashboards, patient management, and report generation.",
  },
  {
    icon: Database,
    name: "Supabase",
    role: "Data platform",
    description:
      "Underpins authentication, patient records, and image storage across the existing system.",
  },
];

export function Technology() {
  return (
    <section
      id="technology"
      className="scroll-mt-24 bg-secondary/40 py-20 sm:py-24 lg:py-28"
      aria-labelledby="technology-heading"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Technology"
          title="Powered by Intelligent Technology."
          description="A focused, proven stack—each layer chosen for what it enables in a clinical workflow."
        />

        <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {techs.map((tech, i) => (
            <Reveal key={tech.name} delay={0.06 * i}>
              <article className="group relative h-full overflow-hidden rounded-2xl border border-border bg-card p-6 transition-all duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-[0_24px_50px_-24px_color-mix(in_oklab,var(--color-primary)_50%,transparent)]">
                <div className="flex items-center justify-between gap-3">
                  <span className="flex h-11 w-11 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary transition-colors group-hover:bg-primary/15">
                    <tech.icon className="h-5 w-5" aria-hidden="true" />
                  </span>
                  <span className="rounded-full border border-border bg-muted px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {tech.role}
                  </span>
                </div>
                <h3 className="mt-5 font-mono text-base font-semibold tracking-tight text-foreground">
                  {tech.name}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {tech.description}
                </p>
              </article>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.1}>
          <p className="mx-auto mt-10 max-w-2xl text-center text-xs leading-relaxed text-muted-foreground">
            Technology descriptions reflect the implementation of the NammaRetina project. No model
            performance claims, medical certifications, or regulatory approvals are stated or implied.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
