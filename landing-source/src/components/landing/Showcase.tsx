import { useRef, useState } from "react";
import { ChevronsLeftRight, FileCheck2, FileImage, Layers, ListChecks, Stethoscope, type LucideIcon } from "lucide-react";

import { Reveal, SectionHeading } from "./Reveal";
import fundusPrimary from "@/assets/fundus-primary.jpg";
import fundusHeatmap from "@/assets/fundus-heatmap.jpg";

const MIN_POS = 4;
const MAX_POS = 96;

function CompareSlider() {
  const [pos, setPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const updateFromClientX = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const next = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(MAX_POS, Math.max(MIN_POS, next)));
  };

  return (
    <div
      ref={containerRef}
      className="group relative aspect-square w-full max-w-xl cursor-ew-resize touch-none select-none overflow-hidden rounded-3xl border border-white/10 shadow-[0_50px_120px_-40px_rgb(0_0_0_/_0.8)]"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        setDragging(true);
        updateFromClientX(e.clientX);
      }}
      onPointerMove={(e) => {
        if (dragging) updateFromClientX(e.clientX);
      }}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
    >
      {/* before: original fundus image */}
      <img
        src={fundusPrimary}
        width={1024}
        height={1024}
        loading="lazy"
        alt="Example retinal fundus image (demonstration data)"
        className="absolute inset-0 h-full w-full object-cover"
        draggable={false}
      />
      {/* after: simulated Grad-CAM overlay, clipped to the right of the handle */}
      <img
        src={fundusHeatmap}
        width={1024}
        height={1024}
        loading="lazy"
        alt="The same example image with a simulated Grad-CAM-style visual explanation overlay (demonstration data)"
        className="absolute inset-0 h-full w-full object-cover"
        style={{ clipPath: `inset(0 0 0 ${pos}%)` }}
        draggable={false}
      />

      {/* corner labels */}
      <span className="absolute left-4 top-4 rounded-full bg-black/55 px-3 py-1 text-[11px] font-semibold tracking-wide text-white backdrop-blur-sm">
        Fundus Image
      </span>
      <span className="absolute right-4 top-4 rounded-full bg-black/55 px-3 py-1 text-[11px] font-semibold tracking-wide text-white backdrop-blur-sm">
        Visual Explanation
      </span>

      {/* drag handle */}
      <div
        role="slider"
        tabIndex={0}
        aria-label="Comparison slider: drag or use arrow keys to reveal the simulated Grad-CAM overlay"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pos)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
            e.preventDefault();
            setPos((p) => Math.max(MIN_POS, p - 4));
          } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
            e.preventDefault();
            setPos((p) => Math.min(MAX_POS, p + 4));
          }
        }}
        className="absolute inset-y-0 z-10 w-0 outline-none"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute inset-y-0 -left-px w-0.5 bg-white/90 shadow-[0_0_18px_rgb(255_255_255_/_0.6)]" />
        <button
          type="button"
          tabIndex={-1}
          className="absolute left-1/2 top-1/2 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/40 bg-white/15 text-white backdrop-blur-md transition-transform group-hover:scale-105"
          aria-hidden="true"
        >
          <ChevronsLeftRight className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}

interface Panel {
  icon: LucideIcon;
  title: string;
  children: React.ReactNode;
}

function Panel({ icon: Icon, title, children }: Panel) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-5 backdrop-blur-sm transition-colors hover:border-white/20">
      <div className="flex items-center gap-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/15 text-[color-mix(in_oklab,var(--color-primary)_70%,white)]">
          <Icon className="h-4 w-4" aria-hidden="true" />
        </span>
        <h3 className="text-sm font-semibold text-night-foreground">{title}</h3>
      </div>
      <div className="mt-4">{children}</div>
    </div>
  );
}

const severityLevels = [
  { label: "No apparent diabetic retinopathy", tone: "bg-emerald-400" },
  { label: "Mild", tone: "bg-lime-400" },
  { label: "Moderate", tone: "bg-amber-400", active: true },
  { label: "Severe", tone: "bg-orange-400" },
  { label: "Proliferative", tone: "bg-red-400" },
];

export function Showcase() {
  return (
    <section
      id="showcase"
      className="relative scroll-mt-24 overflow-hidden bg-night py-20 text-night-foreground sm:py-24 lg:py-28"
      aria-labelledby="showcase-heading"
    >
      {/* ambience: fine grid + soft glows */}
      <div className="pointer-events-none absolute inset-0" aria-hidden="true">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgb(255_255_255_/_0.04)_1px,transparent_1px),linear-gradient(to_bottom,rgb(255_255_255_/_0.04)_1px,transparent_1px)] bg-[size:56px_56px] [mask-image:radial-gradient(ellipse_at_center,black_30%,transparent_75%)]" />
        <div className="absolute -left-40 top-10 h-96 w-96 rounded-full bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-primary)_20%,transparent),transparent_70%)] blur-3xl" />
        <div className="absolute -right-40 bottom-0 h-96 w-96 rounded-full bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-accent)_16%,transparent),transparent_70%)] blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          dark
          eyebrow="Interactive Showcase"
          title="Every Pixel Tells a Story."
          description="Explore how AI-assisted visual explanations can help clinicians inspect retinal screening results."
        />

        <Reveal delay={0.1} className="mt-14 flex justify-center">
          <CompareSlider />
        </Reveal>

        <Reveal delay={0.15}>
          <p className="mt-6 text-center text-xs text-night-muted">
            Drag the slider (or use arrow keys) to compare the example fundus image with a simulated
            Grad-CAM-style overlay.
          </p>
        </Reveal>

        {/* example interface panels */}
        <div className="mx-auto mt-12 grid max-w-5xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Reveal delay={0.05}>
            <Panel icon={FileImage} title="Retinal Image">
              <img
                src={fundusPrimary}
                width={1024}
                height={1024}
                loading="lazy"
                alt="Example retinal fundus thumbnail (demonstration data)"
                className="aspect-square w-full rounded-xl border border-white/10 object-cover"
              />
            </Panel>
          </Reveal>
          <Reveal delay={0.1}>
            <Panel icon={Layers} title="Visual Explanation">
              <img
                src={fundusHeatmap}
                width={1024}
                height={1024}
                loading="lazy"
                alt="Simulated Grad-CAM heatmap thumbnail (demonstration data)"
                className="aspect-square w-full rounded-xl border border-white/10 object-cover"
              />
            </Panel>
          </Reveal>
          <Reveal delay={0.15}>
            <Panel icon={ListChecks} title="Severity Assessment">
              <ul className="space-y-2">
                {severityLevels.map((level) => (
                  <li
                    key={level.label}
                    className={`flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[12px] ${
                      level.active
                        ? "bg-white/10 font-semibold text-night-foreground"
                        : "text-night-muted"
                    }`}
                  >
                    <span className={`h-2 w-2 shrink-0 rounded-full ${level.tone}`} aria-hidden="true" />
                    {level.label}
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-[10px] leading-relaxed text-night-muted">
                Example output — demonstration data only.
              </p>
            </Panel>
          </Reveal>
          <Reveal delay={0.2}>
            <Panel icon={Stethoscope} title="Clinician Review">
              <div className="space-y-3 text-[12px] text-night-muted">
                <div className="flex items-start gap-2.5">
                  <FileCheck2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
                  <p>
                    Findings are reviewed by the clinician before a report is exported. The
                    platform supports—not replaces—professional judgment.
                  </p>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
                  <p className="font-medium text-night-foreground">Status</p>
                  <p>Pending clinician review · Example</p>
                </div>
              </div>
            </Panel>
          </Reveal>
        </div>

        <Reveal delay={0.1}>
          <p className="mt-10 text-center text-xs font-medium tracking-wide text-night-muted">
            Illustrative demonstration — not an actual patient diagnosis.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
