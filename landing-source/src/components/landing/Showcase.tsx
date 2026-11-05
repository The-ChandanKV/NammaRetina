import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import {
  FileCheck2,
  FileImage,
  Layers,
  ListChecks,
  Stethoscope,
  type LucideIcon,
} from "lucide-react";

import { Reveal, SectionHeading } from "./Reveal";
import fundusPrimary from "@/assets/fundus-primary.jpg";
import fundusHeatmap from "@/assets/fundus-heatmap.jpg";

type OverlayMode = "original" | "overlay";

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${src}`));
    img.src = src;
  });
}

/**
 * Builds a translucent heat-only layer from the illustrative heatmap asset by
 * keeping warm/cyan attention colors and clearing near-neutral fundus tones.
 * Result is spatially aligned with the original photograph.
 */
async function buildHeatOnlyCanvas(
  heatSrc: string,
  size = 768,
): Promise<HTMLCanvasElement> {
  const heat = await loadImage(heatSrc);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return canvas;

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(heat, 0, 0, size, size);
  const image = ctx.getImageData(0, 0, size, size);
  const { data } = image;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] ?? 0;
    const g = data[i + 1] ?? 0;
    const b = data[i + 2] ?? 0;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const sat = max === 0 ? 0 : (max - min) / max;
    const warmth = (r - b) / 255;
    const cyanBias = (g + b - 2 * r) / 255;
    // Keep vivid heat / teal attention pixels; drop fundus-like neutrals.
    const heatScore = Math.max(0, sat - 0.18) * 1.35 + Math.max(0, warmth) * 0.55 + Math.max(0, cyanBias) * 0.35;
    const alpha = Math.min(255, Math.round(heatScore * 220));
    data[i + 3] = alpha < 18 ? 0 : alpha;
  }

  ctx.putImageData(image, 0, 0);
  // Soften blockiness without inventing new hotspots.
  const soft = document.createElement("canvas");
  soft.width = size;
  soft.height = size;
  const softCtx = soft.getContext("2d");
  if (softCtx) {
    softCtx.filter = "blur(1.2px)";
    softCtx.drawImage(canvas, 0, 0);
    return soft;
  }
  return canvas;
}

interface VizFrameProps {
  label: string;
  src: string;
  alt: string;
  badge?: string;
}

function VizFrame({ label, src, alt, badge }: VizFrameProps) {
  return (
    <figure className="flex flex-col gap-3">
      <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-[0_40px_100px_-48px_rgb(0_0_0_/_0.85)]">
        <img
          src={src}
          width={1024}
          height={1024}
          loading="lazy"
          alt={alt}
          className="aspect-square w-full object-cover"
          draggable={false}
        />
        {badge ? (
          <span className="absolute left-4 top-4 rounded-full border border-white/15 bg-black/55 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-white/90 backdrop-blur-sm">
            {badge}
          </span>
        ) : null}
      </div>
      <figcaption className="text-center text-sm font-semibold text-night-foreground">
        {label}
      </figcaption>
    </figure>
  );
}

function GradCamOverlayDemo() {
  const sliderId = useId();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatLayerRef = useRef<HTMLCanvasElement | null>(null);
  const primaryRef = useRef<HTMLImageElement | null>(null);
  const [intensity, setIntensity] = useState(62);
  const [mode, setMode] = useState<OverlayMode>("overlay");
  const [ready, setReady] = useState(false);

  const effectiveIntensity = mode === "original" ? 0 : intensity;

  useEffect(() => {
    let cancelled = false;

    async function prepare() {
      try {
        const [primary, heatLayer] = await Promise.all([
          loadImage(fundusPrimary),
          buildHeatOnlyCanvas(fundusHeatmap),
        ]);
        if (cancelled) return;
        primaryRef.current = primary;
        heatLayerRef.current = heatLayer;
        setReady(true);
      } catch {
        if (!cancelled) setReady(false);
      }
    }

    void prepare();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const primary = primaryRef.current;
    const heatLayer = heatLayerRef.current;
    if (!canvas || !primary || !heatLayer || !ready) return;

    const size = 768;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(primary, 0, 0, size, size);

    const opacity = Math.min(1, Math.max(0, effectiveIntensity / 100));
    if (opacity > 0) {
      ctx.save();
      // Cap below 1 so retinal vessels remain readable at "100%".
      ctx.globalAlpha = 0.18 + opacity * 0.72;
      ctx.drawImage(heatLayer, 0, 0, size, size);
      ctx.restore();
    }
  }, [effectiveIntensity, ready]);

  return (
    <div className="mx-auto w-full max-w-3xl">
      <figure className="flex flex-col gap-4">
        <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-[0_50px_120px_-40px_rgb(0_0_0_/_0.8)]">
          {/* CSS fallback keeps layout stable while the canvas heat layer prepares */}
          <img
            src={fundusPrimary}
            width={1024}
            height={1024}
            loading="lazy"
            alt=""
            aria-hidden="true"
            className={`aspect-square w-full object-cover ${ready ? "invisible absolute inset-0" : ""}`}
            draggable={false}
          />
          {!ready ? (
            <img
              src={fundusHeatmap}
              width={1024}
              height={1024}
              loading="lazy"
              alt=""
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 h-full w-full object-cover"
              style={{ opacity: effectiveIntensity / 100 }}
              draggable={false}
            />
          ) : null}
          <canvas
            ref={canvasRef}
            className={`aspect-square h-auto w-full ${ready ? "" : "absolute inset-0 opacity-0"}`}
            role="img"
            aria-label="Illustrative Grad-CAM overlay on an example retinal fundus photograph (demonstration data)"
          />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/55 via-black/15 to-transparent px-5 pb-4 pt-16">
            <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-white/80">
              Grad-CAM Overlay · Illustrative
            </p>
          </div>
        </div>

        <figcaption className="space-y-2 text-center">
          <p className="text-sm font-semibold text-night-foreground">Grad-CAM Overlay</p>
          <p className="mx-auto max-w-xl text-sm leading-relaxed text-night-muted">
            Explore how a Grad-CAM-style heatmap can be superimposed on retinal imagery to visualize
            regions influencing an AI assessment.
          </p>
          <p className="mx-auto max-w-xl text-xs leading-relaxed text-night-muted/90">
            Illustrative visualization for demonstration purposes. Highlighted regions do not
            necessarily indicate retinal disease.
          </p>
        </figcaption>
      </figure>

      <div className="mt-6 space-y-4 rounded-2xl border border-white/10 bg-white/[0.04] p-4 backdrop-blur-sm sm:p-5">
        <div
          className="flex flex-wrap items-center justify-center gap-2"
          role="group"
          aria-label="Overlay display mode"
        >
          {(
            [
              { id: "original", label: "Original" },
              { id: "overlay", label: "Grad-CAM Overlay" },
            ] as const
          ).map((option) => {
            const active = mode === option.id;
            return (
              <button
                key={option.id}
                type="button"
                aria-pressed={active}
                onClick={() => setMode(option.id)}
                className={`rounded-full px-4 py-1.5 text-xs font-semibold tracking-wide transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color-mix(in_oklab,var(--color-primary)_70%,white)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-night)] ${
                  active
                    ? "bg-primary text-primary-foreground"
                    : "border border-white/15 bg-white/[0.04] text-night-muted hover:border-white/25 hover:text-night-foreground"
                }`}
              >
                {option.label}
              </button>
            );
          })}
        </div>

        <div className="mx-auto flex w-full max-w-md flex-col gap-2">
          <div className="flex items-center justify-between gap-3">
            <label htmlFor={sliderId} className="text-xs font-semibold text-night-foreground">
              Heatmap Intensity
            </label>
            <span className="tabular-nums text-xs font-medium text-night-muted" aria-live="polite">
              {effectiveIntensity}%
            </span>
          </div>
          <input
            id={sliderId}
            type="range"
            min={0}
            max={100}
            step={1}
            value={effectiveIntensity}
            onChange={(e) => {
              const next = Number(e.target.value);
              setIntensity(next);
              setMode(next === 0 ? "original" : "overlay");
            }}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={effectiveIntensity}
            aria-valuetext={
              effectiveIntensity === 0
                ? "Original image only"
                : `${effectiveIntensity} percent heatmap intensity`
            }
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[var(--color-primary)] [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-[var(--color-primary)] [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--color-primary)] [&::-webkit-slider-thumb]:shadow-[0_0_0_3px_rgb(255_255_255_/_0.12)]"
          />
          <div className="flex justify-between text-[10px] uppercase tracking-[0.12em] text-night-muted">
            <span>0% · Original</span>
            <span>100% · Strongest overlay</span>
          </div>
        </div>
      </div>
    </div>
  );
}

interface PanelProps {
  icon: LucideIcon;
  title: string;
  children: ReactNode;
}

function Panel({ icon: Icon, title, children }: PanelProps) {
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
      <div className="pointer-events-none absolute inset-0" aria-hidden="true">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgb(255_255_255_/_0.04)_1px,transparent_1px),linear-gradient(to_bottom,rgb(255_255_255_/_0.04)_1px,transparent_1px)] bg-[size:56px_56px] [mask-image:radial-gradient(ellipse_at_center,black_30%,transparent_75%)]" />
        <div className="absolute -left-40 top-10 h-96 w-96 rounded-full bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-primary)_20%,transparent),transparent_70%)] blur-3xl" />
        <div className="absolute -right-40 bottom-0 h-96 w-96 rounded-full bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-accent)_16%,transparent),transparent_70%)] blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div id="showcase-heading">
          <SectionHeading
            dark
            eyebrow="Interactive Showcase"
            title="SCAN & GRAD-CAM"
            description="Explore how AI-assisted visual explanations can help clinicians inspect retinal screening results."
          />
        </div>

        {/* Row 1 — existing original + standalone heatmap */}
        <div className="mx-auto mt-14 grid max-w-5xl gap-6 md:grid-cols-2 md:gap-8">
          <Reveal delay={0.05}>
            <VizFrame
              label="Original Retinal Image"
              src={fundusPrimary}
              alt="Example retinal fundus image (demonstration data)"
            />
          </Reveal>
          <Reveal delay={0.1}>
            <VizFrame
              label="Regions the AI Focused On"
              src={fundusHeatmap}
              alt="Simulated Grad-CAM heatmap highlighting illustrative regions of model attention (demonstration data)"
              badge="Illustrative"
            />
          </Reveal>
        </div>

        {/* Row 2 — integrated Grad-CAM overlay */}
        <Reveal delay={0.15} className="mt-10 sm:mt-12">
          <GradCamOverlayDemo />
        </Reveal>

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
                    Findings are reviewed by the clinician before a report is exported. The platform
                    supports—not replaces—professional judgment.
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
