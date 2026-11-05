import { useRef } from "react";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, ScanLine, Sparkles, Activity } from "lucide-react";

import fundusHero from "@/assets/fundus-secondary.jpg";
import { DoctorSignInLink } from "./DoctorSignInLink";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
};

const item = {
  hidden: { opacity: 0, y: 26 },
  show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.21, 0.47, 0.32, 0.98] as const } },
};

/** Small labeled data-point chip floating over the retinal composition. */
function FloatingTag({
  className,
  children,
  delay,
}: {
  className: string;
  children: React.ReactNode;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ delay, duration: 0.6, ease: "easeOut" }}
      className={`absolute ${className}`}
    >
      <motion.div
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay }}
        className="flex items-center gap-1.5 rounded-full border border-border/70 bg-background/85 px-3 py-1.5 text-[11px] font-semibold tracking-wide text-foreground shadow-[0_10px_30px_-12px_rgb(11_22_40_/_0.4)] backdrop-blur-md"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden="true" />
        {children}
      </motion.div>
    </motion.div>
  );
}

export function Hero() {
  const visualRef = useRef<HTMLDivElement>(null);
  const reduceMotion = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: visualRef,
    offset: ["start end", "end start"],
  });
  const visualY = useTransform(scrollYProgress, [0, 1], [reduceMotion ? 0 : 24, reduceMotion ? 0 : -24]);

  return (
    <section className="relative overflow-hidden pt-28 sm:pt-32 lg:pt-36" aria-labelledby="hero-heading">
      {/* ambient background */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        aria-hidden="true"
      >
        <div className="absolute -top-40 right-[-10%] h-[34rem] w-[34rem] rounded-full bg-[radial-gradient(circle_at_center,color-mix(in_oklab,var(--color-primary)_14%,transparent),transparent_65%)] blur-2xl" />
        <div className="absolute bottom-[-30%] left-[-12%] h-[30rem] w-[30rem] rounded-full bg-[radial-gradient(circle_at_center,color-mix(in_oklab,var(--color-accent)_10%,transparent),transparent_65%)] blur-2xl" />
        {/* fine grid lines */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,color-mix(in_oklab,var(--color-foreground)_4%,transparent)_1px,transparent_1px),linear-gradient(to_bottom,color-mix(in_oklab,var(--color-foreground)_4%,transparent)_1px,transparent_1px)] bg-[size:56px_56px] [mask-image:radial-gradient(ellipse_at_top,black_20%,transparent_70%)]" />
      </div>

      <div className="mx-auto grid max-w-7xl items-center gap-14 px-4 pb-20 sm:px-6 lg:grid-cols-[1.05fr_1fr] lg:gap-10 lg:pb-28 lg:px-8">
        {/* Copy */}
        <motion.div variants={container} initial="hidden" animate="show" className="max-w-xl">
          <motion.p
            variants={item}
            className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/5 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-primary"
          >
            <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
            Intelligent Retinal Care
          </motion.p>

          <motion.h1
            variants={item}
            id="hero-heading"
            className="mt-6 text-balance text-4xl font-extrabold leading-[1.08] tracking-tight text-foreground sm:text-5xl lg:text-[3.4rem]"
          >
            See Beyond the Scan.
            <br />
            Understand the{" "}
            <span className="text-gradient-teal">Progression.</span>
          </motion.h1>

          <motion.p variants={item} className="mt-6 text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg">
            An AI-assisted retinal screening platform helping clinicians assess diabetic
            retinopathy severity, visualize model insights, monitor patient progression,
            and generate structured reports—all in one workflow.
          </motion.p>

          <motion.div variants={item} className="mt-9 flex flex-wrap items-center gap-4">
            <DoctorSignInLink
              className="group inline-flex items-center gap-2 rounded-full bg-primary px-7 py-3.5 text-sm font-semibold text-primary-foreground shadow-[0_18px_40px_-16px_color-mix(in_oklab,var(--color-primary)_70%,transparent)] transition-all hover:bg-primary/90 hover:shadow-[0_22px_48px_-16px_color-mix(in_oklab,var(--color-primary)_80%,transparent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            >
              Sign In
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" aria-hidden="true" />
            </DoctorSignInLink>
            <a
              href="#platform"
              className="inline-flex items-center gap-2 rounded-full border border-border bg-background px-7 py-3.5 text-sm font-semibold text-foreground transition-colors hover:border-primary/40 hover:bg-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            >
              Explore the Platform
            </a>
          </motion.div>

          <motion.p variants={item} className="mt-8 text-xs leading-relaxed text-muted-foreground">
            Built for ophthalmologists, retina specialists, and authorized clinicians.
            An academic, AI-assisted decision-support project.
          </motion.p>
        </motion.div>

        {/* Visual composition */}
        <motion.div ref={visualRef} style={{ y: visualY }} className="relative mx-auto w-full max-w-[520px]">
          <motion.div
            initial={{ opacity: 0, scale: 0.94 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.9, ease: "easeOut", delay: 0.15 }}
            className="relative aspect-square"
          >
            {/* glow */}
            <div
              className="absolute inset-6 rounded-full bg-[radial-gradient(circle_at_35%_30%,color-mix(in_oklab,var(--color-primary)_22%,transparent),transparent_70%)] blur-2xl"
              aria-hidden="true"
            />

            {/* rotating dashed ring */}
            <motion.svg
              viewBox="0 0 100 100"
              className="absolute -inset-3 h-auto w-auto"
              aria-hidden="true"
              animate={reduceMotion ? {} : { rotate: 360 }}
              transition={{ duration: 90, repeat: Infinity, ease: "linear" }}
            >
              <circle
                cx="50"
                cy="50"
                r="48.5"
                fill="none"
                stroke="var(--color-primary)"
                strokeOpacity="0.35"
                strokeWidth="0.3"
                strokeDasharray="2 3"
              />
            </motion.svg>

            {/* retinal image */}
            <div className="absolute inset-0 overflow-hidden rounded-full border border-border/60 shadow-[0_40px_90px_-30px_rgb(7_20_33_/_0.55)]">
              <img
                src={fundusHero}
                width={1024}
                height={1024}
                alt="Example retinal fundus image used for demonstration"
                className="h-full w-full object-cover"
              />
              {/* subtle aqua tint at rim */}
              <div className="absolute inset-0 rounded-full bg-[radial-gradient(circle_at_center,transparent_58%,color-mix(in_oklab,var(--color-accent)_18%,transparent)_92%)]" aria-hidden="true" />

              {/* animated scan line */}
              {!reduceMotion ? (
                <motion.div
                  className="absolute inset-x-6"
                  aria-hidden="true"
                  initial={{ top: "12%" }}
                  animate={{ top: ["12%", "86%", "12%"] }}
                  transition={{ duration: 7, repeat: Infinity, ease: "easeInOut", repeatDelay: 0.6 }}
                >
                  <div className="relative h-px w-full bg-gradient-to-r from-transparent via-[var(--color-accent)] to-transparent" />
                  <div className="mx-auto h-10 w-[85%] -translate-y-1/2 bg-[linear-gradient(to_bottom,color-mix(in_oklab,var(--color-accent)_18%,transparent),transparent)]" />
                </motion.div>
              ) : null}
            </div>

            {/* anatomical data points */}
            <svg viewBox="0 0 100 100" className="absolute inset-0" aria-hidden="true">
              {[
                { x: 30, y: 34 },
                { x: 66, y: 26 },
                { x: 74, y: 62 },
                { x: 34, y: 72 },
              ].map((p, i) => (
                <g key={i}>
                  <circle cx={p.x} cy={p.y} r="0.9" fill="var(--color-accent)">
                    <animate
                      attributeName="opacity"
                      values="1;0.3;1"
                      dur={`${2.4 + i * 0.5}s`}
                      repeatCount="indefinite"
                    />
                  </circle>
                  <circle cx={p.x} cy={p.y} r="2.6" fill="none" stroke="var(--color-accent)" strokeOpacity="0.35" strokeWidth="0.25" />
                </g>
              ))}
            </svg>

            {/* floating labels */}
            <FloatingTag className="right-[-2%] top-[10%]" delay={0.7}>
              Retinal Imaging
            </FloatingTag>
            <FloatingTag className="left-[-4%] top-[46%]" delay={0.95}>
              <ScanLine className="h-3 w-3 text-primary" aria-hidden="true" />
              Explainable AI
            </FloatingTag>
            <FloatingTag className="bottom-[12%] right-[2%]" delay={1.2}>
              <Activity className="h-3 w-3 text-primary" aria-hidden="true" />
              Severity Assessment
            </FloatingTag>

            {/* floating analysis panel */}
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.1, duration: 0.7, ease: "easeOut" }}
              className="absolute -bottom-6 left-1/2 w-[78%] -translate-x-1/2 sm:-bottom-10 sm:left-auto sm:right-[-4%] sm:w-[64%] sm:translate-x-0"
            >
              <motion.div
                animate={reduceMotion ? {} : { y: [0, -7, 0] }}
                transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                className="glow-ring rounded-2xl border border-border/70 bg-background/90 p-4 backdrop-blur-xl sm:p-5"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                      <ScanLine className="h-4 w-4 text-primary" aria-hidden="true" />
                    </span>
                    <p className="text-sm font-semibold text-foreground">Retinal Analysis</p>
                  </div>
                  <span className="rounded-full border border-border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Demo data
                  </span>
                </div>
                <div className="mt-4 space-y-2.5" aria-hidden="true">
                  {[
                    { label: "No apparent DR", w: "18%", active: false },
                    { label: "Mild", w: "34%", active: false },
                    { label: "Moderate", w: "72%", active: true },
                    { label: "Severe / Proliferative", w: "12%", active: false },
                  ].map((row) => (
                    <div key={row.label} className="flex items-center gap-3">
                      <span className="w-32 shrink-0 text-[11px] font-medium text-muted-foreground">
                        {row.label}
                      </span>
                      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                        <motion.span
                          className={`block h-full rounded-full ${row.active ? "bg-primary" : "bg-muted-foreground/30"}`}
                          initial={{ width: 0 }}
                          animate={{ width: row.w }}
                          transition={{ delay: 1.5, duration: 0.9, ease: "easeOut" }}
                        />
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground">
                  Illustrative demonstration of the analysis interface—not an actual patient result.
                </p>
              </motion.div>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
