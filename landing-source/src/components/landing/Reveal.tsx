import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface RevealProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  y?: number;
}

/**
 * Scroll-triggered entrance reveal. `MotionConfig reducedMotion="user"` in the
 * route root disables the transform for visitors who prefer reduced motion.
 */
export function Reveal({ children, className, delay = 0, y = 28 }: RevealProps) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-72px" }}
      transition={{ duration: 0.7, delay, ease: [0.21, 0.47, 0.32, 0.98] }}
    >
      {children}
    </motion.div>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  align = "center",
  dark = false,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  align?: "center" | "left";
  dark?: boolean;
}) {
  const alignment = align === "center" ? "mx-auto text-center items-center" : "text-left items-start";
  return (
    <div className={`flex max-w-2xl flex-col gap-4 ${alignment}`}>
      {eyebrow ? (
        <Reveal>
          <span className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/5 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
            {eyebrow}
          </span>
        </Reveal>
      ) : null}
      <Reveal delay={0.05}>
        <h2
          className={`text-balance text-3xl font-bold leading-tight tracking-tight sm:text-4xl lg:text-[2.75rem] ${
            dark ? "text-night-foreground" : "text-foreground"
          }`}
        >
          {title}
        </h2>
      </Reveal>
      {description ? (
        <Reveal delay={0.1}>
          <p className={`text-pretty text-base leading-relaxed sm:text-lg ${dark ? "text-night-muted" : "text-muted-foreground"}`}>
            {description}
          </p>
        </Reveal>
      ) : null}
    </div>
  );
}
