interface LogoProps {
  className?: string;
}

/**
 * Custom eye/retina-inspired mark: an iris ring with a subtle vessel sweep,
 * drawn as pure SVG so it stays crisp at any size.
 */
export function Logo({ className = "h-8 w-8" }: LogoProps) {
  return (
    <svg
      viewBox="0 0 40 40"
      fill="none"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <linearGradient id="nr-logo-grad" x1="8" y1="6" x2="32" y2="34">
          <stop offset="0%" stopColor="var(--color-primary)" />
          <stop offset="100%" stopColor="var(--color-accent)" />
        </linearGradient>
      </defs>
      {/* outer iris ring */}
      <circle cx="20" cy="20" r="17" stroke="url(#nr-logo-grad)" strokeWidth="2.5" />
      {/* inner pupil */}
      <circle cx="20" cy="20" r="8.5" stroke="url(#nr-logo-grad)" strokeWidth="2" />
      {/* retinal vessel sweep */}
      <path
        d="M20 20c3.5-2.5 7-3 12-2M20 20c-2 3-5.5 4.5-10 4M20 20c1.5-4 4.5-6.5 9-7.5"
        stroke="var(--color-primary)"
        strokeWidth="1.4"
        strokeLinecap="round"
        opacity="0.75"
      />
      <circle cx="20" cy="20" r="2.2" fill="var(--color-primary)" />
    </svg>
  );
}
