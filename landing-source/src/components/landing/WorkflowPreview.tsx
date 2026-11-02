import {
  Database,
  FileDown,
  LayoutDashboard,
  ScanEye,
  UserRound,
  Users,
} from "lucide-react";

import { Reveal, SectionHeading } from "./Reveal";
import fundusHeatmap from "@/assets/fundus-heatmap.jpg";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard" },
  { icon: Users, label: "Patients", active: true },
  { icon: ScanEye, label: "Screening" },
  { icon: FileDown, label: "Reports" },
];

const history = [
  { date: "14 Mar 2025", severity: "No apparent DR", tone: "bg-emerald-500" },
  { date: "02 Aug 2025", severity: "Mild", tone: "bg-lime-500" },
  { date: "19 Jan 2026", severity: "Moderate", tone: "bg-amber-500" },
];

/** Illustrative severity-progression polyline (fictional demonstration data). */
function ProgressionChart() {
  return (
    <svg viewBox="0 0 260 120" className="h-auto w-full" role="img" aria-label="Illustrative chart showing recorded severity changing over three example examinations">
      <defs>
        <linearGradient id="nr-prog-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--color-primary)" stopOpacity="0.25" />
          <stop offset="100%" stopColor="var(--color-primary)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[20, 50, 80].map((y) => (
        <line key={y} x1="10" x2="250" y1={y} y2={y} stroke="var(--color-border)" strokeWidth="1" strokeDasharray="3 4" />
      ))}
      <path d="M30 88 L130 70 L230 40 L230 110 L130 110 L30 110 Z" fill="url(#nr-prog-fill)" />
      <polyline
        points="30,88 130,70 230,40"
        fill="none"
        stroke="var(--color-primary)"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      {[
        { x: 30, y: 88, tone: "fill-emerald-500" },
        { x: 130, y: 70, tone: "fill-lime-500" },
        { x: 230, y: 40, tone: "fill-amber-500" },
      ].map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="5" className={p.tone} stroke="white" strokeWidth="2" />
      ))}
      {["Mar 25", "Aug 25", "Jan 26"].map((label, i) => (
        <text key={label} x={30 + i * 100} y="118" textAnchor="middle" fontSize="9" fill="var(--color-muted-foreground)">
          {label}
        </text>
      ))}
    </svg>
  );
}

export function WorkflowPreview() {
  return (
    <section
      id="workflow-preview"
      className="py-20 sm:py-24 lg:py-28"
      aria-labelledby="workflow-heading"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Clinician Workflow"
          title="Designed Around Clinical Clarity."
          description="A conceptual look at how the existing NammaRetina workflow brings patient history, screening results, and explainability together."
        />

        <Reveal delay={0.1} className="mt-14">
          <div className="overflow-hidden rounded-3xl border border-border bg-card shadow-[0_40px_100px_-40px_rgb(7_20_33_/_0.35)]">
            {/* browser chrome */}
            <div className="flex items-center gap-3 border-b border-border bg-secondary/60 px-4 py-3">
              <div className="flex gap-1.5" aria-hidden="true">
                <span className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
              </div>
              <span className="mx-auto hidden rounded-md bg-background px-3 py-1 text-[11px] text-muted-foreground sm:block">
                nammaretina · patients / DEMO-0042
              </span>
            </div>

            <div className="grid md:grid-cols-[210px_1fr]">
              {/* sidebar */}
              <aside className="hidden border-r border-border bg-secondary/30 p-4 md:block">
                <div className="flex items-center gap-2 px-2 pb-4">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                    <UserRound className="h-4 w-4 text-primary" aria-hidden="true" />
                  </span>
                  <div>
                    <p className="text-xs font-semibold text-foreground">Dr. A. Rao</p>
                    <p className="text-[10px] text-muted-foreground">Retina specialist</p>
                  </div>
                </div>
                <nav aria-label="Workflow preview navigation">
                  <ul className="space-y-1">
                    {navItems.map((item) => (
                      <li
                        key={item.label}
                        className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-xs font-medium ${
                          item.active
                            ? "bg-primary/10 text-primary"
                            : "text-muted-foreground"
                        }`}
                      >
                        <item.icon className="h-3.5 w-3.5" aria-hidden="true" />
                        {item.label}
                      </li>
                    ))}
                  </ul>
                </nav>
                <div className="mt-6 flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2 text-[10px] text-muted-foreground">
                  <Database className="h-3.5 w-3.5" aria-hidden="true" />
                  Records stored securely
                </div>
              </aside>

              {/* main panel */}
              <div className="p-4 sm:p-6">
                {/* patient overview */}
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-background p-4">
                  <div className="flex items-center gap-3">
                    <span className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
                      D4
                    </span>
                    <div>
                      <p className="text-sm font-semibold text-foreground">Patient DEMO-0042</p>
                      <p className="text-[11px] text-muted-foreground">
                        Type 2 diabetes · 4 recorded examinations · Example record
                      </p>
                    </div>
                  </div>
                  <div className="pointer-events-none inline-flex items-center gap-2 rounded-full bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground">
                    <FileDown className="h-3.5 w-3.5" aria-hidden="true" />
                    Export PDF Report
                  </div>
                </div>

                <div className="mt-4 grid gap-4 lg:grid-cols-2">
                  {/* screening history */}
                  <div className="rounded-2xl border border-border bg-background p-4">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Retinal Screening History
                    </h3>
                    <ul className="mt-3 space-y-2">
                      {history.map((row) => (
                        <li
                          key={row.date}
                          className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5 text-xs"
                        >
                          <span className="text-muted-foreground">{row.date}</span>
                          <span className="flex items-center gap-2 font-medium text-foreground">
                            <span className={`h-2 w-2 rounded-full ${row.tone}`} aria-hidden="true" />
                            {row.severity}
                          </span>
                        </li>
                      ))}
                    </ul>
                    <p className="mt-3 text-[10px] text-muted-foreground">
                      Fictional demonstration data — not a real patient.
                    </p>
                  </div>

                  {/* progression chart */}
                  <div className="rounded-2xl border border-border bg-background p-4">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Severity Progression
                    </h3>
                    <div className="mt-3">
                      <ProgressionChart />
                    </div>
                  </div>

                  {/* heatmap preview */}
                  <div className="rounded-2xl border border-border bg-background p-4">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Explainability Heatmap
                    </h3>
                    <div className="mt-3 flex items-center gap-4">
                      <img
                        src={fundusHeatmap}
                        width={1024}
                        height={1024}
                        loading="lazy"
                        alt="Simulated Grad-CAM heatmap preview (demonstration data)"
                        className="h-24 w-24 rounded-xl border border-border object-cover"
                      />
                      <p className="text-[11px] leading-relaxed text-muted-foreground">
                        Visual explanation highlighting regions that influenced the AI-generated
                        assessment for the selected examination. Example imagery.
                      </p>
                    </div>
                  </div>

                  {/* report summary */}
                  <div className="rounded-2xl border border-border bg-background p-4">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Structured Report
                    </h3>
                    <ul className="mt-3 space-y-2 text-[11px] text-muted-foreground">
                      {[
                        "Screening result and severity level",
                        "Explanatory heatmap images",
                        "Recorded progression context",
                        "Clinician review notes",
                      ].map((line) => (
                        <li key={line} className="flex items-center gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden="true" />
                          {line}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Reveal>

        <Reveal delay={0.15}>
          <p className="mt-6 text-center text-xs font-medium text-muted-foreground">
            Conceptual workflow preview — demonstration data only. Not connected to real patient
            records.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
