import { createFileRoute } from "@tanstack/react-router";
import { ArrowLeft, ShieldCheck } from "lucide-react";
import { Link } from "@tanstack/react-router";

import { Logo } from "@/components/landing/Logo";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Doctor Sign In — NammaRetina" },
      {
        name: "description",
        content:
          "Doctor sign-in for the NammaRetina platform. Authentication is provided by the deployed NammaRetina application.",
      },
      { property: "og:title", content: "Doctor Sign In — NammaRetina" },
      {
        property: "og:description",
        content:
          "Doctor sign-in for the NammaRetina platform. Authentication is provided by the deployed NammaRetina application.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: LoginPage,
});

/**
 * The landing page intentionally does NOT implement its own authentication.
 * In the final Flask deployment, this path is served by the existing Flask
 * authentication flow; this page only exists so the standalone preview has a
 * sane destination instead of a 404.
 */
function LoginPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-md rounded-3xl border border-border bg-card p-8 text-center shadow-[0_40px_100px_-40px_rgb(7_20_33_/_0.3)]">
        <div className="flex justify-center">
          <Logo className="h-10 w-10" />
        </div>
        <h1 className="mt-5 text-2xl font-bold tracking-tight text-foreground">Doctor Sign In</h1>
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
          Sign-in is handled by the deployed NammaRetina application. This preview hosts the public
          landing page only—no new accounts or credentials are created here.
        </p>
        <p className="mt-4 inline-flex items-center gap-2 rounded-full border border-border bg-muted px-3.5 py-1.5 text-xs font-medium text-muted-foreground">
          <ShieldCheck className="h-3.5 w-3.5 text-primary" aria-hidden="true" />
          Access is intended for authorized clinical users.
        </p>
        <div className="mt-7">
          <Link
            to="/"
            className="inline-flex items-center gap-2 rounded-full bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Back to the landing page
          </Link>
        </div>
      </div>
    </div>
  );
}
