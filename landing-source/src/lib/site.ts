/**
 * Site-wide configuration for the NammaRetina landing page.
 *
 * The landing page is a public, static-facing entry point. In the final
 * Flask deployment the doctor sign-in route lives at `/login` (the existing
 * Flask authentication). Change `loginPath` here if the deployed route
 * differs — no other component hardcodes it.
 */
export const site = {
  name: "NammaRetina",
  loginPath: "/login",
  githubUrl: "https://github.com/The-ChandanKV/NammaRetina",
  sections: [
    { id: "platform", label: "Platform" },
    { id: "how-it-works", label: "How It Works" },
    { id: "technology", label: "Technology" },
    { id: "about", label: "About" },
  ],
} as const;
