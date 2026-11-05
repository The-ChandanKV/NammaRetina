# NammaRetina landing page (source)

React + Vite + Tailwind source for the public marketing site.

## Develop

```bash
npm ci
npm run dev
```

Opens a preview on http://localhost:5173. Doctor Sign In links navigate to `/signin`
(expects the Flask app when integrated).

## Production build (for Flask)

```bash
npm ci
npm run build
```

Emits compiled assets to `../static/landing/`, which Flask serves:

- `/` → `static/landing/index.html`
- `/landing/*` → JS, CSS, images, favicon

Do **not** put patient data under this folder.

## Notes

- Authentication is **not** implemented here — Flask owns `/signin`, `/register`, sessions.
- Legacy TanStack Start preview routes under `src/routes/` are unused by the SPA build.
- Nested `.git` must never be committed inside this folder.


Use a supported Node release: 22.13+, 24.x, or 26+. Run `npm run test`, `npm run lint`, and `npm run typecheck`.
The production build also runs the TypeScript check. The checks cover `spa-main.tsx`,
`LandingApp.tsx`, the landing components, and their imported modules. Unused TanStack
Start routes, server scaffolding, and stock UI templates are reference files outside
this app's build/type/lint scope; they require their own dependencies before reuse.
Vitest runs in Node and renders the real SPA without unused router or jsdom packages.

The minimatch override selects a patched version for the development tool graph.
Keep the lockfile committed and validate all checks when upgrading dependencies.
