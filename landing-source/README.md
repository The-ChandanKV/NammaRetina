# NammaRetina landing page (source)

React + Vite + Tailwind source for the public marketing site.

## Develop

```bash
npm install
npm run dev
```

Opens a preview on http://localhost:5173. Doctor Sign In links navigate to `/login`
(expects the Flask app when integrated).

## Production build (for Flask)

```bash
npm install
npm run build
```

Emits compiled assets to `../static/landing/`, which Flask serves:

- `/` → `static/landing/index.html`
- `/landing/*` → JS, CSS, images, favicon

Do **not** put patient data under this folder.

## Notes

- Authentication is **not** implemented here — Flask owns `/login`, `/register`, sessions.
- Legacy TanStack Start preview routes under `src/routes/` are unused by the SPA build.
- Nested `.git` must never be committed inside this folder.
