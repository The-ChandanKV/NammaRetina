import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

// The Flask SPA is the supported app; old TanStack/UI templates are reference
// files, not build inputs, and require packages this application does not use.
export default tseslint.config(
  { ignores: ["node_modules/**", "src/components/ui/**", "src/routes/**", "src/router.tsx", "src/routeTree.gen.ts", "src/server.ts", "src/start.ts", "src/lib/error-*.ts", "src/hooks/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx,js,mjs}"],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    plugins: { "react-hooks": reactHooks },
    rules: { ...reactHooks.configs.recommended.rules },
  },
);
