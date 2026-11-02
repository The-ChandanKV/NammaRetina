import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/**
 * SPA build for Flask integration.
 * Assets are emitted to ../static/landing and served under /landing/* by Flask.
 * The compiled index.html is served at site root `/`.
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  base: "/landing/",
  publicDir: "public",
  build: {
    outDir: path.resolve(__dirname, "../static/landing"),
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
