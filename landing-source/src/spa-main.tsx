import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MotionConfig } from "framer-motion";

import "./styles.css";
import { LandingApp } from "./LandingApp";

const root = document.getElementById("root");
if (!root) {
  throw new Error("Root element #root not found");
}

createRoot(root).render(
  <StrictMode>
    <MotionConfig reducedMotion="user">
      <LandingApp />
    </MotionConfig>
  </StrictMode>,
);
