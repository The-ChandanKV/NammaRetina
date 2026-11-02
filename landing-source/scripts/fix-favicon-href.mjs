import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const indexPath = path.join(root, "static/landing/index.html");
let html = readFileSync(indexPath, "utf8");
html = html.replaceAll("/landing/favicon.ico", "/favicon.ico");
writeFileSync(indexPath, html);
console.log("favicon href → /favicon.ico");
