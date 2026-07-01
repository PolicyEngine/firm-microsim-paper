// Export the slide deck to a single PDF — one page per slide.
//
// The deck is a Next.js/React slideshow (not reveal.js), so there is no built-in
// export. This drives a headless Chromium over each `?slide=N` view, screenshots
// it at 1080p (2x for crisp text), and assembles the shots into one PDF at the
// standard 16:9 slide size. Screenshots (rather than print-to-PDF) guarantee the
// output matches the on-screen render exactly — KaTeX, gradients, custom layout.
//
// One-time setup:
//   npm i -D playwright pdf-lib && npx playwright install chromium
//
// Run:
//   npm run export:pdf                 # build, serve, export -> slides.pdf
//   SKIP_BUILD=1 npm run export:pdf    # reuse an existing .next build
//   SLIDE_COUNT=28 OUT=deck.pdf npm run export:pdf
//
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { PDFDocument } from "pdf-lib";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PORT = Number(process.env.PORT || 3210);
const BASE = `http://localhost:${PORT}`;
const VIEW_W = Number(process.env.SLIDE_WIDTH || 1920);
const VIEW_H = Number(process.env.SLIDE_HEIGHT || 1080);
const PAGE_W = 960; // PDF points: standard 16:9 slide (13.3in)
const PAGE_H = 540;
const OUT = resolve(root, process.env.OUT || "slides.pdf");

// Number of slides: the `slides: [ ... ]` array in slides/config.ts, or SLIDE_COUNT.
function slideCount() {
  if (process.env.SLIDE_COUNT) return Number(process.env.SLIDE_COUNT);
  const cfg = readFileSync(resolve(root, "slides/config.ts"), "utf8");
  const m = cfg.match(/slides:\s*\[([\s\S]*?)\]/);
  if (!m) throw new Error("Could not find slides array in slides/config.ts; set SLIDE_COUNT.");
  const names = m[1]
    .split(",")
    .map((s) => s.replace(/\/\/.*$/gm, "").trim())
    .filter((s) => /^[A-Za-z]/.test(s));
  return names.length;
}

function run(cmd, args) {
  return new Promise((res, rej) => {
    const p = spawn(cmd, args, { cwd: root, stdio: "inherit" });
    p.on("exit", (c) => (c === 0 ? res() : rej(new Error(`${cmd} ${args.join(" ")} -> exit ${c}`))));
  });
}

async function waitForServer(url, timeoutMs = 90000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      // server not up yet
    }
    await sleep(500);
  }
  throw new Error(`Server not ready at ${url} after ${timeoutMs}ms`);
}

async function main() {
  const count = slideCount();
  console.log(`Exporting ${count} slides -> ${OUT}`);

  if (!process.env.SKIP_BUILD) {
    console.log("Building (next build)…");
    await run("npm", ["run", "build"]);
  }

  console.log(`Starting server on :${PORT}…`);
  const server = spawn("npx", ["next", "start", "-p", String(PORT)], {
    cwd: root,
    stdio: "inherit",
  });

  try {
    await waitForServer(BASE);
    const browser = await chromium.launch();
    const page = await browser.newPage({
      viewport: { width: VIEW_W, height: VIEW_H },
      deviceScaleFactor: 2,
    });

    const pdf = await PDFDocument.create();
    for (let i = 0; i < count; i++) {
      await page.goto(`${BASE}/?slide=${i}&export=1`, { waitUntil: "networkidle" });
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(400); // settle KaTeX / transitions
      const shot = await page.screenshot({
        type: "png",
        clip: { x: 0, y: 0, width: VIEW_W, height: VIEW_H },
      });
      const png = await pdf.embedPng(shot);
      const pg = pdf.addPage([PAGE_W, PAGE_H]);
      pg.drawImage(png, { x: 0, y: 0, width: PAGE_W, height: PAGE_H });
      console.log(`  slide ${i + 1}/${count}`);
    }

    await browser.close();
    writeFileSync(OUT, await pdf.save());
    console.log(`Wrote ${OUT}`);
  } finally {
    server.kill("SIGTERM");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
