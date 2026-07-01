# VAT microsimulation presentation

Conference slide app for the IMA World Congress 2026 talk on
*A Firm-Level Microsimulation for VAT Policy Analysis*.

Built on the PolicyEngine slideshow template (Next.js 16 / React 19 / Tailwind v4,
KaTeX for math). Slides are React components; a client-side viewer swaps between
them via keyboard (`→`/`←`/`Space`, `Home`/`End`, `f` for fullscreen) or click.

## Commands

Install dependencies from this directory:

```bash
npm install
```

Run the local deck:

```bash
npm run dev
```

Check the app:

```bash
npm run typecheck
npm run lint
npm run build
```

Export the deck to PDF (screenshots each slide via Playwright):

```bash
npm run export:pdf            # one-time: npm i -D playwright pdf-lib && npx playwright install chromium
```

The deck is defined in `slides/config.ts` (order + metadata) and
`slides/vat-ima-2026.tsx` (the slides). Figures live in `public/figures/`.
