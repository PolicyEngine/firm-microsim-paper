# Conference poster

A0 portrait poster for *A Firm-Level Microsimulation for VAT Policy Analysis*
(Vahid Ahmadi, PolicyEngine).

Built on the [Gemini](https://github.com/anishathalye/gemini) `beamerposter`
theme with a custom PolicyEngine colour theme.

## Build

```bash
xelatex poster.tex   # run twice
```

Requires **XeLaTeX** (for `fontspec`). The Raleway and Lato font files are
vendored in `fonts/` so no system font installation is needed.

## Files

- `poster.tex` — the poster source.
- `beamerthemegemini.sty` — Gemini theme (fonts repointed at `fonts/`).
- `beamercolorthemepolicyengine.sty` — PolicyEngine blue/teal colour theme.
- `fonts/` — vendored Raleway (`.otf`) and Lato (`.ttf`) faces.
- `figures/` — the five figures used, copied from `paper/figures/`.

## Notes

Figures and headline numbers follow the committed paper text
(`paper/Sections/`, common static base £183.6bn). The `fix-vat-liability-scaling`
branch regenerates results with a different base (~£184.65bn) and larger reform
costs; if those become the final numbers, update the reform-menu table and the
behavioural figures here to match.
