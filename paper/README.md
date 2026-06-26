# Paper

The manuscript is a Quarto paper project. The PDF entrypoint is `index.qmd`; it
renders the existing LaTeX section files through `quarto-template.tex` so the PDF
keeps the same journal-style formatting as `main.tex`.

Render with:

```bash
cd paper
quarto render index.qmd --to pdf
```

The rendered Quarto artifact is written under `paper/out/`. The checked-in
submission PDF remains `paper/main.pdf`.

The web paper follows the PolicyBench pattern: a generated Quarto HTML
manuscript, PolicyEngine design tokens, and a paper-specific theme. Render it
with:

```bash
cd paper
python3 build_site.py
```

`web.qmd` is generated from the LaTeX section files and ignored by git; the
source of truth remains the LaTeX manuscript. The local preview output is
`paper/site/index.html`; `paper/web.html` remains an intermediate Quarto output.
The Paper GitHub Actions workflow builds this site on PRs. The current Vercel
deployment is:

<https://firm-microsim-paper.vercel.app/>

For direct LaTeX debugging, `main.tex` is kept as a thin wrapper around the same
shared inputs:

```bash
cd paper
latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex
```
