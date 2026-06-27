"""Generate a PolicyBench-style Quarto HTML manuscript from LaTeX sources."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
WEB_QMD = PAPER_DIR / "web.qmd"


def expand_inputs(text: str) -> str:
    """Expand LaTeX input commands while leaving commented inputs untouched."""

    def expand_line(line: str) -> str:
        if line.lstrip().startswith("%"):
            return line

        def replace(match: re.Match[str]) -> str:
            input_path = match.group(1)
            path = PAPER_DIR / input_path
            if path.suffix == "":
                path = path.with_suffix(".tex")
            return expand_inputs(path.read_text())

        return re.sub(r"\\input\{([^}]+)\}", replace, line)

    return "\n".join(expand_line(line) for line in text.splitlines())


def normalize_for_pandoc(text: str) -> str:
    """Convert LaTeX constructs that Pandoc handles poorly for HTML."""

    def replace_subfigure(match: re.Match[str]) -> str:
        caption = match.group("caption")
        path = match.group("path")
        return (
            "\\includegraphics[width=0.48\\textwidth]{"
            + path
            + "}\n\n{\\small\\emph{"
            + caption
            + "}}\n"
        )

    return re.sub(
        r"\\subfigure\[(?P<caption>[^\]]+)\]\s*\{\s*"
        r"\\includegraphics(?:\[[^\]]+\])?\{(?P<path>[^}]+)\}\s*\}",
        replace_subfigure,
        text,
        flags=re.DOTALL,
    )


def pandoc_latex_to_markdown(latex: str, *, citeproc: bool = False) -> str:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tex", dir=PAPER_DIR, delete=False
    ) as source:
        source.write(latex)
        source_path = Path(source.name)

    try:
        command = [
            "quarto",
            "pandoc",
            str(source_path.name),
            "--from=latex",
            "--to=markdown",
            "--wrap=none",
        ]
        if citeproc:
            command.extend(["--citeproc", "--bibliography=references.bib"])
        result = subprocess.run(
            command,
            cwd=PAPER_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    finally:
        source_path.unlink(missing_ok=True)


def restore_empty_citation_spans(markdown: str) -> str:
    """Restore citation syntax that Pandoc emits as empty spans in raw HTML."""

    def replace(match: re.Match[str]) -> str:
        keys = match.group("keys").split()
        if len(keys) == 1:
            return f"@{keys[0]}"
        return "[" + "; ".join(f"@{key}" for key in keys) + "]"

    return re.sub(
        r'<span class="citation" data-cites="(?P<keys>[^"]+)"></span>',
        replace,
        markdown,
    )


def frontmatter_abstract() -> str:
    frontmatter = (PAPER_DIR / "frontmatter.tex").read_text()
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        frontmatter,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError("frontmatter.tex does not contain an abstract")
    return pandoc_latex_to_markdown(match.group(1).strip())


def body_markdown() -> str:
    latex = expand_inputs((PAPER_DIR / "body.tex").read_text())
    latex = normalize_for_pandoc(latex)
    body = pandoc_latex_to_markdown(latex)
    return restore_empty_citation_spans(body)


def main() -> None:
    abstract = frontmatter_abstract()
    body = body_markdown()
    WEB_QMD.write_text(
        f"""---
title: "A Firm-Level Microsimulation for VAT Policy Analysis"
subtitle: "An open firm-level model for UK VAT threshold reform"
author:
  - name: "Vahid Ahmadi"
    affiliation: "PolicyEngine"
bibliography: references.bib
citeproc: true
link-citations: true
format:
  html:
    toc: true
    toc-depth: 3
    theme: none
    css:
      - pe-tokens.css
      - firm-microsim-theme.css
---

::: {{.paper-abstract}}
**Abstract.**

{abstract}
:::

**Keywords:** value-added tax, microsimulation, tax notch, policy

{body}

## References

::: {{#refs}}
:::
"""
    )


if __name__ == "__main__":
    main()
