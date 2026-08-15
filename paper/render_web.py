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


def quarto_equation_id(label: str) -> str:
    """Return a Quarto-compatible equation id for a LaTeX label."""

    return label.replace(":", "-")


def normalize_equations(markdown: str) -> str:
    """Convert LaTeX equation environments to Quarto display equations."""

    def replace_equation(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        equation_id = quarto_equation_id(match.group("label"))
        # Quarto only registers the label when the attributed math is a
        # standalone block; with --wrap=none the surrounding prose keeps both
        # the opening and closing `$$` inside one paragraph, the label leaks
        # as literal page text, and every `@eq-...` reference renders as an
        # unresolved citation.
        return f"\n\n$$\n{body}\n$$ {{#{equation_id}}}\n\n"

    markdown = re.sub(
        r"\$\$\\begin\{equation\}\s*"
        r"(?P<body>.*?)"
        r"\s*\\label\{(?P<label>eq:[^}]+)\}\s*"
        r"\\end\{equation\}\s*\$\$",
        replace_equation,
        markdown,
        flags=re.DOTALL,
    )

    markdown = re.sub(
        r'\[\\\[(?P<label>eq:[^\]]+)\\\]\]\(#(?P=label)\)'
        r'\{reference-type="eqref" reference="(?P=label)"\}',
        lambda match: f"@{quarto_equation_id(match.group('label'))}",
        markdown,
    )
    return re.sub(
        r'<a href="#(?P<label>eq:[^"]+)" data-reference-type="eqref" '
        r'data-reference="(?P=label)">\[(?P=label)\]</a>',
        lambda match: f"@{quarto_equation_id(match.group('label'))}",
        markdown,
    )


def bibliography_keys() -> frozenset[str]:
    bibliography = (PAPER_DIR / "references.bib").read_text()
    return frozenset(
        re.findall(r"^@\w+\{([^,\s]+)\s*,", bibliography, flags=re.MULTILINE)
    )


def split_glued_citation_keys(markdown: str) -> str:
    r"""Detach hyphenated suffixes that Pandoc glued onto citation keys.

    ``\citet{liuetal2021}-normalised`` converts to ``@liuetal2021-normalised``,
    a single unknown citation key, because single internal hyphens are valid
    key characters. Rewrite any unknown key that extends a real bibliography
    key to the explicit-key form ``@{liuetal2021}-normalised`` so the suffix
    stays prose.
    """
    keys = bibliography_keys()

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in keys:
            return match.group(0)
        for key in sorted(keys, key=len, reverse=True):
            if token.startswith(key) and len(token) > len(key):
                return f"@{{{key}}}{token[len(key):]}"
        return match.group(0)

    return re.sub(r"@([A-Za-z][\w-]*)", replace, markdown)


def mathjax_safe_pounds(text: str) -> str:
    r"""Make LaTeX ``\pounds`` render under MathJax.

    Pandoc converts text-mode ``\pounds`` to a literal £, but any ``\pounds``
    left inside math (e.g. ``$-\pounds292$m``) is passed through to MathJax,
    which has no such command and prints a red literal ``\pounds``. MathJax
    does understand ``\unicode{xA3}``, which renders the £ glyph, so swap it in.
    Every remaining ``\pounds`` at this stage sits inside math, so a plain
    replacement is safe.
    """
    return text.replace(r"\pounds", r"\unicode{xA3}")


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
    body = restore_empty_citation_spans(body)
    body = split_glued_citation_keys(body)
    return normalize_equations(body)


def main() -> None:
    abstract = mathjax_safe_pounds(frontmatter_abstract())
    body = mathjax_safe_pounds(body_markdown())
    WEB_QMD.write_text(
        f"""---
title: "An Open Firm-Level Microsimulation for UK VAT Threshold Policy Analysis"
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
    html-math-method: mathjax
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
