"""Build the static web-paper site for deployment."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import render_web


PAPER_DIR = Path(__file__).resolve().parent
BUILD_DIR = PAPER_DIR / "_webbuild"
SITE_DIR = PAPER_DIR / "site"
CSS_NAMES = ["pe-tokens.css", "firm-microsim-theme.css"]


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    render_web.main()

    # Render in an isolated directory: paper/_quarto.yml declares a manuscript
    # project whose article is index.qmd, so rendering web.qmd in paper/ makes
    # Quarto treat it as an embedded notebook and skip the crossref filter —
    # equation labels leak as literal text and every `@eq-...` reference
    # renders as an unresolved citation.
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir()
    copy_path(render_web.WEB_QMD, BUILD_DIR / "web.qmd")
    for name in ["references.bib", *CSS_NAMES]:
        copy_path(PAPER_DIR / name, BUILD_DIR / name)
    copy_path(PAPER_DIR / "figures", BUILD_DIR / "figures")

    subprocess.run(
        ["quarto", "render", "web.qmd", "--to", "html"],
        cwd=BUILD_DIR,
        check=True,
    )

    html_path = BUILD_DIR / "web.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Quarto did not write {html_path}")

    shutil.rmtree(SITE_DIR, ignore_errors=True)
    SITE_DIR.mkdir()
    (SITE_DIR / ".nojekyll").write_text("")

    copy_path(html_path, SITE_DIR / "index.html")
    for name in CSS_NAMES:
        copy_path(BUILD_DIR / name, SITE_DIR / name)

    # Quarto emits its stylesheets and scripts into a sibling `web_files/`
    # directory that the rendered HTML links to by relative path. Without it
    # the page loads unstyled, so it must ship too.
    copy_path(BUILD_DIR / "web_files", SITE_DIR / "web_files")

    copy_path(PAPER_DIR / "figures", SITE_DIR / "figures")

    shutil.rmtree(BUILD_DIR)


if __name__ == "__main__":
    main()
