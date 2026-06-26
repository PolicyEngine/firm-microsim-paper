"""Build the static web-paper site for deployment."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import render_web


PAPER_DIR = Path(__file__).resolve().parent
SITE_DIR = PAPER_DIR / "site"
HTML_CANDIDATES = [
    PAPER_DIR / "web.html",
    PAPER_DIR / "out" / "web.html",
    PAPER_DIR / "out" / "web-preview.html",
]
ASSET_DIR_NAMES = ["web_files", "web-preview_files"]


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def find_rendered_html() -> Path:
    for path in HTML_CANDIDATES:
        if path.exists():
            return path
    candidates = ", ".join(str(path.relative_to(PAPER_DIR)) for path in HTML_CANDIDATES)
    raise FileNotFoundError(f"Quarto did not write one of: {candidates}")


def main() -> None:
    render_web.main()
    for path in HTML_CANDIDATES:
        remove_path(path)
    for name in ASSET_DIR_NAMES:
        remove_path(PAPER_DIR / name)
        remove_path(PAPER_DIR / "out" / name)

    subprocess.run(
        ["quarto", "render", "web.qmd", "--to", "html"],
        cwd=PAPER_DIR,
        check=True,
    )

    html_path = find_rendered_html()
    html_dir = html_path.parent

    shutil.rmtree(SITE_DIR, ignore_errors=True)
    SITE_DIR.mkdir()
    (SITE_DIR / ".nojekyll").write_text("")

    copy_path(html_path, SITE_DIR / "index.html")
    for name in ["pe-tokens.css", "firm-microsim-theme.css"]:
        source = html_dir / name
        if not source.exists():
            source = PAPER_DIR / name
        copy_path(source, SITE_DIR / name)

    for name in ASSET_DIR_NAMES:
        source = html_dir / name
        if source.exists():
            copy_path(source, SITE_DIR / name)
    copy_path(PAPER_DIR / "figures", SITE_DIR / "figures")


if __name__ == "__main__":
    main()
