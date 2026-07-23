.PHONY: sync check test lint paper

sync:
	uv sync --extra dev --locked

test:
	uv run python -m pytest

lint:
	uv run ruff check .

check: test lint

paper:
	cd paper && TEXMFVAR=/tmp/firm-microsim-tex-cache latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex
