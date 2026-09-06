.PHONY: sync check test lint paper etl reproduce reproduce-heavy figures-sync

sync:
	uv sync --extra dev --locked

test:
	uv run python -m pytest

lint:
	uv run ruff check .

check: test lint

paper:
	cd paper && TEXMFVAR=/tmp/firm-microsim-tex-cache latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex

# ---------------------------------------------------------------------------
# Reproduction (issue #40). `reproduce` regenerates both synthetic vintages and
# every checked results/*.txt and figure the paper reads, then syncs figures
# into paper/figures/. `reproduce-heavy` adds the two multi-build artifacts
# (seed sensitivity = two extra full builds; placebo B = one extra build).
# The PDF is rendered by the Paper workflow in CI (quarto is not required
# locally); see paper/README.md.
# ---------------------------------------------------------------------------
etl:
	uv run python scripts/etl_ons_tables.py

reproduce: etl
	uv run firm-microsim --seed 42
	uv run firm-microsim-static
	uv run python analysis/static_results_dump.py
	uv run firm-microsim-bunching --figures-only
	uv run python scripts/bunching_inference.py
	uv run firm-microsim-notch
	uv run firm-microsim-dynamic
	uv run firm-microsim-reform-menu
	uv run firm-microsim-dominated-region
	uv run firm-microsim-verify-optimum
	uv run firm-microsim-formulation-a-optima
	uv run python analysis/recovery_bunching.py
	$(MAKE) figures-sync
	uv run python scripts/claims.py --check

reproduce-heavy: reproduce
	uv run python scripts/seed_sensitivity.py --seeds 42 7 99
	uv run firm-microsim-placebo
	uv run python scripts/claims.py --check

figures-sync:
	@for f in paper/figures/*.png; do \
	  b=$$(basename $$f); \
	  if [ -f results/$$b ]; then cp results/$$b paper/figures/$$b; else echo "no results/$$b"; fi; \
	done
