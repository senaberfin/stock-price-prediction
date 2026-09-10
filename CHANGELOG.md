# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows informal, date-based versioning (no strict SemVer,
since it's a learning/reference project rather than a published package).

## [Unreleased]

### Added
- Initial project scaffold: data pipeline, LSTM/GRU models, training loop,
  evaluation utilities.
- Synthetic sample dataset generator (`data/generate_sample_data.py`) so the
  notebook runs without external network access.
- Executed reference notebook (`notebooks/stock_price_prediction.ipynb`) with
  baked-in outputs and plots, including a naive-baseline comparison and
  residual analysis.
- Unit test suite (`tests/`) covering preprocessing, models, training, and
  evaluation (18 tests).
- GitHub Actions CI workflow running tests + flake8 on Python 3.10/3.11.
- `CONTRIBUTING.md`, `LICENSE` (MIT), `docs/METHODOLOGY.md`, and this
  changelog.

### Known limitations
- Included sample data is synthetically generated (Geometric Brownian
  Motion), not real market data — see the README for how to swap in a real
  dataset.
- No walk-forward / multi-fold validation yet — evaluation uses a single
  chronological 80/20 split.
- Single-feature model (Close price only); no volume, technical indicators,
  or exogenous variables.
