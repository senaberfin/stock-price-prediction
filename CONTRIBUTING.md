# Contributing

Thanks for considering a contribution! This is a learning/reference project,
so contributions that improve clarity, correctness, or extend the model
comparison are especially welcome.

## Ways to contribute

- **Bug fixes** — incorrect shapes, broken plots, data-leakage in the
  train/test split, etc.
- **New features** — additional input features (volume, technical
  indicators), new model architectures (e.g. Transformer, TCN), walk-forward
  validation, a simple backtest of a trading strategy.
- **Documentation** — clarifying the README, adding docstring examples,
  fixing typos.
- **Tests** — increasing coverage in `tests/`, especially edge cases.

## Development setup

```bash
git clone <your-fork-url>
cd stock-price-prediction
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Before opening a pull request

1. **Run the test suite:**
   ```bash
   pytest tests/ -v
   ```
2. **Lint your changes:**
   ```bash
   flake8 src/ tests/ --max-line-length=110 --extend-ignore=E203
   ```
3. **Re-run the notebook top to bottom** if you touched `src/` or the data
   pipeline, so committed notebook outputs stay in sync with the code:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/stock_price_prediction.ipynb
   ```
4. Keep pull requests focused — one logical change per PR is easier to
   review than a bundle of unrelated edits.

## Code style

- Follow PEP 8 (enforced via `flake8`, max line length 110).
- Add a docstring to any new public function or class.
- Prefer small, composable functions in `src/` over notebook-only logic, so
  behavior can be unit tested.

## Reporting issues

Please include:
- Python version and OS
- Full traceback (if applicable)
- Steps to reproduce, ideally a minimal snippet

## Code of Conduct

Be respectful and constructive. Assume good faith. This project has no
formal enforcement process beyond maintainer discretion, but harassment or
abusive behavior will not be tolerated in issues or pull requests.
