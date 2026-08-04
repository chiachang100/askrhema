# AskRhema Useful Commands

- Clean the directory
  - `pyclean . --debris`

- Test
  - `uv run pytest`

- Fix Ruff issues automatically
  - `uv run ruff check . --fix`
  - `uv run ruff format .`
  - `uv run ruff check .`

- Run MyPy:
  - `uv run mypy .`

- Auto generate `requirements.txt` file
  - `uv export --format requirements.txt --no-dev --output-file requirements.txt`

---
