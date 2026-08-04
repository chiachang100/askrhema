# AskRhema Useful Commands


- Fix Ruff issues automatically
  - `uv run ruff check . --fix`
  - `uv run ruff format .`
  - `uv run ruff check .`

- Run MyPy:
  - `uv run mypy .`

- Auto generate `requirements.txt` file
  - `uv export --format requirements.txt --no-dev --output-file requirements.txt`

---
