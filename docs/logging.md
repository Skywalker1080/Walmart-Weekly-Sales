# Logging (Project)

✅ **Purpose:** Provide a compact, consistent logging setup for scripts, training jobs, FastAPI apps, and CI runs.

## Quick start

1. Configure and initialize logging at your entrypoint (CLI script, `main` or FastAPI startup):

```py
from project_logger import setup_logging, get_logger

setup_logging()  # creates `logs/app.log` by default
logger = get_logger(__name__)
logger.info("starting job")
```

2. Use `get_logger(__name__)` in modules.

## Features

- Rotating file logs at `logs/app.log` (10 MB default, 5 backups). 🔄
- Console logs for local visibility. 💬
- Optional JSON logs if `python-json-logger` is installed. 🧾
- MLflow-aware filter that attaches `mlflow_run_id` when an active run exists. 🔬

## FastAPI

Call `setup_logging()` inside a FastAPI startup event to ensure uvicorn logs propagate to the same handlers.

## Tests

`tests/test_logger.py` contains basic unit tests verifying file creation and the MLflow filter.
