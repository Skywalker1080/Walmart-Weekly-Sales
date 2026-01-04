import logging
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import project_logger


def test_setup_creates_log_file(tmp_path):
    log_dir = tmp_path / "logs"
    project_logger.setup_logging(log_dir=str(log_dir))

    logger = project_logger.get_logger("tests")
    logger.info("test message from test_setup_creates_log_file")

    # Flush handlers
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass

    log_file = log_dir / "app.log"
    assert log_file.exists()
    text = log_file.read_text(encoding="utf8")
    assert "test message from test_setup_creates_log_file" in text


def test_mlflow_filter_adds_run_id(monkeypatch):
    # Create a fake active run object
    class _FakeInfo:
        def __init__(self):
            self.run_id = "fake-run-id"
            self.experiment_id = "exp-123"

    class _FakeRun:
        def __init__(self):
            self.info = _FakeInfo()

    fake_run = _FakeRun()

    class FakeMLflow:
        @staticmethod
        def active_run():
            return fake_run

    monkeypatch.setitem("sys.modules", "mlflow", FakeMLflow())

    f = project_logger.MLflowRunFilter()
    record = logging.LogRecord(name="x", level=logging.INFO, pathname=__file__, lineno=1, msg="m", args=(), exc_info=None)
    f.filter(record)

    assert getattr(record, "mlflow_run_id") == "fake-run-id"
    assert getattr(record, "mlflow_experiment_id") == "exp-123"
