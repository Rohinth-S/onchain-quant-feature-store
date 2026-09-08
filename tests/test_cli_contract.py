import json
import subprocess
import sys

from feature_store import __version__


def _invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "feature_store.cli", *map(str, args)], capture_output=True, text=True, timeout=45,
    )


def test_cli_version_and_report(tmp_path):
    version = _invoke("--version")
    assert version.returncode == 0, version.stderr
    assert __version__ in version.stdout

    report = tmp_path / "reports" / "demo.json"
    run = _invoke("--output", report)
    assert run.returncode == 0, run.stderr
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["producer"]["version"] == __version__
    assert value["producer"]["format_version"] == 1
    assert list(report.parent.glob("*.tmp")) == []


def test_cli_refuses_to_overwrite_input(tmp_path):
    original = tmp_path / "capture.json"
    original.write_text("keep this capture", encoding="utf-8")
    rejected = _invoke("--input", original, "--output", original)
    assert rejected.returncode == 2
    assert "distinct paths" in rejected.stderr
    assert original.read_text(encoding="utf-8") == "keep this capture"
