from __future__ import annotations

import json
from pathlib import Path

import pytest

from cdd import __version__
from cdd.cli import main


def test_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema"]) == 0
    assert "span" in json.loads(capsys.readouterr().out)["properties"]


def test_generate(tmp_path: Path) -> None:
    config = tmp_path / "bench.yaml"
    config.write_text("name: cli\nn_streams: 2\nseed: 1\n", encoding="utf-8")
    assert main(["generate", "--config", str(config), "--out", str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["cli-000.json", "cli-001.json"]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out
