"""``eyas prepare`` subcommand contract tests (zero network, zero downloads).

The subcommand is a pure forwarder: the dataset-preparation parser and logic live
in ``eyas.prepare`` (moved from ``scripts/prepare_quickstart_data.py`` so the
capability ships with the package, not only with a repo checkout). These tests
pin the forwarding, the missing-extra guidance, and the historical script shim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from eyas import cli


def test_prepare_forwards_the_whole_argv_tail(monkeypatch):
    import eyas.prepare as prepare_mod

    seen = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(prepare_mod, "main", fake_main)
    rc = cli.main(["prepare", "--dataset", "textvqa", "--source", "auto"])
    assert rc == 0
    assert seen["argv"] == ["--dataset", "textvqa", "--source", "auto"]


def test_prepare_reports_missing_extra_with_install_hint(monkeypatch, capsys):
    import eyas.prepare as prepare_mod

    def boom(argv):
        raise ModuleNotFoundError("No module named 'datasets'", name="datasets")

    monkeypatch.setattr(prepare_mod, "main", boom)
    rc = cli.main(["prepare", "--dataset", "textvqa"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "datasets" in err
    assert "eyas[prepare]" in err


def test_prepare_appears_in_top_level_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "prepare" in capsys.readouterr().out


def test_prepare_main_parser_requires_dataset(monkeypatch):
    # real parser (no monkeypatched main): an unknown/absent dataset must fail fast
    with pytest.raises((SystemExit, ValueError)):
        cli.main(["prepare", "--dataset", "no-such-dataset"])


def test_historical_script_shim_delegates_to_package():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_quickstart_data.py"
    spec = importlib.util.spec_from_file_location("prepare_quickstart_shim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is sys.modules["eyas.prepare"].main
