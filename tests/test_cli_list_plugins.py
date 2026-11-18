"""`eyas list` contract: plain listing knows the built-ins only (it reads no
config); ``--config`` additionally imports that config's ``plugins`` so plugin
registrations become visible — the claim the custom-method tutorial makes."""

from __future__ import annotations

from eyas.cli import main

_PLUGIN_SRC = """\
from eyas.engine.registry import register_strategy
from eyas.engine.strategy import Strategy


@register_strategy("_cli_list_probe")
class _Probe(Strategy):
    def init(self, seeds):
        raise NotImplementedError

    def step(self, state, view):
        raise NotImplementedError
"""


def test_list_shows_plugin_only_with_config(tmp_path, monkeypatch, capsys):
    (tmp_path / "_probe_plugin.py").write_text(_PLUGIN_SRC, encoding="utf-8")
    (tmp_path / "c.yaml").write_text("plugins: [_probe_plugin]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # bare list: no config is read, so the plugin is invisible. Must run
    # first — the registration below persists in-process for the whole session.
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "_cli_list_probe" not in out

    # --config: imports the config's plugins before the registry lookup
    assert main(["list", "--config", "c.yaml"]) == 0
    out = capsys.readouterr().out
    strategy_lines = [line for line in out.splitlines() if line.startswith("engine:strategy")]
    assert strategy_lines and "_cli_list_probe" in strategy_lines[0]
