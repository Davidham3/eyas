def test_version():
    import eyas

    assert eyas.__version__


def test_version_matches_installed_metadata():
    import importlib.metadata

    import eyas

    assert eyas.__version__ == importlib.metadata.version("eyas")


def test_builtin_components_registered():
    import eyas.builtin  # noqa: F401
    from eyas import registry

    assert "mock" in registry.registered("backend")
    assert "identity" in registry.registered("protocol")
    assert "exact_match" in registry.registered("judge")
    assert "demo_shapes" in registry.registered("task")


def test_cli_list(capsys):
    from eyas.cli import main

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "hill_climb" in out
