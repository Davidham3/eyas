"""Fast tests for the runtime config utilities (``eyas.runtime.config``): the
``plugins`` key imports and registers external plugin modules — shape validation
fails fast, CWD joins sys.path, and real modules can register through it; plus
the ``--set`` dotted-path override semantics. Zero GPU."""

from __future__ import annotations

import pytest
import yaml

from eyas.runtime.config import apply_overrides, load_plugins


def test_plugins_absent_is_noop():
    load_plugins({})  # no plugins key: silently skipped
    load_plugins({"plugins": None})


def test_plugins_rejects_bad_shape():
    with pytest.raises(ValueError, match="plugins"):
        load_plugins({"plugins": "not.a.list"})
    with pytest.raises(ValueError, match="plugins"):
        load_plugins({"plugins": [123]})
    with pytest.raises(ValueError, match="plugins"):
        load_plugins({"plugins": [""]})


def test_plugins_imports_module_and_adds_cwd_to_path():
    # a stdlib module suffices to verify the import path (may already be loaded -> a no-op also passes)
    import math  # noqa: F401 — load_plugins imports it by name; never referenced

    load_plugins({"plugins": ["math"]})
    import sys
    from pathlib import Path

    assert str(Path.cwd()) in sys.path


def test_plugins_missing_module_fails_fast():
    with pytest.raises(ModuleNotFoundError):
        load_plugins({"plugins": ["no.such.module"]})


def test_plugin_module_import_and_register(tmp_path, monkeypatch):
    # the temp plugin module registers on import (the full path for external plugins joining via the plugins key)
    (tmp_path / "toy_eyas_plugin.py").write_text(
        "from eyas.registry import register_task\n"
        "from eyas.tasks.base import Task\n"
        "\n"
        "@register_task('toy_plugin_task')\n"
        "class ToyPluginTask(Task):\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    load_plugins({"plugins": ["toy_eyas_plugin"]})

    from eyas.registry import get

    assert get("task", "toy_plugin_task").name == "toy_plugin_task"


# --- --set dotted-path overrides (eyas.runtime.config.apply_overrides) ---


def test_override_replaces_terminal_value():
    config = {"run": {"seed": 42}}
    apply_overrides(config, ["run.seed=43"])
    assert config == {"run": {"seed": 43}}


def test_override_value_is_yaml_parsed():
    config = {"model": None}
    apply_overrides(config, ["model=true"])
    assert config == {"model": True}  # bool, not the string "true"


def test_override_missing_key_still_fails_fast():
    # typo protection is the fail-fast doctrine: absent keys are never created
    with pytest.raises(ValueError, match="does not exist"):
        apply_overrides({"run": {"seed": 1}}, ["run.nope=2"])
    with pytest.raises(ValueError, match="does not exist"):
        apply_overrides({}, ["judge.name=exact_match"])


def test_override_over_a_non_mapping_raises():
    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides({"run": 5}, ["run.seed=2"])


def test_override_fills_null_valued_optional_key():
    # ``judge: null`` is a documented config shape (the slot exists, disabled);
    # --set judge.name=... materialises the section instead of erroring
    config = {"judge": None}
    apply_overrides(config, ["judge.name=exact_match"])
    assert config == {"judge": {"name": "exact_match"}}


def test_override_fills_nested_null_valued_optional_key():
    config = {"backend": {"params": None}}
    apply_overrides(config, ["backend.params.model=qwen"])
    assert config == {"backend": {"params": {"model": "qwen"}}}


def test_override_creates_leaf_inside_existing_params():
    # params is the open kwargs namespace (assembled cls(**params)): a config may list
    # only some of the kwargs, and --set backend.params.<new_key> must still work --
    # endpoint/model injected via --set onto a demo config whose mock params never
    # mentioned them must land as new leaves.
    config = {"backend": {"name": "mock", "params": {"responses": ["red"]}}}
    apply_overrides(config, ["backend.params.base_url=http://127.0.0.1:8000/v1/"])
    assert config["backend"]["params"]["base_url"] == "http://127.0.0.1:8000/v1/"


def test_override_materialises_absent_params_section():
    # ``backend: {name: openai}`` without params is a valid config shape; --set may
    # still build the kwargs map
    config = {"backend": {"name": "openai"}}
    apply_overrides(config, ["backend.params.model=qwen"])
    assert config == {"backend": {"name": "openai", "params": {"model": "qwen"}}}


def test_override_typo_inside_params_dies_at_constructor_not_here():
    # open namespace means --set cannot catch a params-key typo; the contract is that
    # it fails one step later at assembly, on the constructor's TypeError (startup).
    # mock backend stands in for openai here (strict kwargs, no heavy deps)
    import eyas.builtin  # noqa: F401  (registration)
    from eyas import registry

    config = {"backend": {"name": "mock", "params": {"responses": ["red"]}}}
    apply_overrides(config, ["backend.params.base_Url=http://x"])  # accepted here
    backend_cls = registry.get("backend", "mock")
    with pytest.raises(TypeError):
        backend_cls(**config["backend"]["params"])


def test_override_intermediate_below_params_still_fails_fast():
    # opening the params namespace does not open arbitrary nesting below it
    with pytest.raises(ValueError, match="does not exist"):
        apply_overrides({"backend": {"params": {}}}, ["backend.params.nested.key=1"])


# --- malformed items are rejected at the boundary (validate/apply split) ---


def test_override_without_equals_raises():
    # the form check is the first gate, not the path walk
    with pytest.raises(ValueError, match="key=value form"):
        apply_overrides({"run": {"seed": 1}}, ["run.seed"])


def test_override_empty_key_raises():
    with pytest.raises(ValueError, match="key=value form"):
        apply_overrides({"run": {"seed": 1}}, ["=5"])


def test_override_empty_value_is_documented_null_coercion():
    # YAML-parsed values contract: ``--set key=`` empty string parses to null, which is
    # how --set disables a section — pinned so the validate split cannot "fix" it away
    config = {"judge": {"name": "exact_match"}}
    apply_overrides(config, ["judge.name="])
    assert config == {"judge": {"name": None}}


def test_override_malformed_yaml_value_raises_actionably():
    # Malformed YAML must not escape as a bare yaml ParserError quoting a buffer
    # offset and naming no item; it raises ValueError naming the offending --set
    # item, same fail-fast family as every other --set rejection.
    with pytest.raises(ValueError, match=r"--set value is not valid YAML: .*a\.b='\[1'"):
        apply_overrides({"a": {"b": 0}}, ["a.b=[1"])
    # the wrapper keeps the scanner error chained for debugging
    with pytest.raises(ValueError) as excinfo:
        apply_overrides({"a": {"b": 0}}, ["a.b={oops:"])
    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_override_value_validated_before_path_walk_mutates():
    # the boundary validates the value before the path walk, so a bad item cannot
    # half-edit the config
    config = {"judge": None}
    with pytest.raises(ValueError, match="not valid YAML"):
        apply_overrides(config, ["judge.name.x=[1"])
    assert config == {"judge": None}  # no slot materialisation survived the failure


# --- atomicity: a bad item anywhere in the list leaves the config untouched ---


def test_override_malformed_later_item_leaves_earlier_ones_unapplied() -> None:
    config = {"run": {"seed": 1}}
    with pytest.raises(ValueError, match="key=value form"):
        apply_overrides(config, ["run.seed=43", "bad item"])
    assert config["run"]["seed"] == 1  # the valid first item was never applied


def test_override_bad_path_later_item_leaves_earlier_ones_unapplied() -> None:
    config = {"run": {"seed": 1}}
    with pytest.raises(ValueError, match="path does not exist"):
        apply_overrides(config, ["run.seed=43", "run.nope=2"])
    assert config["run"]["seed"] == 1


def test_override_failure_does_not_materialise_null_slots() -> None:
    config = {"run": {"seed": 1}, "judge": None}
    with pytest.raises(ValueError, match="path does not exist"):
        apply_overrides(config, ["judge.name=exact_match", "run.nope=2"])
    assert config == {"run": {"seed": 1}, "judge": None}  # no half-filled slot survives
