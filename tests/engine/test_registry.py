import pytest

from eyas import registry
from eyas.engine import registry as engine_registry


def test_duplicate_registration_raises():
    from eyas.backends.base import InferenceBackend

    with pytest.raises(ValueError, match="duplicate registration"):

        @registry.register_backend("mock")
        class AnotherMock(InferenceBackend):
            name = "mock"


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="unknown registry group"):
        registry.register("nope", "x")


def test_get_unknown_name_lists_registered():
    import eyas.builtin  # noqa: F401

    with pytest.raises(KeyError, match="no_such_task"):
        registry.get("task", "no_such_task")


def test_registration_sets_name():
    import eyas.builtin  # noqa: F401

    assert registry.get("task", "demo_shapes").name == "demo_shapes"


def test_registrar_semantics_preserved():
    @engine_registry.register_strategy("tmp_s")
    class S: ...

    @engine_registry.register_eval_model("tmp_e")
    class E: ...

    assert S.name == "tmp_s"  # strategy sets cls.name
    assert not hasattr(E, "name")  # eval_model does not
    assert engine_registry.get("strategy", "tmp_s") is S
    assert engine_registry.get("eval_model", "tmp_e") is E


def test_registrar_semantics_preserved_pairwise_and_budget():
    @engine_registry.register_pairwise_judge("tmp_p")
    class P: ...

    @engine_registry.register_budget("tmp_b")
    class B: ...

    assert P.name == "tmp_p"  # pairwise_judge sets cls.name
    assert not hasattr(B, "name")  # budget does not
    assert engine_registry.get("pairwise_judge", "tmp_p") is P
    assert engine_registry.get("budget", "tmp_b") is B


def test_duplicate_budget_registration_raises():
    @engine_registry.register_budget("tmp_dup")
    class First: ...

    with pytest.raises(ValueError, match="duplicate registration"):

        @engine_registry.register_budget("tmp_dup")
        class Second: ...


def test_plugin_load_failure_raises_again_on_the_next_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken plugin must break EVERY get() until it is fixed: the loaded flag
    may only latch after the load loop completes, never before it (a flag set
    up front turns the second call into a silent skip of the broken plugin)."""

    class BrokenEntryPoint:
        name = "broken_plugin"

        def load(self) -> None:
            raise RuntimeError("broken plugin exploded")

    # monkeypatch teardown restores both the real entry_points and the original
    # _plugins_loaded flag, so no other test sees the broken plugin or a reset flag
    monkeypatch.setattr(registry, "entry_points", lambda group: [BrokenEntryPoint()])
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    with pytest.raises(RuntimeError, match="broken plugin exploded"):
        registry.get("task", "demo_shapes")
    with pytest.raises(RuntimeError, match="broken plugin exploded"):
        registry.get("task", "demo_shapes")
