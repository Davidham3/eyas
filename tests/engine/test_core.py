import dataclasses

import pytest

from eyas.core.objective import Objective


def test_objective_oriented_direction():
    max_o = Objective(key="accuracy", maximize=True)
    min_o = Objective(key="loss", maximize=False)
    assert max_o.oriented(0.7) == 0.7
    assert min_o.oriented(0.7) == -0.7


def test_objective_display_name_defaults_to_key():
    o = Objective(key="field_acc")
    assert o.display_name == "field_acc"


def test_objective_is_frozen():
    o = Objective(key="accuracy")
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.key = "other"  # type: ignore[misc]
