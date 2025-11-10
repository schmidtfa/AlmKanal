from __future__ import annotations

import json
import pytest

from almkanal.report.stepspecs.registry import get_registry, load_stepspec_package


@pytest.fixture(scope="module", autouse=True)
def _load_specs() -> None:
    load_stepspec_package("almkanal.report.stepspecs")


def test_filter_settings_pick_expected_keys() -> None:
    reg = get_registry()
    if "Filter" not in reg:
        pytest.skip("Filter StepSpec not registered")

    info = {
        "l_freq": 0.1,
        "h_freq": 40.0,
        "method": "fir",
        "phase": "zero",
        "fir_window": "hamming",
        "fir_design": "firwin",
        "pad": "reflect_limited",
        "l_trans_bandwidth": "auto",
        "h_trans_bandwidth": "auto",
        "filter_length": "auto",
        "skip_by_annotation": ["edge"],
        "EXTRA_FIELD": "SHOULD_BE_DROPPED",
    }
    out = reg["Filter"].settings_fn(info)
    # must include known keys, and must not pass through unknowns
    assert "l_freq" in out and "h_freq" in out and "EXTRA_FIELD" not in out
    json.dumps(out)


def test_events_and_epochs_minimal() -> None:
    reg = get_registry()
    if "Events" not in reg or "Epochs" not in reg:
        pytest.skip("Events/Epochs StepSpecs not registered")

    events = {
        "event_id": {"A": 1, "B": 2},
        "stim_channel": "STI101",
        "min_duration": 0.001,
        "consecutive": "increasing",
    }
    epochs = {
        "tmin": -0.2,
        "tmax": 0.5,
        "baseline": [None, 0.0],
        "reject": {"mag": 4e-12},
        "preload": True,
    }

    ev_out = reg["Events"].settings_fn(events)
    ep_out = reg["Epochs"].settings_fn(epochs)

    assert ev_out.get("event_id") == {"A": 1, "B": 2}
    assert ep_out.get("tmin") == -0.2 and ep_out.get("tmax") == 0.5
    json.dumps(ev_out); json.dumps(ep_out)
