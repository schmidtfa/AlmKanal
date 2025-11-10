from __future__ import annotations

import json
import pytest

from typing import Any

from almkanal.report.stepspecs.registry import get_registry, load_stepspec_package


@pytest.fixture(scope="module", autouse=True)
def _load_specs() -> None:
    # Import all specs so module-level @register_step runs
    load_stepspec_package("almkanal.report.stepspecs")


def test_registry_contains_core_steps() -> None:
    reg = get_registry()
    # We expect at least these from the scaffold we built together
    expected = {
        "Filter",
        "ICA",
        "Events",
        "Epochs",
        "ForwardModel",
        "SpatialFilter",
        "SourceReconstruction",
    }
    missing = expected - set(reg)
    assert not missing, f"Missing StepSpecs: {sorted(missing)}"


@pytest.mark.parametrize("step_name", [
    "Filter", "Events", "Epochs", "ReReference",
    "ForwardModel", "SpatialFilter", "SourceReconstruction",
])
def test_settings_fn_returns_jsonable_dict(step_name: str) -> None:
    reg = get_registry()
    if step_name not in reg:
        pytest.skip(f"{step_name} not registered in this build")

    # minimal, permissive example payloads per step
    examples: dict[str, dict[str, Any]] = {
        "Filter": {
            "l_freq": 0.1, "h_freq": 40.0, "method": "fir", "phase": "zero",
            "fir_window": "hamming", "fir_design": "firwin", "pad": "reflect_limited",
            "l_trans_bandwidth": "auto", "h_trans_bandwidth": "auto", "filter_length": "auto",
            "skip_by_annotation": ["edge", "bad_acq_skip"],
        },
        "Events": {
            "event_id": {"A": 1, "B": 2}, "stim_channel": "STI101",
            "min_duration": 0.002, "consecutive": "increasing",
        },
        "Epochs": {
            "tmin": -0.2, "tmax": 0.5, "baseline": [None, 0.0],
            "reject": {"mag": 4e-12}, "flat": None, "preload": True,
        },
        "ReReference": {"reference": "average", "projection": False},
        "ForwardModel": {
            "fwd": {
                "nsource": 5124, "source_ori": 2, "is_free_ori": False, "coord_frame": 4,
                "src": [
                    {"nuse": 2562, "coord_frame": 4, "subject_his_id": "fsaverage"},
                    {"nuse": 2562, "coord_frame": 4, "subject_his_id": "fsaverage"},
                ],
            },
            "source_type": "surface",
            "subject_id_freesurfer": "fsaverage",
            "subject_dir": "/path/to/subjects_dir",
        },
        "SpatialFilter": {
            "filters": {
                "kind": "LCMV", "pick_ori": "max-power", "weight_norm": "nai", "rank": 56,
                "is_free_ori": False, "n_sources": 5124, "src_type": "surface",
                "data_cov": {"data": {}}, "noise_cov": {"data": {}, "source": "empty_room"},
            },
            "lcmv_settings": {"reg": 0.05, "pick_ori": "max-power", "weight_norm": "nai", "rank": {"mag": 56}},
        },
        "SourceReconstruction": {
            "orig_data_type": "raw", "source": "surface", "atlas": "glasser",
            "label_mode": "pca_flip", "subjects_dir": "/path/to/subjects_dir",
            "subject_id": "fsaverage", "n_labels": 360,
        },
    }


    payload = examples[step_name]
    spec = reg[step_name]
    out = spec.settings_fn(payload)
    assert isinstance(out, dict), "settings_fn must return a dict"
    # Must be JSON-serializable
    json.dumps(out)
