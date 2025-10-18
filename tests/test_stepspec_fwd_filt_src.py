from __future__ import annotations

import json

from almkanal.report.stepspecs.registry import get_registry, load_stepspec_package


def test_forward_model_selection_extracts_key_fields() -> None:
    load_stepspec_package("almkanal.report.stepspecs")
    spec = get_registry()["ForwardModel"]

    info = {
        "fwd": {
            "nsource": 5124,
            "source_ori": 2,
            "is_free_ori": False,
            "coord_frame": 4,
            "src": [
                {"nuse": 2562, "coord_frame": 4, "subject_his_id": "fsaverage"},
                {"nuse": 2562, "coord_frame": 4, "subject_his_id": "fsaverage"},
            ],
        },
        "source_type": "surface",
        "subject_id_freesurfer": "fsaverage",
        "subject_dir": "/subjects",
    }
    out = spec.settings_fn(info)
    assert out["source_type"] == "surface"
    assert out["free_orientation"] in (True, False)
    assert out["n_total"] == 5124
    assert out["ico_level"] == "ico-4"  # from 2562 per hemi
    assert out["template_subject"] == "fsaverage"
    json.dumps(out)  # serializable


def test_spatial_filter_selection_extracts_cov_and_norm() -> None:
    load_stepspec_package("almkanal.report.stepspecs")
    spec = get_registry()["SpatialFilter"]

    info = {
        "filters": {
            "kind": "LCMV",
            "pick_ori": "max-power",
            "weight_norm": "nai",
            "rank": 56,
            "is_free_ori": False,
            "n_sources": 5124,
            "src_type": "surface",
            "data_cov": {"data": {}},
            "noise_cov": {"data": {}, "source": "empty_room"},
        },
        "lcmv_settings": {"reg": 0.05, "rank": {"mag": 56}},
    }
    out = spec.settings_fn(info)
    assert out["kind"] == "LCMV"
    assert out["pick_ori"] == "max-power"
    assert out["weight_norm"] == "nai"
    assert out["has_data_cov"] is True
    assert out["has_noise_cov"] is True
    assert out["noise_cov_source"] == "empty_room"
    json.dumps(out)


def test_source_reconstruction_selection_extracts_parcellation() -> None:
    load_stepspec_package("almkanal.report.stepspecs")
    spec = get_registry()["SourceReconstruction"]

    info = {
        "orig_data_type": "raw",
        "source": "surface",
        "atlas": "glasser",
        "label_mode": "pca_flip",
        "subjects_dir": "/subjects",
        "subject_id": "fsaverage",
        "n_labels": 360,
    }
    out = spec.settings_fn(info)
    assert out["atlas"] == "glasser"
    assert out["label_mode"] == "pca_flip"
    assert out["n_labels"] == 360
    json.dumps(out)
