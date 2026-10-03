from __future__ import annotations

import json
from numpy import isclose

from almkanal.report.stepspecs.registry import get_registry, load_stepspec_package


def test_ica_settings_and_summary_roundtrip() -> None:
    load_stepspec_package("almkanal.report.stepspecs")
    spec = get_registry()["ICA"]

    # Two subjects, different numbers of rejected components
    info1 = {
        # settings
        "method": "picard",
        "n_components": 50,
        "ica_hp_freq": 1.0,
        "ica_lp_freq": None,
        "resample_freq": 200,
        "corr_metric": "correlation",
        "eog": True, "eog_corr_thresh": 0.8,
        "ecg": True, "ecg_corr_thresh": 0.4, "ecg_from_meg": True,
        "emg": False, "emg_thresh": 0.5,
        "train": True, "train_freq": 16, "train_thresh": 3.0,
        "fit_only": False,
        "surrogate_eog_chs": {"left": ["MEG0121"], "right": ["MEG1211"]},
        # results inputs
        "components_dict": {"eog": [1, 5], "ecg": [7], "train": []},
    }
    info2 = {
        "method": "picard",
        "n_components": 50,
        "ica_hp_freq": 1.0,
        "resample_freq": 200,
        "corr_metric": "correlation",
        "eog": True, "eog_corr_thresh": 0.8,
        "ecg": True, "ecg_corr_thresh": 0.4,
        "emg": False,
        "train": True, "train_freq": 16, "train_thresh": 3.0,
        "fit_only": False,
        "components_dict": {"eog": [3], "ecg": [], "train": [9]},
    }

    # settings sanitization
    settings = spec.settings_fn(info1)
    assert settings["method"] == "picard"
    assert settings["n_components"] == 50
    assert settings["ica_hp_freq"] == 1.0
    assert settings["resample_freq"] == 200
    # surrogate channel structure normalized
    assert settings["surrogate_eog_chs"] == {"left": ["MEG0121"], "right": ["MEG1211"]}

    # JSON-serializable
    json.dumps(settings)

    # summarize across two subjects
    summary = spec.summarize_fn([info1, info2])  # type: ignore[arg-type]
    # expected counts: subj1 total=3 (2 eog + 1 ecg); subj2 total=2 (1 eog + 1 train)
    # means
    assert isclose(summary["total_mean"], 2.5, rtol=1e-3)
    # SD of [3,2] (population SD) = 0.5
    assert isclose(summary["total_sd"], 0.5, rtol=1e-3)
    # fallback fields exist
    assert "total_median" in summary and "total_range" in summary
