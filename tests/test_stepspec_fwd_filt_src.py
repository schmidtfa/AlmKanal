from __future__ import annotations

import json

from almkanal.report.stepspecs.registry import get_registry, load_stepspec_package


def test_forward_model_selection_extracts_key_fields() -> None:
    load_stepspec_package('almkanal.report.stepspecs')
    spec = get_registry()['ForwardModel']
    info = {
        'source_type': 'surface',
        'source_spacing': 'ico4',
        'volume_spacing_mm': None,
        'anatomy': 'scaled_fsaverage',
        'bem_layers': 1,
        'bem_conductivity': [0.3],
        'min_dist_src_mm': 5.0,
        'meg': True,
        'eeg': False,
        # Realized / subject-specific provenance should not enter settings.
        'n_sources': 5124,
        'n_sources_by_space': [2562, 2562],
        'subject_id_freesurfer': 'sub-01_from_template',
        'subjects_dir': '/subjects',
        'template_src': '/subjects/fsaverage/bem/fsaverage-ico-4-src.fif',
    }

    out = spec.settings_fn(info)

    assert out == {
        'source_type': 'surface',
        'source_spacing': 'ico4',
        'volume_spacing_mm': None,
        'anatomy': 'scaled_fsaverage',
        'bem_layers': 1,
        'bem_conductivity': [0.3],
        'min_dist_src_mm': 5.0,
        'meg': True,
        'eeg': False,
    }
    json.dumps(out)


def test_spatial_filter_selection_extracts_cov_and_norm() -> None:
    load_stepspec_package('almkanal.report.stepspecs')
    spec = get_registry()['SpatialFilter']

    info = {
        'reg': 0.05,
        'pick_ori': 'max-power',
        'weight_norm': 'nai',
        'reduce_rank': False,
        'data_cov_source': 'continuous',
        'noise_cov_source': 'empty_room',

        # Provenance that should not enter report settings.
        'filters': object(),
        'data_cov': object(),
        'noise_cov': object(),
        'extra_data': None,
    }

    out = spec.settings_fn(info)

    assert out == {
        'reg': 0.05,
        'pick_ori': 'max-power',
        'weight_norm': 'nai',
        'reduce_rank': False,
        'data_cov_source': 'continuous',
        'noise_cov_source': 'empty_room',
    }

    json.dumps(out)


def test_source_reconstruction_selection_extracts_parcellation() -> None:
    load_stepspec_package('almkanal.report.stepspecs')
    spec = get_registry()['SourceReconstruction']

    info = {
        'orig_data_type': 'raw',
        'morph2fsaverage': True,
        'return_parc': True,
        'atlas': 'glasser',
        'effective_label_mode': 'pca_flip',

        # provenance only
        'subject_id': 'fsaverage',
        'subjects_dir': '/tmp/freesurfer',
        'source': 'surface',
    }

    out = spec.settings_fn(info)

    assert out == {
        'orig_data_type': 'raw',
        'morph2fsaverage': True,
        'return_parc': True,
        'atlas': 'glasser',
        'effective_label_mode': 'pca_flip',
    }

    json.dumps(out)


def test_source_reconstruction_omits_parcellation_settings_when_disabled() -> None:
    load_stepspec_package('almkanal.report.stepspecs')
    spec = get_registry()['SourceReconstruction']

    info = {
        'orig_data_type': 'epochs',
        'morph2fsaverage': False,
        'return_parc': False,
        'atlas': None,
        'effective_label_mode': None,
    }

    assert spec.settings_fn(info) == {
        'orig_data_type': 'epochs',
        'morph2fsaverage': False,
        'return_parc': False,
    }