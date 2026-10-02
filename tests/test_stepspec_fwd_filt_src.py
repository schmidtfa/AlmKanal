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
        'filters': {
            'kind': 'LCMV',
            'pick_ori': 'max-power',
            'weight_norm': 'nai',
            'rank': 56,
            'is_free_ori': False,
            'n_sources': 5124,
            'src_type': 'surface',
            'data_cov': {'data': {}},
            'noise_cov': {'data': {}, 'source': 'empty_room'},
        },
        'lcmv_settings': {'reg': 0.05, 'rank': {'mag': 56}},
    }
    out = spec.settings_fn(info)
    assert out['kind'] == 'LCMV'
    assert out['pick_ori'] == 'max-power'
    assert out['weight_norm'] == 'nai'
    assert out['has_data_cov'] is True
    assert out['has_noise_cov'] is True
    assert out['noise_cov_source'] == 'empty_room'
    json.dumps(out)


def test_source_reconstruction_selection_extracts_parcellation() -> None:
    load_stepspec_package('almkanal.report.stepspecs')
    spec = get_registry()['SourceReconstruction']
    info = {
        'orig_data_type': 'raw',
        'source': 'surface',
        'atlas': 'glasser',
        'label_mode': 'pca_flip',
        'subjects_dir': '/subjects',
        'subject_id': 'fsaverage',
        'n_labels': 360,
    }
    out = spec.settings_fn(info)
    assert out['atlas'] == 'glasser'
    assert out['label_mode'] == 'pca_flip'
    assert out['n_labels'] == 360
    json.dumps(out)
