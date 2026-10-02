from __future__ import annotations

from typing import Any

from .registry import StepSpec, keys_selector, register_step

# ---------- helpers


def _get(d: dict, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ---------- ForwardModel


@register_step('ForwardModel')
def forward_model_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'source_type',
            'source_spacing',
            'volume_spacing_mm',
            'anatomy',
            'bem_layers',
            'bem_conductivity',
            'min_dist_src_mm',
            'meg',
            'eeg',
        )
    )


# ---------- SpatialFilter (e.g., LCMV beamformer)


def _select_spatial_filter(info: dict[str, Any]) -> dict[str, Any]:
    filt = info.get('filters') or {}
    lcmv = info.get('lcmv_settings') or {}

    # detect empty-room provenance if you log it at either top or inside noise_cov
    noise_cov = filt.get('noise_cov') or {}
    noise_src = info.get('noise_cov_source') or noise_cov.get('source')

    return {
        'kind': (filt.get('kind') or 'LCMV'),
        'pick_ori': (filt.get('pick_ori') or lcmv.get('pick_ori')),
        'weight_norm': (filt.get('weight_norm') or lcmv.get('weight_norm')),
        #'rank': (filt.get('rank') or lcmv.get('rank')),
        'is_free_ori': bool(filt.get('is_free_ori')),
        'n_sources': filt.get('n_sources')
        or _get(info, 'filters', 'vertices')
        and sum(
            _get(info, 'filters', 'vertices')[i]['size'] for i in (0, 1) if len(_get(info, 'filters', 'vertices')) > i
        )
        or None,
        'src_type': filt.get('src_type'),
        # covariance book-keeping
        'has_data_cov': bool(filt.get('data_cov')),
        'has_noise_cov': bool(filt.get('noise_cov')),
        'noise_cov_source': noise_src,  # e.g. "empty_room", "pre-stimulus", etc.
        'reg': lcmv.get('reg'),
    }


@register_step('SpatialFilter')
def spatial_filter_spec() -> StepSpec:
    return StepSpec(settings_fn=_select_spatial_filter)


# ---------- SourceReconstruction / parcellation


def _select_source_recon(info: dict[str, Any]) -> dict[str, Any]:
    return {
        'orig_data_type': info.get('orig_data_type'),
        'source': info.get('source'),  # "surface"|"volume" if you log it here
        'atlas': info.get('atlas'),  # e.g., "glasser" (HCP-MMP1)
        'label_mode': info.get('label_mode'),  # e.g., "pca_flip"
        'subjects_dir': info.get('subjects_dir'),
        #'subject_id': info.get('subject_id'),
        # optional: number of labels if you log it
        'n_labels': info.get('n_labels'),
    }


@register_step('SourceReconstruction')
def source_reconstruction_spec() -> StepSpec:
    return StepSpec(settings_fn=_select_source_recon)
