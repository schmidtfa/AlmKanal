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


@register_step('SpatialFilter')
def spatial_filter_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'reg',
            'pick_ori',
            'weight_norm',
            'reduce_rank',
            'data_cov_source',
            'noise_cov_source',
        )
    )


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
