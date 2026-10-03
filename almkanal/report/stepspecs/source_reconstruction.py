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
    settings = {
        'orig_data_type': info['orig_data_type'],
        'morph2fsaverage': info['morph2fsaverage'],
        'return_parc': info['return_parc'],
    }

    if info['return_parc']:
        settings['atlas'] = info['atlas']
        settings['effective_label_mode'] = info['effective_label_mode']

    return settings


@register_step('SourceReconstruction')
def source_reconstruction_spec() -> StepSpec:
    return StepSpec(settings_fn=_select_source_recon)
