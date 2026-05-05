from __future__ import annotations

from typing import Any

from .registry import StepSpec, register_step

# ---------- helpers


def _get(d: dict, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ico_level_from_nuse(n: int | None) -> str | None:
    """Rough helper: map nuse per hemi -> ico level (surf)."""
    if n is None:
        return None
    # classic fsaverage ico grids per hemi
    mapping = {642: 'ico-3', 2562: 'ico-4', 10242: 'ico-5', 40962: 'ico-6'}
    return mapping.get(int(n))


# ---------- ForwardModel


def _select_forward_model(info: dict[str, Any]) -> dict[str, Any]:
    fwd = info.get('fwd') or {}
    src_list = _get(fwd, 'src', default=[])
    # try to read per-hemi nuse and subject id
    nuse_l = _get(src_list[0], 'nuse') if len(src_list) > 0 else None
    nuse_r = _get(src_list[1], 'nuse') if len(src_list) > 1 else None
    ico = _ico_level_from_nuse(nuse_l) if isinstance(nuse_l, int) else None

    return {
        # surface/volume model
        'source_type': info.get('source_type') or _get(fwd, 'src_type'),
        # orientation mode
        'free_orientation': bool(_get(fwd, 'is_free_ori', default=False))
        or (str(_get(fwd, 'source_ori', default='')) in {'2', 'free', 'FIFFV_MNE_FREE_ORI'}),
        # number of sources
        'n_per_hemi': [int(nuse_l)]
        if nuse_r is None and nuse_l is not None
        else ([int(nuse_l), int(nuse_r)] if (nuse_l and nuse_r) else None),
        'n_total': int(
            _get(fwd, 'nsource', default=nuse_l + nuse_r if isinstance(nuse_l, int) and isinstance(nuse_r, int) else 0)
        )
        or None,
        # grid / spacing hints
        'ico_level': ico,  # e.g., "ico-4" if surface decimated
        # frames / subject template
        'coord_frame': _get(fwd, 'coord_frame') or _get(src_list[0] or {}, 'coord_frame'),
        #'template_subject': info.get('subject_id_freesurfer') or _get(src_list[0] or {}, 'subject_his_id'),
        'subjects_dir': info.get('subject_dir'),
        # bem summary if you log it (optional—will render if present)
        'bem_model': info.get('bem_model'),
        'bem_layers': info.get('bem_layers'),
        'bem_conductivity': info.get('bem_conductivity'),
    }


@register_step('ForwardModel')
def forward_model_spec() -> StepSpec:
    return StepSpec(settings_fn=_select_forward_model)


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
