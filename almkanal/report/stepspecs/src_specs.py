from .registry import StepSpec, keys_selector, register_step


def _select_fwd(info: dict) -> dict:
    fwd = info.get('fwd') or {}
    return {
        'source_type': info.get('source_type'),
        'nsource': fwd.get('nsource'),
        'source_ori': fwd.get('source_ori'),
        'surf_ori': fwd.get('surf_ori'),
        'is_free_ori': fwd.get('is_free_ori'),
    }


@register_step('ForwardModel')
def fwd_spec() -> StepSpec:
    return StepSpec(settings_fn=_select_fwd)


def _select_sfilt(info: dict) -> dict:
    filt = info.get('filters') or {}
    lcmv = info.get('lcmv_settings') or {}
    return {
        'kind': filt.get('kind'),
        'pick_ori': (filt.get('pick_ori') or lcmv.get('pick_ori')),
        'weight_norm': (filt.get('weight_norm') or lcmv.get('weight_norm')),
        'rank': (filt.get('rank') or lcmv.get('rank')),
        'is_free_ori': filt.get('is_free_ori'),
        'n_sources': filt.get('n_sources'),
        'src_type': filt.get('src_type'),
    }


@register_step('SpatialFilter')
def spatial_spec() -> StepSpec:
    return StepSpec(settings_fn=_select_sfilt)


@register_step('SourceReconstruction')
def source_recon_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector('orig_data_type', 'source', 'atlas', 'label_mode', 'subject_id', 'subjects_dir')
    )
