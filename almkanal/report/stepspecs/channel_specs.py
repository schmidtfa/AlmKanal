from .registry import StepSpec, keys_selector, register_step


@register_step('Maxwell')
def maxwell_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'coord_frame',
            'destination',
            'st_duration',
            'st_correlation',
            'calibration_file',
            'cross_talk_file',
            'calibration_applied',
            'cross_talk_applied',
        )
    )


def multiblock_maxwell_settings(info: dict) -> dict:
    settings = {
        'coord_frame': info['coord_frame'],
        'destination_source': info['destination_source'],
        'st_duration': info['st_duration'],
        'st_correlation': info['st_correlation'],
        'calibration_file': info['calibration_file'],
        'cross_talk_file': info['cross_talk_file'],
        'calibration_applied': info['calibration_applied'],
        'cross_talk_applied': info['cross_talk_applied'],
    }

    if info['destination_source'] == 'explicit':
        settings['destination'] = info['destination']

    return settings


@register_step('MultiBlockMaxwell')
def multiblock_maxwell_spec() -> StepSpec:
    return StepSpec(
        settings_fn=multiblock_maxwell_settings,
    )


@register_step('EEGRANSAC')
def ransac_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'ransac_epoch_duration',
            'n_resample',
            'min_channels',
            'min_corr',
            'unbroken_time',
            'random_state',
            'interpolation_method',
        )
    )


@register_step('ReReference')
def rereference_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'ref_channels',
            'projection',
            'resolved_ch_type',
            'joint',
        )
    )
