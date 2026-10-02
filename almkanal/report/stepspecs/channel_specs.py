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


@register_step('MultiBlockMaxwell')
def mulit_maxwell_spec() -> StepSpec:
    # Expand/limit keys as your JSON stabilizes
    return StepSpec(
        settings_fn=keys_selector(
            'coord_frame',
            'destination',
            'destination_source',
            'n_blocks',
            'st_duration',
            'st_correlation',
            'calibration_file',
            'cross_talk_file',
            'calibration_applied',
            'cross_talk_applied',
        )  #'destination',
    )
    # return StepSpec(settings_fn=lambda info: dict(info))


@register_step('EEGRANSAC')
def ransac_spec() -> StepSpec:
    return StepSpec(settings_fn=lambda info: dict(info))


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
