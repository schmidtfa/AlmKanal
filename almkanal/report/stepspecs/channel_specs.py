from .registry import StepSpec, keys_selector, register_step


@register_step('Maxwell')
def maxwell_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector('coord_frame', 'destination', 'calibration_file', 'cross_talk_file', 'st_duration')
    )


@register_step('MultiBlockMaxwell')
def mulit_maxwell_spec() -> StepSpec:
    # Expand/limit keys as your JSON stabilizes
    return StepSpec(
        settings_fn=keys_selector('coord_frame', 'calibration_file', 'cross_talk_file', 'st_duration')  #'destination',
    )
    # return StepSpec(settings_fn=lambda info: dict(info))


@register_step('RANSAC')
def ransac_spec() -> StepSpec:
    return StepSpec(settings_fn=lambda info: dict(info))


@register_step('ReReference')
def reref_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('reference', 'ref_channels', 'projection'))
