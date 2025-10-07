from .registry import StepSpec, keys_selector, register_step


@register_step('Filter')
def filter_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'l_freq', 'h_freq', 'method', 'phase', 'fir_window', 'fir_design', 'pad', 'skip_by_annotation'
        )
    )


@register_step('Resample')
def resample_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('sfreq', 'new_sfreq', 'resample_sfreq', 'npad', 'window'))
