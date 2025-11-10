from .registry import StepSpec, keys_selector, register_step


@register_step('Filter')
def filter_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'l_freq',
            'h_freq',
            'method',
            'iir_params',
            'phase',
            'fir_window',
            'fir_design',
            'pad',
            'l_trans_bandwidth',
            'h_trans_bandwidth',
        )
    )


@register_step('Resample')
def resample_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('sfreq', 'new_sfreq', 'resample_sfreq', 'npad', 'window'))
