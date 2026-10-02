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
            'filter_length',
            'filter_length_samples',
            'filter_length_seconds',
            'filter_order',
            'l_trans_bandwidth',
            'h_trans_bandwidth',
            'applied_to',
            'pad',
        )
    )


@register_step('Resample')
def resample_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector('sfreq', 'original_sfreq', 'applied_to', 'method', 'window', 'npad', 'pad')
    )
