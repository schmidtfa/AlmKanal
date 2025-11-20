from .registry import StepSpec, keys_selector, register_step


@register_step('Events')
def events_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('event_id', 'stim_channel', 'min_duration', 'consecutive'))


@register_step('Epochs')
def epochs_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('tmin', 'tmax', 'baseline', 'reject', 'flat', 'preload'))


@register_step('EpochTRF')
def epochs_trf_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('epoch_len_s'))
