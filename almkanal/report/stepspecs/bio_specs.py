from .registry import StepSpec, register_step


@register_step('PhysioCleaner')
def physio_spec() -> StepSpec:
    return StepSpec(settings_fn=lambda info: dict(info))
