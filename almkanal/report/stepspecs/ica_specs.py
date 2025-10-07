from statistics import median

from .registry import StepSpec, register_step


def _settings(info: dict) -> dict:
    ignore = {'component_ids', 'components_dict', 'eog_scores', 'ecg_scores'}
    return {k: v for k, v in info.items() if k not in ignore}


def _summarize(infos: list[dict]) -> dict:
    eog, ecg, total = [], [], []
    for info in infos:
        comps = info.get('components_dict') or {}
        e = set(comps.get('eog') or [])
        c = set(comps.get('ecg') or [])
        eog.append(len(e))
        ecg.append(len(c))
        total.append(len(e | c))

    def med_rng(xs: list[int]) -> tuple[str, str]:
        if not xs:
            return 'n/a', 'n/a'
        xs = sorted(xs)
        med = median(xs)
        rng = f'{xs[0]}–{xs[-1]}' if xs[0] != xs[-1] else f'{xs[0]}'
        return (f'{med:.1f}' if isinstance(med, float) else str(med)), rng

    em, er = med_rng(eog)
    cm, cr = med_rng(ecg)
    tm, tr = med_rng(total)

    return {
        'eog_median': em,
        'eog_range': er,
        'ecg_median': cm,
        'ecg_range': cr,
        'total_median': tm,
        'total_range': tr,
    }


@register_step('ICA')
def spec() -> StepSpec:
    return StepSpec(settings_fn=_settings, summarize_fn=_summarize)
