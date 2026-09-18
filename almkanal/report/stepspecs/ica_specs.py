from __future__ import annotations

from typing import Any

from numpy import mean, median, std

from .registry import StepSpec, register_step

# Settings keys we want to keep if present in the JSON
_ALLOWED_KEYS: set[str] = {
    'method',
    'n_components',
    'ica_hp_freq',
    'ica_lp_freq',
    'resample_freq',
    'corr_metric',
    'eog',
    'eog_corr_thresh',
    'surrogate_eog_chs',
    'ecg',
    'ecg_corr_thresh',
    'ecg_from_meg',
    'emg',
    'emg_thresh',
    'train',
    'train_freq',
    'train_thresh',
    'fit_only',
}

# Keys to drop (non-serializable or large)
_IGNORE_KEYS: set[str] = {
    'ica',
    'component_ids',
    'components_dict',
    'eog_scores',
    'ecg_scores',
    # add others here if they appear, e.g. 'emg_scores'
}


def _to_native(x: Any) -> Any:
    try:
        item = getattr(x, 'item', None)
        if callable(item):
            return item()
    except Exception:
        pass
    return x


def _sanitize_surrogate(chs: Any) -> dict[str, list[str]] | None:
    if not isinstance(chs, dict):
        return None
    left = chs.get('left')
    right = chs.get('right')
    if not isinstance(left, list | tuple) or not isinstance(right, list | tuple):
        return None
    return {'left': [str(c) for c in left], 'right': [str(c) for c in right]}


def _settings(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in info.items():
        if k in _IGNORE_KEYS:
            continue
        if k in _ALLOWED_KEYS:
            if k == 'surrogate_eog_chs':
                sv = _sanitize_surrogate(v)
                if sv is not None:
                    out[k] = sv
                continue
            out[k] = _to_native(v)
    return out


def _med_range(xs: list[int]) -> tuple[str, str]:
    if not xs:
        return 'n/a', 'n/a'
    xs_sorted = sorted(xs)
    med = median(xs_sorted)  # may be float
    rng = f'{xs_sorted[0]}–{xs_sorted[-1]}' if xs_sorted[0] != xs_sorted[-1] else str(xs_sorted[0])
    return (f'{float(med):.1f}' if isinstance(med, float) else str(med)), rng


def _mean_sd(xs: list[int]) -> tuple[float, float] | None:
    if not xs:
        return None
    m = float(mean(xs))
    s = float(std(xs)) if len(xs) > 1 else 0.0  # NumPy std defaults to population SD (ddof=0)
    return m, s


def _summarize(infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-subject ICA rejections (EOG/ECG/EMG/train and total)."""
    eog_counts: list[int] = []
    ecg_counts: list[int] = []
    emg_counts: list[int] = []
    train_counts: list[int] = []
    total_counts: list[int] = []

    for info in infos:
        comps = info.get('components_dict') or {}
        e = set(comps.get('eog') or [])
        c = set(comps.get('ecg') or [])
        m = set(comps.get('emg') or [])
        t = set(comps.get('train') or [])
        eog_counts.append(len(e))
        ecg_counts.append(len(c))
        emg_counts.append(len(m))
        train_counts.append(len(t))
        total_counts.append(len(e | c | m | t))

    out: dict[str, Any] = {}

    # Means ± SDs (JSON-friendly floats)
    for key, arr in (
        ('eog', eog_counts),
        ('ecg', ecg_counts),
        ('emg', emg_counts),
        ('train', train_counts),
        ('total', total_counts),
    ):
        ms = _mean_sd(arr)
        if ms is not None:
            out[f'{key}_mean'], out[f'{key}_sd'] = ms

    # Medians + ranges (fallbacks)
    for key, arr in (
        ('eog', eog_counts),
        ('ecg', ecg_counts),
        ('emg', emg_counts),
        ('train', train_counts),
        ('total', total_counts),
    ):
        med, rng = _med_range(arr)
        out[f'{key}_median'] = med
        out[f'{key}_range'] = rng

    return out


@register_step('ICA')
def ica_spec() -> StepSpec:
    return StepSpec(settings_fn=_settings, summarize_fn=_summarize)
