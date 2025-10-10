from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Any

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
    'surrogate_eog_chs',  # expect {"left": [...], "right": [...]}
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
    'ica',  # mne.preprocessing.ICA object
    'component_ids',
    'components_dict',  # used only in summarize()
    'eog_scores',
    'ecg_scores',
}


def _to_native(x: Any) -> Any:
    """Convert numpy scalars/arrays to Python types; leave mappings/lists as-is."""
    try:
        # numpy scalar has .item(); guard with try to avoid importing numpy here
        item = getattr(x, 'item', None)
        if callable(item):
            return item()
    except Exception:
        pass
    return x


def _sanitize_surrogate(chs: Any) -> dict[str, list[str]] | None:
    """Ensure surrogate_eog_chs has {'left': [..], 'right': [..]} lists of strings."""
    if not isinstance(chs, dict):
        return None
    left = chs.get('left')
    right = chs.get('right')
    if not isinstance(left, list | tuple) or not isinstance(right, list | tuple):
        return None
    return {
        'left': [str(c) for c in left],
        'right': [str(c) for c in right],
    }


def _settings(info: dict[str, Any]) -> dict[str, Any]:
    """Pick & sanitize ICA settings for reporting."""
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
    med = median(xs_sorted)
    rng = f'{xs_sorted[0]}–{xs_sorted[-1]}' if xs_sorted[0] != xs_sorted[-1] else str(xs_sorted[0])
    return (f'{med:.1f}' if isinstance(med, float) else str(med)), rng


def _summarize(infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-subject ICA rejections (EOG/ECG/total)."""
    eog_counts: list[int] = []
    ecg_counts: list[int] = []
    total_counts: list[int] = []

    for info in infos:
        comps = info.get('components_dict') or {}
        e = set(comps.get('eog') or [])
        c = set(comps.get('ecg') or [])
        eog_counts.append(len(e))
        ecg_counts.append(len(c))
        total_counts.append(len(e | c))

    out: dict[str, Any] = {}

    # Preferred: mean ± SD
    if total_counts:
        out['total_mean'] = mean(total_counts)
        out['total_sd'] = pstdev(total_counts) if len(total_counts) > 1 else 0.0
    if eog_counts:
        out['eog_mean'] = mean(eog_counts)
        out['eog_sd'] = pstdev(eog_counts) if len(eog_counts) > 1 else 0.0
    if ecg_counts:
        out['ecg_mean'] = mean(ecg_counts)
        out['ecg_sd'] = pstdev(ecg_counts) if len(ecg_counts) > 1 else 0.0

    # Fallback: median + range
    em, er = _med_range(eog_counts)
    cm, cr = _med_range(ecg_counts)
    tm, tr = _med_range(total_counts)
    out.update(
        {
            'eog_median': em,
            'eog_range': er,
            'ecg_median': cm,
            'ecg_range': cr,
            'total_median': tm,
            'total_range': tr,
        }
    )
    return out


@register_step('ICA')
def ica_spec() -> StepSpec:
    return StepSpec(settings_fn=_settings, summarize_fn=_summarize)
