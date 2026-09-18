from typing import Any

import numpy as np

from .registry import StepSpec, keys_selector, register_step


@register_step('Events')
def events_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('event_id', 'stim_channel', 'min_duration', 'consecutive'))


@register_step('Epochs')
def epochs_spec() -> StepSpec:
    return StepSpec(settings_fn=keys_selector('tmin', 'tmax', 'baseline', 'reject', 'flat', 'preload'))


def _summarize_trf(infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool all successful trials, using saved moments rather than truncated lists."""
    inference_counts: dict[str, int] = {}
    for info in infos:
        for rate, count in info.get('alignment_info', {}).get('end_inference_drift_counts', {}).items():
            inference_counts[rate] = inference_counts.get(rate, 0) + count
    results: dict[str, Any] = {
        'n_trials_end_inferred': sum(inference_counts.values()),
        'end_inference_drift_counts': dict(sorted(inference_counts.items(), key=lambda item: float(item[0]))),
    }
    alignments = [info['alignment_info'] for info in infos if info.get('realign_audio') and 'alignment_info' in info]
    if not alignments:
        return results
    results.update(
        {
            key: sum(alignment[key] for alignment in alignments)
            for key in ('n_trials_found', 'n_trials_aligned', 'n_trials_failed', 'n_trials_epoched')
        }
    )
    results['n_recordings'] = len(alignments)
    metrics: dict[str, list[dict[str, Any]]] = {}
    for alignment in alignments:
        for name, summary in alignment['summary'].items():
            if summary['n']:
                metrics.setdefault(name, []).append(summary)
    pooled = {}
    for name, summaries in metrics.items():
        n = sum(summary['n'] for summary in summaries)
        mean = sum(summary['n'] * summary['mean'] for summary in summaries) / n
        variance = sum(summary['n'] * (summary['sd'] ** 2 + (summary['mean'] - mean) ** 2) for summary in summaries) / n
        pooled[name] = {
            'n': n,
            'mean': mean,
            'sd': float(np.sqrt(variance)),
            'min': min(summary['min'] for summary in summaries),
            'max': max(summary['max'] for summary in summaries),
        }
    results['metrics'] = pooled
    return results


@register_step('EpochTRF')
def epochs_trf_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'epoch_len_s',
            'feature',
            'audio_cutoff_hz',
            'hw_delay_s',
            'applied_hw_delay_s',
            'realign_audio',
            'alignment_method',
            'realign_without_audio',
            'fallback_drift_us_per_s',
            'audio_channels',
            'alignment_kwargs',
            'preserve_annotations',
            'on_alignment_error',
        ),
        summarize_fn=_summarize_trf,
    )
