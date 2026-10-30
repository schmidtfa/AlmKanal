from __future__ import annotations

import warnings
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import librosa
import mne
import numpy as np
from scipy.signal import butter, correlate, hilbert, sosfiltfilt
from scipy.stats import theilslopes

from almkanal.defaults import function_defaults
from almkanal.stim_utils.audio_utils import resample_poly_exact

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

VARIANCE_EPSILON = 1e-12
ENERGY_EPSILON = 1e-20
MIN_ANCHORS_FOR_DRIFT = 2
DEFAULT_FALLBACK_DRIFT_US_PER_S = 0.0  # General profile; Salzburg uses +499 us/s.
LAG_LIMIT_MARGIN_FRACTION = 0.05
MIN_ANCHOR_TIME_COVERAGE = 0.5


def resolve_stim_channel(raw: mne.io.BaseRaw, stim_channel: str | None) -> str:
    """Resolve the trigger channel, requiring an unambiguous choice.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording whose channel names and types are inspected.
    stim_channel : str | None
        Explicit channel name, or ``None`` to select the sole stim channel.

    Returns
    -------
    name : str
        Trigger channel name. Raises if the named channel is absent or automatic
        selection finds zero or multiple stim channels.
    """
    if stim_channel is not None:
        if stim_channel not in raw.ch_names:
            raise ValueError(f'Stim channel {stim_channel!r} is not present in the Raw object.')
        return stim_channel

    picks = mne.pick_types(raw.info, meg=False, stim=True)
    if len(picks) != 1:
        names = [raw.ch_names[pick] for pick in picks]
        raise RuntimeError(f'Specify stim_channel explicitly; detected stim channels: {names}')
    return raw.ch_names[picks[0]]


def _infer_audio_trial_end(
    trial: dict[str, Any],
    raw: mne.io.BaseRaw,
    boundary_sample: int,
    base_audio_path: str | Path | None,
    drift_us_per_s: float,
) -> dict[str, Any]:
    """Estimate a missing endpoint without writing a trigger into the Raw data."""
    wav_path = Path(trial['wav_file'])
    if not wav_path.is_absolute():
        if base_audio_path is None:
            raise ValueError('base_audio_path is required to infer trial ends from relative WAV paths.')
        wav_path = Path(base_audio_path) / wav_path
    if not wav_path.is_file():
        raise FileNotFoundError(f'WAV file does not exist: {wav_path}')
    duration_s = float(librosa.get_duration(path=wav_path))
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f'WAV duration must be finite and positive: {wav_path}')
    duration_samples = int(round(duration_s * (1.0 + drift_us_per_s / 1e6) * raw.info['sfreq']))
    end_sample = trial['onset_sample'] + duration_samples
    if duration_samples < 1:
        raise ValueError(f'Inferred trial duration must span at least one raw sample: {wav_path}')
    if end_sample > boundary_sample:
        raise RuntimeError(
            f'Inferred end for WAV={wav_path} at sample {end_sample} exceeds the next audio onset '
            f'or recording end at sample {boundary_sample}. Check the WAV, onset triggers, and fallback drift.'
        )
    # Endpoints passed to Raw.crop are inclusive; the recording stop is exclusive.
    end_sample = min(end_sample, int(raw.first_samp + raw.n_times - 1))
    if end_sample <= trial['onset_sample']:
        raise ValueError(f'Insufficient recording samples to infer an end for WAV={wav_path}.')
    return {
        **trial,
        'end_code': None,
        'end_sample': end_sample,
        'end_inferred': True,
        'end_inference_drift_us_per_s': float(drift_us_per_s),
        'end_inference_wav_duration_s': duration_s,
    }


@function_defaults(
    'trials',
    aliases={
        'stim_channel': ('events', 'stim_channel'),
        'fallback_drift_us_per_s': ('stimuli', 'fallback_drift_us_per_s'),
    },
)
def find_audio_trials(  # noqa: C901, PLR0912, PLR0915
    raw: mne.io.BaseRaw,
    onset_trigger_to_wav: Mapping[int, str | Path],
    end_triggers: int | Sequence[int] | None = None,
    *,
    stim_channel: str | None = None,
    infer_missing_ends: bool = False,
    base_audio_path: str | Path | None = None,
    fallback_drift_us_per_s: float = DEFAULT_FALLBACK_DRIFT_US_PER_S,
) -> list[dict[str, Any]]:
    """Pair audio onset/end triggers, optionally inferring missing endpoints.

    Defaults below describe the general profile; omitted settings use the
    active defaults profile.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording containing the stimulus triggers.
    onset_trigger_to_wav : mapping of int to str | pathlib.Path
        Onset trigger codes mapped to WAV paths. Repeated codes create
        separate trials referencing the same WAV.
    end_triggers : int | sequence of int | None, default=None
        Codes that end a trial. ``None`` requires ``infer_missing_ends=True``.
    stim_channel : str | None, default=None
        Trigger channel name. ``None`` requires exactly one stim channel.
    infer_missing_ends : bool, default=False
        Estimate missing endpoints from WAV duration and assumed drift.
        Recorded end triggers take precedence.
    base_audio_path : str | pathlib.Path | None, default=None
        WAV directory, required for relative paths when inferring ends.
    fallback_drift_us_per_s : float, default=0.0
        Assumed drift in microseconds per second for end inference.
        Positive values lengthen the span; Salzburg defaults to +499.

    Returns
    -------
    trials : list of dict
        Trials in trigger order with ``onset_code``, ``end_code``, ``wav_file``,
        ``onset_sample``, and ``end_sample``. Samples include ``raw.first_samp``.
        Inferred ends also include ``end_inferred`` and the drift/duration used.

    Notes
    -----
    Inference uses ``WAV duration * (1 + drift / 1e6)`` on the raw clock and
    raises if the endpoint exceeds the next onset or recording end. It does not
    alter triggers, resample data, or correct physical delay.

    Without inference, a new onset before an end raises an error; a final trial
    without an end emits a warning and is omitted. Unrelated triggers are ignored.
    """
    if not onset_trigger_to_wav:
        raise ValueError('onset_trigger_to_wav must contain at least one onset-trigger/WAV pair.')
    if infer_missing_ends and (not np.isfinite(fallback_drift_us_per_s) or 1.0 + fallback_drift_us_per_s / 1e6 <= 0):
        raise ValueError('fallback_drift_us_per_s must be finite and yield a positive clock slope (> -1000000).')

    stim_channel = resolve_stim_channel(raw, stim_channel)
    trigger_to_wav = {int(code): wav_file for code, wav_file in onset_trigger_to_wav.items()}
    if len(trigger_to_wav) != len(onset_trigger_to_wav):
        raise ValueError('onset_trigger_to_wav contains trigger keys that collapse to the same integer value.')
    onset_codes = set(trigger_to_wav)
    if end_triggers is None:
        end_codes = set()
    elif isinstance(end_triggers, Integral):
        end_codes = {int(end_triggers)}
    else:
        end_codes = {int(code) for code in cast('Sequence[int]', end_triggers)}
    if not end_codes and not infer_missing_ends:
        raise ValueError('end_triggers must contain at least one trigger value unless infer_missing_ends=True.')
    overlap = onset_codes & end_codes
    if overlap:
        raise ValueError(f'Trigger values cannot be both onset and end triggers: {sorted(overlap)}')

    events = mne.find_events(
        raw,
        stim_channel=stim_channel,
        shortest_event=1,
        consecutive=True,
        initial_event=True,
        verbose=False,
    )

    trials: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for event_sample, _, raw_event_value in events:
        sample = int(event_sample)
        event_value = int(raw_event_value)

        if event_value in onset_codes:
            if current is not None:
                if not infer_missing_ends:
                    raise RuntimeError('New audio onset before the preceding end trigger.')
                trials.append(_infer_audio_trial_end(current, raw, sample, base_audio_path, fallback_drift_us_per_s))
            current = {
                'onset_code': event_value,
                'wav_file': str(trigger_to_wav[event_value]),
                'onset_sample': sample,
            }
        elif current is not None and event_value in end_codes:
            current['end_code'] = event_value
            current['end_sample'] = sample
            trials.append(current)
            current = None

    if current is not None:
        if infer_missing_ends:
            trials.append(
                _infer_audio_trial_end(
                    current, raw, int(raw.first_samp + raw.n_times), base_audio_path, fallback_drift_us_per_s
                )
            )
        else:
            warnings.warn('The final audio trial has no end trigger and was ignored.', stacklevel=2)

    return trials


def _make_sync_envelope(
    signal: NDArray[Any],
    sfreq: float,
    *,
    target_sfreq: float,
    band: tuple[float, float],
    envelope_lowpass: float,
) -> NDArray[np.float64]:
    if sfreq <= 0 or target_sfreq <= 0:
        raise ValueError('Sampling frequencies must be positive.')
    signal = np.asarray(signal, dtype=float)
    signal -= np.mean(signal)
    if np.std(signal) < VARIANCE_EPSILON:
        raise ValueError('Audio signal has essentially zero variance.')

    nyquist = sfreq / 2.0
    low = float(band[0])
    high = min(float(band[1]), 0.99 * nyquist)
    if low <= 0 or low >= high:
        raise ValueError(f'Invalid audio band {band} for sfreq={sfreq}.')

    signal = sosfiltfilt(butter(4, [low, high], btype='bandpass', fs=sfreq, output='sos'), signal)
    envelope = np.abs(hilbert(signal))

    if envelope_lowpass <= 0 or envelope_lowpass >= min(nyquist, target_sfreq / 2.0):
        raise ValueError('envelope_lowpass must be positive and below both input and target Nyquist frequencies.')
    envelope = sosfiltfilt(butter(4, envelope_lowpass, btype='lowpass', fs=sfreq, output='sos'), envelope)

    if not np.isclose(sfreq, target_sfreq):
        envelope = resample_poly_exact(envelope, sfreq, target_sfreq)

    envelope -= np.mean(envelope)
    scale = np.std(envelope)
    if scale < VARIANCE_EPSILON:
        raise ValueError('Envelope has essentially zero variance.')
    return np.asarray(envelope / scale, dtype=float)


def _normalized_xcorr_valid(signal: NDArray[np.float64], template: NDArray[np.float64]) -> NDArray[np.float64]:
    signal = np.asarray(signal, dtype=float)
    template = np.asarray(template, dtype=float)
    n_template = len(template)
    if len(signal) < n_template:
        raise ValueError('signal must be longer than template.')

    template_centered = template - np.mean(template)
    template_energy = np.sum(template_centered**2)
    if template_energy < ENERGY_EPSILON:
        return np.full(len(signal) - n_template + 1, np.nan)

    numerator = correlate(signal, template_centered, mode='valid', method='fft')
    cumulative = np.concatenate(([0.0], np.cumsum(signal)))
    cumulative_squared = np.concatenate(([0.0], np.cumsum(signal**2)))
    window_sum = cumulative[n_template:] - cumulative[:-n_template]
    window_sum_squared = cumulative_squared[n_template:] - cumulative_squared[:-n_template]
    window_energy = np.maximum(window_sum_squared - window_sum**2 / n_template, 0.0)
    denominator = np.sqrt(window_energy * template_energy)

    result = np.full_like(numerator, np.nan, dtype=float)
    good = denominator > ENERGY_EPSILON
    result[good] = numerator[good] / denominator[good]
    return result


def _estimate_window_lag(
    recorded: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    center_s: float,
    sfreq: float,
    window_s: float,
    max_lag_s: float,
) -> tuple[float | None, float | None]:
    n_window = int(round(window_s * sfreq))
    n_max_lag = int(round(max_lag_s * sfreq))
    reference_start = int(round((center_s - window_s / 2) * sfreq))
    reference_stop = reference_start + n_window
    recorded_start = reference_start - n_max_lag
    recorded_stop = reference_stop + n_max_lag

    if reference_start < 0 or recorded_start < 0 or reference_stop > len(reference) or recorded_stop > len(recorded):
        return None, None

    reference_segment = reference[reference_start:reference_stop]
    recorded_segment = recorded[recorded_start:recorded_stop]
    if np.std(reference_segment) < VARIANCE_EPSILON or np.std(recorded_segment) < VARIANCE_EPSILON:
        return None, None

    correlations = _normalized_xcorr_valid(recorded_segment, reference_segment)
    if not np.any(np.isfinite(correlations)):
        return None, None

    peak_index = int(np.nanargmax(correlations))
    peak_correlation = float(correlations[peak_index])
    peak_position = float(peak_index)
    if 0 < peak_index < len(correlations) - 1:
        left, center, right = correlations[peak_index - 1 : peak_index + 2]
        denominator = left - 2 * center + right
        if abs(denominator) > VARIANCE_EPSILON:
            delta = 0.5 * (left - right) / denominator
            if abs(delta) <= 1:
                peak_position += delta

    lag_s = (peak_position - n_max_lag) / sfreq
    return float(lag_s), peak_correlation


def _alignment_quality_warnings(
    alignment: Mapping[str, Any],
    anchor_lags: NDArray[np.float64],
    inlier_times: NDArray[np.float64],
    candidate_times: NDArray[np.float64],
    *,
    max_lag_s: float,
    sync_sfreq: float,
    warn_residual_rms_ms: float,
) -> list[str]:
    """Flag poor fit precision, a limited search range, or limited time coverage."""
    reasons = []
    if alignment['residual_rms_ms'] > warn_residual_rms_ms:
        reasons.append(
            f'inlier residual RMS {alignment["residual_rms_ms"]:.3f} ms exceeds '
            f'the warning threshold of {warn_residual_rms_ms:g} ms'
        )

    # Match the actual search grid, including a requested range below one sample.
    lag_limit_s = round(max_lag_s * sync_sfreq) / sync_sfreq
    margin_s = max(1.0 / sync_sfreq, LAG_LIMIT_MARGIN_FRACTION * lag_limit_s)
    endpoint_lags = alignment['offset_s'] + (alignment['clock_slope'] - 1.0) * np.array(
        [0.0, alignment['wav_duration_s']]
    )
    if lag_limit_s == 0:
        reasons.append('max_lag_s permits no nonzero lag search at the synchronization sampling rate')
    elif np.any(np.abs(anchor_lags) >= lag_limit_s - margin_s) or np.any(
        np.abs(endpoint_lags) >= lag_limit_s - margin_s
    ):
        reasons.append(
            f'accepted window lags or the fitted lag over the full stimulus approach or exceed '
            f'the +/-{lag_limit_s:g} s search limit (max_lag_s={max_lag_s:g})'
        )

    coverage = float(np.ptp(inlier_times) / np.ptp(candidate_times))
    if coverage < MIN_ANCHOR_TIME_COVERAGE:
        reasons.append(f'inlier anchors span only {coverage:.1%} of the evaluated trial duration')
    return reasons


@function_defaults('alignment')
def estimate_raw_wav_alignment(  # noqa: C901, PLR0912, PLR0915
    raw_trial: mne.io.BaseRaw,
    wav_file: str | Path,
    *,
    audio_channels: Sequence[str],
    sync_sfreq: float = 500.0,
    audio_band: tuple[float, float] = (80.0, 400.0),
    envelope_lowpass: float = 30.0,
    window_s: float = 10.0,
    step_s: float = 5.0,
    max_lag_s: float = 0.250,
    min_corr: float = 0.30,
    min_anchors: int = 5,
    reject_outliers: bool = True,
    warn_residual_rms_ms: float = 10.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Estimate clock offset and drift by matching recorded audio to a WAV.

    Defaults below describe the general profile; omitted settings use the
    active alignment defaults.

    Parameters
    ----------
    raw_trial : mne.io.BaseRaw
        Recording cropped to one trial, starting at its onset trigger.
    wav_file : str | pathlib.Path
        Reference WAV path, absolute or relative to the working directory.
    audio_channels : sequence of str
        Recorded audio channel names. All recorded/WAV channel pairs are tried;
        the pair with the highest median window correlation is used.
    sync_sfreq : float, default=500.0
        Sampling rate in Hz for the envelopes used to estimate alignment.
    audio_band : tuple of float, default=(80.0, 400.0)
        Band-pass limits in Hz before envelope extraction. The upper limit is
        capped below each signal's Nyquist frequency.
    envelope_lowpass : float, default=30.0
        Envelope low-pass cutoff in Hz; must be below both input and sync Nyquist
        frequencies. Independent of EpochStimuls's WAV feature cutoff.
    window_s : float, default=10.0
        Duration in seconds of each window used to estimate a lag.
    step_s : float, default=5.0
        Spacing in seconds between window centers.
    max_lag_s : float, default=0.25
        Search half-width in seconds around zero lag for every window.
        Must cover initial offset plus drift across the full WAV; see Notes.
    min_corr : float, default=0.30
        Minimum normalized correlation, from -1 to 1, for accepting a window
        lag as a synchronization anchor.
    min_anchors : int, default=5
        Required anchor count before and after outlier rejection; at least two.
    reject_outliers : bool, default=True
        Reject lag outliers using a robust preliminary fit before fitting drift.
    warn_residual_rms_ms : float, default=10.0
        Positive residual-RMS threshold in milliseconds for a quality warning.
        Exceeding it does not reject the fit.
    verbose : bool, default=True
        Log fitted offset, drift, and residual RMS. Quality warnings always emit.

    Returns
    -------
    alignment : dict
        Clock mapping (``offset_s``, ``clock_slope``, ``drift_us_per_s``), WAV
        duration and sample rates, selected channels, anchor counts, residuals,
        correlations, and ``quality_warnings``. Usable by ``apply_raw_wav_alignment``.

    Notes
    -----
    The fitted mapping is ``t_raw = offset_s + clock_slope * t_wav``, where
    raw time is relative to the trial start and ``clock_slope = 1 + drift / 1e6``.
    This function estimates alignment without modifying or resampling the input.

    The lag search stays centered on zero; it does not follow accumulated drift.
    Choose ``max_lag_s`` above the largest absolute
    ``offset_s + drift_us_per_s * t / 1e6`` over the WAV, with margin. For example,
    +500 microseconds/s adds 0.63 s over 1260 s, so ``max_lag_s=1.0`` allows margin
    at zero initial offset. Missing-end inference does not widen this range.

    ``RuntimeWarning`` flags excessive residual RMS, lags near the search limit
    (within 5% or one sync sample), or inlier anchors covering less than half the
    evaluated time range. Warnings are returned in ``quality_warnings`` even with
    ``verbose=False``; the fit is still returned. Residual RMS measures agreement
    with the fitted line, not absolute timing accuracy. Inspect the audio match
    before adjusting the search range or warning threshold.
    """
    if not isinstance(raw_trial, mne.io.BaseRaw):
        raise TypeError('raw_trial must be an MNE Raw object.')
    if isinstance(audio_channels, str) or not audio_channels:
        raise ValueError('audio_channels must contain at least one channel name.')
    if sync_sfreq <= 0 or window_s <= 0 or step_s <= 0:
        raise ValueError('sync_sfreq, window_s, and step_s must be positive.')
    if max_lag_s < 0:
        raise ValueError('max_lag_s must be non-negative.')
    if not -1 <= min_corr <= 1:
        raise ValueError('min_corr must be between -1 and 1.')
    if min_anchors < MIN_ANCHORS_FOR_DRIFT:
        raise ValueError('min_anchors must be at least 2 to estimate clock drift.')
    if not np.isfinite(warn_residual_rms_ms) or warn_residual_rms_ms <= 0:
        raise ValueError('warn_residual_rms_ms must be finite and positive.')

    raw_sfreq = float(raw_trial.info['sfreq'])
    missing = [channel for channel in audio_channels if channel not in raw_trial.ch_names]
    if missing:
        raise ValueError(f'Missing audio channels: {missing}')

    recorded = raw_trial.get_data(picks=list(audio_channels))
    wav_path = Path(wav_file)
    if not wav_path.is_file():
        raise FileNotFoundError(f'WAV file does not exist: {wav_path}')
    wav, wav_sfreq = librosa.load(wav_path, sr=None, mono=False)
    wav = np.asarray(wav, dtype=float)
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    wav_sfreq = float(wav_sfreq)
    wav_duration_s = wav.shape[-1] / wav_sfreq

    recorded_sync: list[tuple[int, NDArray[np.float64]]] = []
    for channel_index, channel in enumerate(recorded):
        try:
            envelope = _make_sync_envelope(
                channel,
                raw_sfreq,
                target_sfreq=sync_sfreq,
                band=audio_band,
                envelope_lowpass=envelope_lowpass,
            )
        except ValueError:
            continue
        recorded_sync.append((channel_index, envelope))
    if not recorded_sync:
        raise RuntimeError('None of the recorded audio channels contains a usable signal.')

    wav_sync: list[tuple[int, NDArray[np.float64]]] = []
    for channel_index, channel in enumerate(wav):
        try:
            envelope = _make_sync_envelope(
                channel,
                wav_sfreq,
                target_sfreq=sync_sfreq,
                band=audio_band,
                envelope_lowpass=envelope_lowpass,
            )
        except ValueError:
            continue
        wav_sync.append((channel_index, envelope))
    if not wav_sync:
        raise RuntimeError('None of the WAV channels contains a usable signal.')

    usable_duration = (
        min(
            max(len(envelope) for _, envelope in recorded_sync),
            max(len(envelope) for _, envelope in wav_sync),
        )
        / sync_sfreq
    )
    first_center = window_s / 2 + max_lag_s
    last_center = usable_duration - window_s / 2 - max_lag_s
    if last_center < first_center:
        raise RuntimeError('Trial is too short for the requested synchronization settings.')
    centers = np.arange(first_center, last_center + 1e-12, step_s, dtype=np.float64)

    best_pair: tuple[int, int] | None = None
    best_score = -np.inf
    best_times: NDArray[np.float64] | None = None
    best_lags: NDArray[np.float64] | None = None
    best_correlations: NDArray[np.float64] | None = None
    for recorded_index, recorded_envelope in recorded_sync:
        for wav_index, wav_envelope in wav_sync:
            pair_times: list[float] = []
            pair_lags: list[float] = []
            pair_correlations: list[float] = []
            for center in centers:
                lag, correlation = _estimate_window_lag(
                    recorded_envelope,
                    wav_envelope,
                    center_s=float(center),
                    sfreq=sync_sfreq,
                    window_s=window_s,
                    max_lag_s=max_lag_s,
                )
                if lag is not None and correlation is not None and np.isfinite(correlation):
                    pair_times.append(float(center))
                    pair_lags.append(lag)
                    pair_correlations.append(correlation)
            if pair_correlations:
                score = float(np.median(pair_correlations))
                if score > best_score:
                    best_pair = (recorded_index, wav_index)
                    best_score = score
                    best_times = np.asarray(pair_times, dtype=float)
                    best_lags = np.asarray(pair_lags, dtype=float)
                    best_correlations = np.asarray(pair_correlations, dtype=float)

    if best_pair is None or best_times is None or best_lags is None or best_correlations is None:
        raise RuntimeError('No usable recorded/WAV channel pair could be evaluated.')
    best_recorded_index, best_wav_index = best_pair

    accepted = best_correlations >= min_corr
    anchor_times = best_times[accepted]
    anchor_lags = best_lags[accepted]
    anchor_correlations = best_correlations[accepted]
    if len(anchor_times) < min_anchors:
        raise RuntimeError(
            f'Only {len(anchor_times)} synchronization anchors were accepted; required {min_anchors}. '
            f'Best median correlation: {best_score:.3f}.'
        )

    initial_drift, initial_offset, _, _ = theilslopes(anchor_lags, anchor_times)
    initial_residuals = anchor_lags - (initial_offset + initial_drift * anchor_times)
    inliers = np.ones(len(anchor_times), dtype=bool)
    outlier_threshold_s: float | None = None
    if reject_outliers:
        residual_median = np.median(initial_residuals)
        mad = np.median(np.abs(initial_residuals - residual_median))
        outlier_threshold_s = max(2.0 / sync_sfreq, 5.0 * 1.4826 * mad)
        inliers = np.abs(initial_residuals - residual_median) <= outlier_threshold_s
    if np.sum(inliers) < min_anchors:
        raise RuntimeError(f'Only {np.sum(inliers)} anchors remain after outlier rejection; required {min_anchors}.')

    drift, offset = np.polyfit(anchor_times[inliers], anchor_lags[inliers], 1)
    clock_slope = 1.0 + drift
    if clock_slope <= 0:
        raise RuntimeError(f'Invalid clock slope: {clock_slope}')
    residuals = anchor_lags - (offset + drift * anchor_times)

    alignment: dict[str, Any] = {
        'alignment_method': 'audio',
        'offset_s': float(offset),
        'clock_slope': float(clock_slope),
        'raw_sfreq': raw_sfreq,
        'wav_sfreq': wav_sfreq,
        'wav_duration_s': float(wav_duration_s),
        'drift_us_per_s': float(drift * 1e6),
        'total_drift_ms': float(drift * wav_duration_s * 1000),
        'residual_rms_ms': float(np.sqrt(np.mean(residuals[inliers] ** 2)) * 1000),
        'residual_max_ms': float(np.max(np.abs(residuals[inliers])) * 1000),
        'median_correlation': float(np.median(anchor_correlations[inliers])),
        'n_anchors': len(anchor_times),
        'n_anchor_inliers': int(np.sum(inliers)),
        'selected_recorded_channel': audio_channels[best_recorded_index],
        'selected_wav_channel': int(best_wav_index),
    }
    alignment['quality_warnings'] = _alignment_quality_warnings(
        alignment,
        anchor_lags,
        anchor_times[inliers],
        centers,
        max_lag_s=max_lag_s,
        sync_sfreq=sync_sfreq,
        warn_residual_rms_ms=warn_residual_rms_ms,
    )
    if alignment['quality_warnings']:
        warnings.warn(
            f'Potentially unreliable audio alignment for {wav_path.name} '
            f'({wav_duration_s:g} s stimulus): ' + '; '.join(alignment['quality_warnings']) + '. '
            'Inspect the audio match and consider increasing max_lag_s, especially for long stimuli. '
            'The fitted alignment is still returned.',
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        mne.utils.logger.info(
            f'Alignment {wav_path.name}: offset={offset * 1000:+.3f} ms, '
            f'drift={drift * 1e6:+.3f} us/s, residual RMS={alignment["residual_rms_ms"]:.3f} ms'
        )
    return alignment


@function_defaults('stimuli')
def assume_raw_wav_alignment(
    wav_file: str | Path,
    raw_sfreq: float,
    *,
    fallback_drift_us_per_s: float = DEFAULT_FALLBACK_DRIFT_US_PER_S,
) -> dict[str, Any]:
    """Construct a clock correction from WAV duration and an assumed drift.

    Defaults below describe the general profile; omitted drift uses the active
    stimulus defaults.

    Parameters
    ----------
    wav_file : str | pathlib.Path
        Reference WAV path, absolute or relative to the working directory.
    raw_sfreq : float
        Recording sampling rate in Hz, stored for later alignment validation.
    fallback_drift_us_per_s : float, default=0.0
        Assumed drift in microseconds per second; must be finite and greater
        than -1,000,000. Salzburg defaults to +499.

    Returns
    -------
    alignment : dict
        Assumed clock slope, zero offset, WAV duration, and recording sample
        rate. Fit-quality metrics are ``None`` and anchor counts are zero.

    Notes
    -----
    The trial onset is assumed to coincide with WAV time zero. Positive drift
    lengthens the trial on the recording clock:
    ``t_raw = (1 + drift / 1e6) * t_wav``. No recorded audio is measured.
    """
    slope = 1.0 + fallback_drift_us_per_s / 1e6
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError('fallback_drift_us_per_s must be finite and yield a positive clock slope (> -1000000).')
    wav_path = Path(wav_file)
    if not wav_path.is_file():
        raise FileNotFoundError(f'WAV file does not exist: {wav_path}')
    duration_s = float(librosa.get_duration(path=wav_path))
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f'WAV duration must be finite and positive: {wav_path}')
    return {
        'alignment_method': 'assumed_drift',
        'offset_s': 0.0,
        'clock_slope': float(slope),
        'raw_sfreq': float(raw_sfreq),
        'wav_duration_s': duration_s,
        'drift_us_per_s': float(fallback_drift_us_per_s),
        'total_drift_ms': float(fallback_drift_us_per_s * duration_s / 1000),
        'residual_rms_ms': None,
        'residual_max_ms': None,
        'median_correlation': None,
        'n_anchors': 0,
        'n_anchor_inliers': 0,
    }


def apply_raw_wav_alignment(  # noqa: C901, PLR0912, PLR0915
    raw: mne.io.BaseRaw,
    onset_sample: int,
    alignment: Mapping[str, Any],
    *,
    preserve_annotations: bool = True,
    padding_s: tuple[float, float] = (0.0, 0.0),
    verbose: bool = True,
) -> mne.io.BaseRaw:
    """Extract and clock-correct one trial using MNE's device realignment.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording covering the full trial and requested padding.
        The input is not modified.
    onset_sample : int
        Trial onset in raw sample coordinates, including ``raw.first_samp``.
    alignment : mapping
        Clock correction with ``offset_s``, ``clock_slope``, ``wav_duration_s``,
        and ``raw_sfreq``, as returned by the estimate/assume alignment helpers.
    preserve_annotations : bool, default=True
        Retain annotations and transform their timing with the data.
        ``False`` removes them from the extracted segment.
    padding_s : tuple of float, default=(0.0, 0.0)
        Non-negative ``(before, after)`` margins in seconds on the corrected WAV
        clock. Uses real recording samples to allow later delay correction.
    verbose : bool, default=True
        Emit MNE processing messages.

    Returns
    -------
    segment : mne.io.BaseRaw
        Loaded copy resampled onto the WAV clock at the original sampling rate,
        including padding. No WAV feature channels or physical delay are added.

    Notes
    -----
    Raises if the recording cannot cover the full WAV plus padding or if its
    sample rate differs from ``alignment['raw_sfreq']``.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError('raw must be an MNE Raw object.')
    raw_sfreq = float(raw.info['sfreq'])
    expected_sfreq = float(alignment['raw_sfreq'])
    if not np.isclose(raw_sfreq, expected_sfreq):
        raise ValueError(f'Raw sampling frequency {raw_sfreq} does not match alignment {expected_sfreq}.')

    onset_sample = int(onset_sample)
    onset_index = onset_sample - raw.first_samp
    if onset_index < 0 or onset_index >= raw.n_times:
        raise ValueError('onset_sample is outside the supplied Raw object.')

    offset_s = float(alignment['offset_s'])
    clock_slope = float(alignment['clock_slope'])
    wav_duration_s = float(alignment['wav_duration_s'])
    if not np.isfinite(offset_s) or not np.isfinite(clock_slope) or not np.isfinite(wav_duration_s):
        raise ValueError('Alignment offset, clock slope, and WAV duration must be finite.')
    if clock_slope <= 0 or wav_duration_s <= 0:
        raise ValueError('Alignment clock slope and WAV duration must be positive.')

    if np.shape(padding_s) != (2,) or any(not np.isfinite(pad) or pad < 0 for pad in padding_s):
        raise ValueError('padding_s must contain two finite, non-negative durations.')
    pad_before, pad_after = (int(round(pad * raw_sfreq)) for pad in padding_s)
    wav_start_s = -pad_before / raw_sfreq
    wav_stop_s = wav_duration_s + pad_after / raw_sfreq
    source_start_float = onset_index + (offset_s + clock_slope * wav_start_s) * raw_sfreq
    source_end_float = onset_index + (offset_s + clock_slope * wav_stop_s) * raw_sfreq
    if source_start_float < 0:
        raise RuntimeError(f'Required trial begins {-source_start_float / raw_sfreq:.6f} s before the Raw object.')
    if source_end_float > raw.n_times:
        raise RuntimeError(
            f'Required trial extends {(source_end_float - raw.n_times) / raw_sfreq:.6f} s beyond the Raw object.'
        )

    source_start_index = int(np.floor(source_start_float))
    source_end_index = min(int(np.ceil(source_end_float)) + 4, raw.n_times)
    segment = raw.copy().crop(
        tmin=source_start_index / raw_sfreq,
        tmax=(source_end_index - 1) / raw_sfreq,
    )
    segment.load_data()
    target_n_samples = int(round(wav_duration_s * raw_sfreq)) + pad_before + pad_after
    if target_n_samples < MIN_ANCHORS_FOR_DRIFT:
        raise RuntimeError('WAV duration is too short to construct an aligned Raw object.')
    reference = mne.io.RawArray(
        np.zeros((1, target_n_samples)),
        mne.create_info(['WAV reference'], raw_sfreq, ['misc']),
        verbose=False,
    )
    if not preserve_annotations:
        segment.set_annotations(mne.Annotations([], [], []))

    # realign_raw expects shared times relative to the starts of its two Raw
    # objects. Supplying points on the fitted line delegates cropping,
    # resampling, and annotation correction to MNE without fitting the noisy
    # cross-correlation anchors a second time.
    reference_times = np.linspace(0.0, reference.times[-1], 20)
    wav_times = reference_times + wav_start_s
    trigger_time_in_segment = (onset_index - source_start_index) / raw_sfreq
    recorded_times = trigger_time_in_segment + offset_s + clock_slope * wav_times

    # MNE 1.8 resolves each channel name as a string pick while checking for
    # NaNs. Names such as ``audio`` or ``eeg`` are ambiguous with channel-type
    # selectors, so use collision-free temporary names during realignment.
    original_names = list(segment.ch_names)
    temporary_names = [f'AKALIGN{index:04d}' for index in range(len(original_names))]
    while set(temporary_names) & set(original_names):
        temporary_names = [f'X{name}' for name in temporary_names]
    rename_to_temporary = dict(zip(original_names, temporary_names, strict=True))
    segment.rename_channels(rename_to_temporary)
    try:
        mne.preprocessing.realign_raw(
            reference,
            segment,
            t_raw=reference_times,
            t_other=recorded_times,
            verbose=verbose,
        )
    finally:
        segment.rename_channels({temporary: original for original, temporary in rename_to_temporary.items()})
    if segment.n_times != target_n_samples:
        raise RuntimeError(
            f'Clock correction produced {segment.n_times} samples; {target_n_samples} samples are required.'
        )
    return segment


def summarize_alignments(trials: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Summarize alignment diagnostics across trials.

    Parameters
    ----------
    trials : sequence of mapping
        Per-trial alignment results. Missing or ``None`` metrics are omitted
        independently, allowing measured and assumed alignments to be mixed.

    Returns
    -------
    summary : dict of str to dict
        ``n``, ``mean``, ``sd``, ``min``, and ``max`` per available metric.
        Offset is converted to milliseconds; other metrics retain their units.

    Notes
    -----
    Standard deviations use ``ddof=0``. Counts and moments support pooling
    across recordings, even when JSON export truncates the per-trial lists.
    """
    metrics = {
        'offset_ms': ('offset_s', 1000.0),
        'clock_slope': ('clock_slope', 1.0),
        'drift_us_per_s': ('drift_us_per_s', 1.0),
        'total_drift_ms': ('total_drift_ms', 1.0),
        'residual_rms_ms': ('residual_rms_ms', 1.0),
        'residual_max_ms': ('residual_max_ms', 1.0),
        'median_correlation': ('median_correlation', 1.0),
        'n_anchor_inliers': ('n_anchor_inliers', 1.0),
    }
    summary = {}
    for name, (key, scale) in metrics.items():
        # Assumed drift has no measured residuals or correlations.
        values = np.asarray([float(trial[key]) * scale for trial in trials if trial.get(key) is not None], dtype=float)
        if values.size:
            summary[name] = {
                'n': int(values.size),
                'mean': float(np.mean(values)),
                'sd': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
    return summary
