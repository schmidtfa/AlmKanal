"""Clean continuous EOG using NeuroKit, absolute-z peaks and PCHIP."""

from collections.abc import Iterator
from typing import Any

import mne
import neurokit2 as nk
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

# from scipy.stats import linregress

# algorithm parameters
N_EOG_CHANNELS = 2
MIN_USABLE_SAMPLES = 2
MIN_NR_BLINKS = 5
MIN_ASSIGNMENT_EVENTS = 5

MAX_INTERPOLATION_WINDOW_S = 0.5
MIN_SAMPLING_RATE_HZ = 15.0
BLINK_PEAK_DISTANCE_S = 0.3
BLINK_SEARCH_RADIUS_S = 0.05


def validate_eog_channels(
    raw: mne.io.BaseRaw,
    eog_channels: tuple[str, str],
) -> None:
    """Validate the selected EOG channels."""
    if isinstance(eog_channels, str) or len(eog_channels) != N_EOG_CHANNELS or len(set(eog_channels)) != N_EOG_CHANNELS:
        raise ValueError('Supply two distinct EOG channel names: vertical, horizontal.')

    for channel in eog_channels:
        if channel not in raw.ch_names:
            raise ValueError(f'Missing EOG channel: {channel}')
        if channel in raw.info['bads']:
            raise ValueError(f'EOG channel is already marked bad: {channel}')

    for name in ['vertical', 'horizontal']:
        if name in raw.ch_names and name not in eog_channels:
            raise ValueError(f'Cannot rename EOG: {name} already names another channel.')


def validate_eog_inputs(
    raw: mne.io.BaseRaw,
    eog_channels: tuple[str, str],
    find_blinks: bool,
    interpolate_blinks: bool,
    check_and_correct_eog_polarity: bool,
    check_and_correct_eog_channel_assignment: bool,
    peak_threshold: float,
    blink_window_ms: float,
    flat_threshold_v: float,
    max_amplitude_v: float | None,
    max_bad_fraction: float,
) -> None:
    """Validate the recording and EOG preprocessing parameters."""
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError('Pass one continuous MNE Raw recording.')

    validate_eog_channels(raw, eog_channels)

    for value in [peak_threshold, blink_window_ms, flat_threshold_v]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Peak threshold, blink window and flat threshold must be positive and finite.')
    if max_amplitude_v is not None and (not np.isfinite(max_amplitude_v) or max_amplitude_v <= 0):
        raise ValueError('max_amplitude_v must be positive and finite, or None.')
    if not 0 <= max_bad_fraction <= 1:
        raise ValueError('max_bad_fraction must lie between 0 and 1.')
    if not find_blinks and (
        interpolate_blinks or check_and_correct_eog_polarity or check_and_correct_eog_channel_assignment
    ):
        raise ValueError('Interpolation and automatic checks require find_blinks=True.')
    sfreq = float(raw.info['sfreq'])
    if find_blinks and sfreq <= MIN_SAMPLING_RATE_HZ:
        raise ValueError('NeuroKit detection filtering requires sfreq > 15 Hz.')


def get_intervals(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """Find starts and exclusive stops of consecutive True samples."""
    changes = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return zip(starts.tolist(), stops.tolist())


def get_bad_mask(
    raw: mne.io.BaseRaw,
    signals: np.ndarray,
    flat_threshold_v: float,
    max_amplitude_v: float | None,
) -> np.ndarray:
    """Identify unusable EOG samples from annotations and signal quality."""

    bad_mask = np.zeros(raw.n_times, dtype=bool)
    sfreq = raw.info['sfreq']

    # include existing BAD/EDGE annotations
    for annotation in raw.annotations:
        if annotation['description'].lower().startswith(('bad', 'edge')):
            onset = annotation['onset'] - raw.first_time
            end = onset + annotation['duration']

            start = int(np.floor(np.round(onset * sfreq, 7)))
            stop = int(np.ceil(np.round(end * sfreq, 7)))

            if annotation['duration'] == 0:
                stop = start + 1

            start = min(raw.n_times, max(0, start))
            stop = min(raw.n_times, max(0, stop))
            bad_mask[start:stop] = True

    # check signal quality in one-second windows
    window_samples = max(2, round(sfreq))

    for start in range(0, signals.shape[1], window_samples):
        stop = min(signals.shape[1], start + window_samples)

        usable = ~bad_mask[start:stop]

        if usable.sum() < MIN_USABLE_SAMPLES:
            continue

        for signal in signals:
            segment = signal[start:stop][usable]
            amplitude = np.ptp(segment)

            if amplitude <= flat_threshold_v or (max_amplitude_v is not None and amplitude > max_amplitude_v):
                bad_mask[start:stop] = True
                break

    return bad_mask


def find_blinks_eog(
    signal: np.ndarray,
    sfreq: float,
    bad_mask: np.ndarray,
    peak_threshold: float,
    blink_window_ms: float,
) -> dict[str, Any]:
    """Find blinks in EOG signal using NeuroKit filtering, z-scoring, then absolute-z peaks.
    NeuroKit filters the signal (= detection signal), which is used for peak detection.
    Returns local peak indices, signed z-scores and a merged blink_mask, which is applied to the original
    EOG signal for interpolation. Recording-edge windows areleft unfilled by interpolation and added to bad_mask.
    Windows crossing excluded data are rejected.
    """
    detection = np.full(signal.size, np.nan)
    for start, stop in get_intervals(~bad_mask):
        if stop <= start:
            continue

        detection[start:stop] = nk.eog_clean(
            signal[start:stop],
            sampling_rate=sfreq,
        )

    usable = np.isfinite(detection)

    if usable.sum() < MIN_USABLE_SAMPLES:
        raise ValueError('Not enough usable EOG for blink detection.')

    mean = detection[usable].mean()
    std = detection[usable].std()

    if not np.isfinite(std) or std == 0:
        raise ValueError('Not enough non-flat EOG for blink detection.')

    signal_z = np.full(signal.size, np.nan)
    signal_z[usable] = (detection[usable] - mean) / std

    candidate_peaks = []

    for start, stop in get_intervals(usable):
        segment_z = signal_z[start:stop]

        local_peaks, _ = find_peaks(
            np.abs(segment_z),
            height=peak_threshold,
            distance=max(1, (round(BLINK_PEAK_DISTANCE_S * sfreq))),
        )

        candidate_peaks.extend((local_peaks + start).tolist())

    peaks = np.asarray(candidate_peaks, dtype=int)

    window_s = blink_window_ms / 1000
    samples_before = round(window_s / 3 * sfreq)
    samples_after = round(window_s * 2 / 3 * sfreq)
    blink_mask = np.zeros(signal.size, dtype=bool)
    accepted_peaks = []
    for peak in peaks:
        start = max(0, peak - samples_before)
        stop = min(signal.size, peak + samples_after)
        if usable[start:stop].all():
            blink_mask[start:stop] = True
            accepted_peaks.append(peak)
    accepted_peaks_array = np.asarray(accepted_peaks, dtype=int)
    return {
        'accepted_peaks': accepted_peaks_array,  # for automatic channel-assignment & saving of blink-candidates
        'signed_z': signal_z[accepted_peaks_array],  # for polarity-assessment
        'blink_mask': blink_mask,  # for interpolation & annotation
        'detection': detection,  # for automatic channel-assignment & diagnostic plots
    }


def check_channel_assignment(
    signals: np.ndarray,
    sfreq: float,
    bad_mask: np.ndarray,
    peak_threshold: float,
    blink_window_ms: float,
) -> tuple[int, dict[str, Any]]:
    """Suggest the vertical channel from shared blink-event amplitudes."""
    detections = []
    candidate_peaks = []
    for signal in signals:
        result = find_blinks_eog(signal, sfreq, bad_mask, peak_threshold, blink_window_ms)
        detection = result['detection']
        detection = detection - np.nanmedian(detection)  # center signal around zero
        detections.append(detection)
        candidate_peaks.extend(result['accepted_peaks'].tolist())

    events = np.unique(candidate_peaks)
    amplitudes = []
    radius = max(1, round(BLINK_SEARCH_RADIUS_S * sfreq))

    for peak in events:
        start = max(0, peak - radius)
        stop = min(signals.shape[1], peak + radius + 1)
        first = detections[0][start:stop]
        second = detections[1][start:stop]
        if np.isfinite(first).all() and np.isfinite(second).all():
            amplitudes.append([np.ptp(first), np.ptp(second)])

    if len(amplitudes) < MIN_ASSIGNMENT_EVENTS:
        raise ValueError('Too few candidate events to infer EOG channel assignment.')

    amplitudes_array = np.asarray(amplitudes)
    median_amplitudes = np.median(amplitudes_array, axis=0)

    vertical_idx = 0 if median_amplitudes[0] > median_amplitudes[1] else 1

    return vertical_idx, {
        'n_events': len(amplitudes),
        'median_amplitudes': median_amplitudes,
        'amplitudes': amplitudes,
    }

    # # fit regression: predict eog1 amplitudes from eog2 amplitudes
    # regression = linregress(
    #     amplitudes[:, 1],  # EOG2: predictor (X)
    #     amplitudes[:, 0],  # EOG1: outcome (Y)
    # )
    # slope = regression.slope
    # # determine vertical channel based on regression slope
    # if slope < 1:
    #     vertical_idx = 1  # EOG2
    # elif slope > 1:
    #     vertical_idx = 0  # EOG1
    # else:
    #     raise ValueError(
    #         'EOG assignment is ambiguous. Regression slope is 1.'
    #     )

    # return vertical_idx, {
    #     'n_events': len(amplitudes),
    #     'regression_slope': float(slope),
    #     'regression_intercept': float(regression.intercept),
    #     'regression_r': float(regression.rvalue),
    #     'amplitudes': amplitudes,
    # }


def process_eog_blinks(
    eog: mne.io.BaseRaw,
    find_blinks: bool,
    sfreq: float,
    bad_mask: np.ndarray,
    peak_threshold: float,
    blink_window_ms: float,
    check_and_correct_eog_polarity: bool,
    interpolate_blinks: bool,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Find blinks, check polarity, and optionally interpolate."""
    polarity_info: dict[str, Any] = {'checked': False, 'flipped': False}
    blink_info: dict[str, Any] = {'checked': False}
    interpolated_mask = np.zeros(eog.n_times, dtype=bool)
    unfilled_mask = np.zeros(eog.n_times, dtype=bool)
    if find_blinks:
        vertical = eog.get_data(picks=['vertical'])[0]
        blinks = find_blinks_eog(vertical, sfreq, bad_mask, peak_threshold, blink_window_ms)
        if check_and_correct_eog_polarity:
            polarity_info = correct_eog_polarity(eog, blinks)
        if interpolate_blinks:
            interpolated_mask, unfilled_mask = interpolate_eog_blinks(
                eog,
                blinks['blink_mask'],
                bad_mask,
            )
        else:
            unfilled_mask = blinks['blink_mask'].copy()
        add_eog_annotations(eog, blinks['blink_mask'], 'EOG_blink')
        add_eog_annotations(eog, interpolated_mask, 'EOG_interpolated_blink')
        add_eog_annotations(eog, unfilled_mask, 'BAD_EOG_unfilled_blink')
        blink_info = {
            'checked': True,
            'n_blinks': len(blinks['accepted_peaks']),
            'peak_times_s': (blinks['accepted_peaks'] / sfreq).tolist(),
            'signed_z_before_polarity': blinks['signed_z'].tolist(),
            'peak_threshold': peak_threshold,
            'blink_window_ms': blink_window_ms,
        }
    return (
        blink_info,
        polarity_info,
        interpolated_mask,
        unfilled_mask,
    )


def correct_eog_polarity(
    eog: mne.io.BaseRaw,
    blink_info: dict[str, Any],
) -> dict[str, Any]:
    """Make dominant vertical blink candidates positive;
    If vertical EOG has wrong polarity, multiply signal with (-1)."""

    signed_z = blink_info['signed_z']
    if len(signed_z) < MIN_NR_BLINKS:
        raise ValueError('Too few candidate blinks to infer vertical polarity.')

    positive_blinks_nr = np.sum(signed_z > 0)
    negative_blinks_nr = np.sum(signed_z < 0)

    if positive_blinks_nr > negative_blinks_nr:
        polarity_info = 'positive'
    elif negative_blinks_nr > positive_blinks_nr:
        polarity_info = 'negative'
    else:
        raise ValueError('EOG polarity is ambiguous. Equal number of positive and negative blinks.')
    if polarity_info == 'negative':
        eog.apply_function(
            lambda signal: -signal,
            picks=['vertical'],
        )

    return {
        'checked': True,
        'polarity': polarity_info,
        'n_positive_blinks': int(positive_blinks_nr),
        'n_negative_blinks': int(negative_blinks_nr),
    }


def interpolate_eog_blinks(
    eog: mne.io.BaseRaw,
    blink_mask: np.ndarray,
    bad_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """PCHIP interpolation, both EOG channels
    Merge overlapping windows; leave >0.5 s or edge windows unfilled. Anchors
    must be outside blink/bad intervals with at least one on either side;
    returns interpolated_mask and unfilled_mask
    """
    channels = ['vertical', 'horizontal']
    original = eog.get_data(picks=channels)
    cleaned = original.copy()
    sfreq = eog.info['sfreq']
    interpolated_mask = np.zeros(eog.n_times, dtype=bool)
    unfilled_mask = np.zeros(eog.n_times, dtype=bool)
    for start, stop in get_intervals(blink_mask):
        if (
            (stop - start) / sfreq > MAX_INTERPOLATION_WINDOW_S
            or start == 0
            or stop == eog.n_times
            or bad_mask[start:stop].any()
        ):
            unfilled_mask[start:stop] = True
            continue
        anchors = np.array([start - 2, start - 1, stop, stop + 1])
        anchors = anchors[(anchors >= 0) & (anchors < eog.n_times)]
        anchors = anchors[~blink_mask[anchors] & ~bad_mask[anchors]]
        if not (np.any(anchors < start) and np.any(anchors >= stop)):
            unfilled_mask[start:stop] = True
            continue
        for channel_idx in range(N_EOG_CHANNELS):
            interpolator = PchipInterpolator(
                anchors,
                original[channel_idx, anchors],
                extrapolate=False,
            )
            cleaned[channel_idx, start:stop] = interpolator(np.arange(start, stop))
        interpolated_mask[start:stop] = True
    eog.apply_function(lambda signals: cleaned, picks=channels, channel_wise=False)
    return (
        interpolated_mask,
        unfilled_mask,
    )  # interpolated_mask: True=interpol. blinks, unfilled_mask: True=interpol. would've violated QC measures


def add_eog_annotations(
    eog: mne.io.BaseRaw,
    mask: np.ndarray,
    description: str,
) -> None:
    """Append mask intervals using your original annotation time handling."""
    annotations = eog.annotations.copy()
    if annotations.orig_time is None:
        annotations.onset -= eog.first_time
        time_offset = 0
    else:
        time_offset = eog.first_time
    for start, stop in get_intervals(mask):
        annotations.append(
            onset=time_offset + start / eog.info['sfreq'],
            duration=(stop - start) / eog.info['sfreq'],
            description=description,
        )
    eog.set_annotations(annotations)


def clean_eog(
    raw: mne.io.BaseRaw,
    eog_channels: tuple[str, str] = ('EOG001', 'EOG002'),
    find_blinks: bool = True,
    interpolate_blinks: bool = True,
    peak_threshold: float = 3.0,
    blink_window_ms: float = 300.0,
    check_and_correct_eog_polarity: bool = True,
    check_and_correct_eog_channel_assignment: bool = True,
    flat_threshold_v: float = 1e-8,
    max_amplitude_v: float | None = None,
    max_bad_fraction: float = 0.3,
    l_freq: float | None = 0.1,
    h_freq: float | None = 40.0,
) -> dict:
    """Clean one recording and return {'data': eog, 'eog_info': metadata}.

    eog_channels gives the assumed (vertical, horizontal) order. Automatic
    assignment and polarity are optional experimental checks; ambiguity raises an error.
    Both checks and interpolation require find_blinks=True.

    Basic QC inspects one-second windows. flat_threshold_v is a peak-to-peak threshold in V (default 0.01 microvolts).
    max_amplitude_v optionally limits peak-to-peak amplitude in V. max_bad_fraction rejects the whole recording if
    the union of bad samples exceeds this fraction. Otherwise BAD annotations support later trial rejection.

    The input is copied; every channel and time reference is preserved. Only
    the selected EOGs are renamed/processed. Final MNE filtering uses l_freq and
    h_freq in Hz; set both to None to disable it. BAD/EDGE intervals are skipped.
    No resampling or epoching occurs.

    """
    validate_eog_inputs(
        raw,
        eog_channels,
        find_blinks,
        interpolate_blinks,
        check_and_correct_eog_polarity,
        check_and_correct_eog_channel_assignment,
        peak_threshold,
        blink_window_ms,
        flat_threshold_v,
        max_amplitude_v,
        max_bad_fraction,
    )
    sfreq = float(raw.info['sfreq'])
    eog = raw.copy().load_data()
    signals = eog.get_data(picks=list(eog_channels))
    if not np.isfinite(signals).all():
        raise ValueError('EOG contains NaN or infinite values.')

    bad_mask = get_bad_mask(eog, signals, flat_threshold_v, max_amplitude_v)
    add_eog_annotations(eog, bad_mask, 'BAD_EOG_quality')

    vertical_idx = 0
    assignment_info: dict[str, Any] = {'method': 'supplied_order'}
    if check_and_correct_eog_channel_assignment:
        vertical_idx, assignment_info = check_channel_assignment(
            signals,
            sfreq,
            bad_mask,
            peak_threshold,
            blink_window_ms,
        )
    mapping = {
        eog_channels[vertical_idx]: 'vertical',
        eog_channels[1 - vertical_idx]: 'horizontal',
    }
    eog.rename_channels(mapping)
    eog.set_channel_types({'vertical': 'eog', 'horizontal': 'eog'})
    assignment_info['mapping'] = mapping

    (
        blink_info,
        polarity_info,
        interpolated_mask,
        unfilled_mask,
    ) = process_eog_blinks(
        eog,
        find_blinks,
        sfreq,
        bad_mask,
        peak_threshold,
        blink_window_ms,
        check_and_correct_eog_polarity,
        interpolate_blinks,
    )

    excluded_mask = bad_mask | unfilled_mask

    excluded_fraction = float(excluded_mask.mean())

    if excluded_fraction > max_bad_fraction:
        raise ValueError(f'EOG rejected: {excluded_fraction:.1%} of samples are excluded after cleaning.')
    if l_freq is not None or h_freq is not None:
        eog.filter(l_freq=l_freq, h_freq=h_freq, picks=['vertical', 'horizontal'], skip_by_annotation=('bad', 'edge'))

    return {
        'data': eog,
        'eog_info': {
            'channel_assignment': assignment_info,
            'vertical_polarity': polarity_info,
            'blink_events': blink_info,
            'quality': {
                'excluded_fraction': excluded_fraction,
                'interpolated_fraction': float(interpolated_mask.mean()),
                'unfilled_blink_fraction': float(unfilled_mask.mean()),
                'max_bad_fraction': max_bad_fraction,
                'flat_threshold_v': flat_threshold_v,
                'max_amplitude_v': max_amplitude_v,
            },
            'filter': {'l_freq': l_freq, 'h_freq': h_freq, 'skip': ['bad', 'edge']},
            'sampling_rate': sfreq,
            'first_samp': int(eog.first_samp),
            'peak_time_reference': 'first available sample of this recording',
        },
    }
