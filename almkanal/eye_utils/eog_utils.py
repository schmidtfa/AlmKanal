"""Utilities for detecting, cleaning and interpolating EOG blinks.

Includes signal-quality assessment, channel assignment, polarity
correction and annotation of detected and interpolated blinks.
"""

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

MAX_INTERPOLATION_WINDOW_S = 2.0
MIN_SAMPLING_RATE_HZ = 15.0
BLINK_SEARCH_RADIUS_S = 0.05

# default preprocessing settings
DEFAULT_EOG_CHANNELS = ('EOG001', 'EOG002')
DEFAULT_FIND_BLINKS = True
DEFAULT_INTERPOLATE_BLINKS = True
DEFAULT_PEAK_THRESHOLD = 3.0
DEFAULT_BLINK_REL_HEIGHT = 0.8
DEFAULT_CHECK_POLARITY = True
DEFAULT_CHECK_CHANNEL_ASSIGNMENT = True
DEFAULT_FLAT_THRESHOLD_V = 1e-8
DEFAULT_MAX_BAD_FRACTION = 0.8
DEFAULT_L_FREQ = 0.1
DEFAULT_H_FREQ = 40.0


def validate_eog_channels(
    raw: mne.io.BaseRaw,
    eog_channels: tuple[str, str],
) -> None:
    """Validate the selected EOG channel names.

    Checks that both channels exist, are distinct, are not marked as bad
    and can be temporarily renamed.
    """

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
    blink_rel_height: float,
    flat_threshold_v: float,
    max_bad_fraction: float,
) -> None:
    """Validate the input recording and EOG preprocessing parameters.

    Checks the Raw object, channel selection, parameter ranges and
    sampling frequency.
    """

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError('Pass one continuous MNE Raw recording.')

    validate_eog_channels(raw, eog_channels)

    if not np.isfinite(peak_threshold) or peak_threshold <= 0:
        raise ValueError('peak_threshold must be positive and finite.')

    if not np.isfinite(blink_rel_height) or not 0 < blink_rel_height <= 1:
        raise ValueError('blink_rel_height must be between 0 and 1.')

    if not np.isfinite(flat_threshold_v) or flat_threshold_v <= 0:
        raise ValueError('flat_threshold_v must be positive and finite.')

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
    """Return consecutive True intervals as start and exclusive stop indices.

    Parameters
    ----------
    mask : numpy.ndarray
        One-dimensional Boolean mask.

    Returns
    -------
    iterator of tuple of int
        Start and exclusive stop indices of each True interval.
    """

    changes = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return zip(starts.tolist(), stops.tolist())


def get_bad_mask(
    raw: mne.io.BaseRaw,
    signals: np.ndarray,
    flat_threshold_v: float,
) -> np.ndarray:

    """Identify unusable EOG samples from annotations and signal quality.

    Includes existing BAD and EDGE annotations and marks one-second
    windows as bad if either EOG channel is flat.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording containing the original annotations.
    signals : numpy.ndarray
        EOG signals with shape (2, n_samples).
    flat_threshold_v : float
        Minimum peak-to-peak amplitude in volts.

    Returns
    -------
    numpy.ndarray
        Boolean mask indicating unusable samples.
    """

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

            if amplitude <= flat_threshold_v:
                bad_mask[start:stop] = True
                break

    return bad_mask


def find_blinks_eog(
    signal: np.ndarray,
    sfreq: float,
    bad_mask: np.ndarray,
    peak_threshold: float,
    blink_rel_height: float,
) -> dict[str, Any]:

    """Detect blink peaks and determine their time intervals.

    Cleans usable signal segments, detects positive and negative peaks
    using a z-score threshold and selects the dominant polarity.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional EOG signal.
    sfreq : float
        Sampling frequency in Hz.
    bad_mask : numpy.ndarray
        Boolean mask indicating unusable samples.
    peak_threshold : float
        Minimum absolute z-score for peak detection.
    blink_rel_height : float
        Relative height used to determine peak widths.

    Returns
    -------
    dict
        Detected peaks, blink mask, peak widths, cleaned detection signal
        and polarity information.
    """

    # clean a copy of the original (usable) signal for blink detection
    detection = np.full(signal.size, np.nan)

    for start, stop in get_intervals(~bad_mask):
        detection[start:stop] = nk.eog_clean(
            signal[start:stop].copy(),
            sampling_rate=sfreq,
        )

    # z-score the cleaned detection signal
    usable = np.isfinite(detection)

    if usable.sum() < MIN_USABLE_SAMPLES:
        raise ValueError('Not enough usable EOG for blink detection.')

    mean = detection[usable].mean()
    std = detection[usable].std()

    if not np.isfinite(std) or std == 0:
        raise ValueError('Not enough non-flat EOG for blink detection.')

    signal_z = np.full(signal.size, np.nan)
    signal_z[usable] = (detection[usable] - mean) / std

    # detect positive and negative peaks separately
    positive_peaks = []
    negative_peaks = []

    positive_left_ips = []
    positive_right_ips = []
    negative_left_ips = []
    negative_right_ips = []

    for start, stop in get_intervals(usable):

        segment_z = signal_z[start:stop]

        # positive peaks
        peaks, properties = find_peaks(
            segment_z,
            height=peak_threshold,
            width=(None, None),
            rel_height=blink_rel_height,
        )

        positive_peaks.extend((peaks + start).tolist())
        positive_left_ips.extend((properties['left_ips'] + start).tolist())
        positive_right_ips.extend((properties['right_ips'] + start).tolist())

        # negative peaks
        peaks, properties = find_peaks(
            -segment_z,
            height=peak_threshold,
            width=(None, None),
            rel_height=blink_rel_height,
        )

        negative_peaks.extend((peaks + start).tolist())
        negative_left_ips.extend((properties['left_ips'] + start).tolist())
        negative_right_ips.extend((properties['right_ips'] + start).tolist())

    # determine dominant polarity from qualified peaks
    n_positive = len(positive_peaks)
    n_negative = len(negative_peaks)

    if n_positive == n_negative:
        dominant_polarity = None
    elif n_positive > n_negative:
        dominant_polarity = 'positive'
    else:
        dominant_polarity = 'negative'

    # select peaks and width properties from dominant polarity
    blink_mask = np.zeros(signal.size, dtype=bool)

    if dominant_polarity == 'positive':
        accepted_peaks = np.asarray(positive_peaks, dtype=int)
        left_ips = np.asarray(positive_left_ips)
        right_ips = np.asarray(positive_right_ips)

    elif dominant_polarity == 'negative':
        accepted_peaks = np.asarray(negative_peaks, dtype=int)
        left_ips = np.asarray(negative_left_ips)
        right_ips = np.asarray(negative_right_ips)

    else:
        accepted_peaks = np.array([], dtype=int)
        left_ips = np.array([])
        right_ips = np.array([])

    # create blink mask from the accepted peak widths
    for left, right in zip(left_ips, right_ips):

        blink_start = max(0, int(np.floor(left)))
        blink_stop = min(signal.size, int(np.ceil(right)) + 1)

        if usable[blink_start:blink_stop].all():
            blink_mask[blink_start:blink_stop] = True

    return {
        'accepted_peaks': accepted_peaks,
        'signed_z': signal_z[accepted_peaks],
        'blink_mask': blink_mask,
        'left_ips': left_ips,
        'right_ips': right_ips,
        'detection': detection,
        'dominant_polarity': dominant_polarity,
        'n_positive_candidates': n_positive,
        'n_negative_candidates': n_negative,
    }


def check_channel_assignment(
    signals: np.ndarray,
    sfreq: float,
    bad_mask: np.ndarray,
    peak_threshold: float,
    blink_rel_height: float,
) -> tuple[int, dict[str, Any]]:

    """Identify the vertical EOG channel based on blink amplitudes.

    Detects blink candidates in both channels, compares their peak-to-peak
    amplitudes and assigns the channel with the larger median amplitude
    as vertical.

    Parameters
    ----------
    signals : numpy.ndarray
        EOG signals with shape (2, n_samples).
    sfreq : float
        Sampling frequency in Hz.
    bad_mask : numpy.ndarray
        Boolean mask indicating unusable samples.
    peak_threshold : float
        Minimum absolute z-score for peak detection.
    blink_rel_height : float
        Relative height used to determine peak widths.

    Returns
    -------
    tuple
        Vertical channel index and assignment information.

    Raises
    ------
    ValueError
        If too few events are available or the assignment is ambiguous.
    """
    
    detections = []
    candidate_peaks = []
    for signal in signals:
        result = find_blinks_eog(signal, sfreq, bad_mask, peak_threshold, blink_rel_height)
        detection = result['detection']
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

    if median_amplitudes[0] > median_amplitudes[1]:
        vertical_idx = 0
    elif median_amplitudes[0] < median_amplitudes[1]:
        vertical_idx = 1
    else:
        raise ValueError('EOG assignment is ambiguous. Median amplitudes are equal.')

    return vertical_idx, {
        'n_events': len(amplitudes),
        'median_amplitudes': median_amplitudes,
        'amplitudes': amplitudes,
    }

    # TODO: maybe use regression to determine vertical channel based on amplitudes?
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
    blink_rel_height: float,
    check_and_correct_eog_polarity: bool,
    interpolate_blinks: bool,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:

    """Detect, correct and interpolate blinks in both EOG channels.

    Detects blinks in the vertical channel, optionally corrects its polarity
    and interpolates blink intervals in both channels. Adds annotations
    for detected, interpolated and unfilled blinks.

    Parameters
    ----------
    eog : mne.io.BaseRaw
        Recording with channels named vertical and horizontal.
    find_blinks : bool
        Whether to detect blinks.
    sfreq : float
        Sampling frequency in Hz.
    bad_mask : numpy.ndarray
        Boolean mask indicating unusable samples.
    peak_threshold : float
        Minimum absolute z-score for peak detection.
    blink_rel_height : float
        Relative height used to determine peak widths.
    check_and_correct_eog_polarity : bool
        Whether to correct vertical EOG polarity.
    interpolate_blinks : bool
        Whether to interpolate detected blink intervals.

    Returns
    -------
    tuple
        Blink information, polarity information, interpolated mask
        and unfilled mask.
    """
    
    polarity_info: dict[str, Any] = {'checked': False, 'flipped': False}
    blink_info: dict[str, Any] = {'checked': False}
    interpolated_mask = np.zeros(eog.n_times, dtype=bool)
    unfilled_mask = np.zeros(eog.n_times, dtype=bool)
    if find_blinks:
        vertical = eog.get_data(picks=['vertical'])[0]
        blinks = find_blinks_eog(vertical, sfreq, bad_mask, peak_threshold, blink_rel_height)
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
            'dominant_polarity': blinks['dominant_polarity'],
            'n_positive_candidates': blinks['n_positive_candidates'],
            'n_negative_candidates': blinks['n_negative_candidates'],
            'peak_threshold': peak_threshold,
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

    """Correct vertical EOG polarity so dominant blink peaks are positive.

    Parameters
    ----------
    eog : mne.io.BaseRaw
        Recording containing the vertical EOG channel.
    blink_info : dict
        Blink detection results.

    Returns
    -------
    dict
        Polarity information, including whether the signal was flipped.

    Raises
    ------
    ValueError
        If polarity is ambiguous or too few blinks are available.
    """

    signed_z = blink_info['signed_z']
    dominant_polarity = blink_info['dominant_polarity']

    if dominant_polarity is None:
        raise ValueError('EOG polarity is ambiguous.')

    if len(signed_z) < MIN_NR_BLINKS:
        raise ValueError('Too few candidate blinks to infer vertical polarity.')

    n_positive = blink_info['n_positive_candidates']
    n_negative = blink_info['n_negative_candidates']

    if dominant_polarity == 'negative':

        eog.apply_function(
            lambda signal: -signal,
            picks=['vertical'],
        )

    return {
        'checked': True,
        'polarity': dominant_polarity,
        'flipped': dominant_polarity == 'negative',
        'n_positive_blinks': n_positive,
        'n_negative_blinks': n_negative,
    }



def interpolate_eog_blinks(
    eog: mne.io.BaseRaw,
    blink_mask: np.ndarray,
    bad_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    """Interpolate detected blink intervals in both EOG channels using PCHIP.

    Intervals that are too long, overlap unusable samples, touch recording
    boundaries or lack valid interpolation anchors are left unfilled.

    Parameters
    ----------
    eog : mne.io.BaseRaw
        Recording with vertical and horizontal EOG channels.
    blink_mask : numpy.ndarray
        Boolean mask indicating detected blink intervals.
    bad_mask : numpy.ndarray
        Boolean mask indicating unusable samples.

    Returns
    -------
    tuple of numpy.ndarray
        Masks indicating successfully interpolated and unfilled samples.
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

    """Add annotations for consecutive intervals defined by a Boolean mask.

    Existing annotations are preserved.

    Parameters
    ----------
    eog : mne.io.BaseRaw
        Continuous recording to annotate.
    mask : numpy.ndarray
        Boolean mask defining the intervals.
    description : str
        Annotation description.
    """

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
    eog_channels: tuple[str, str] = DEFAULT_EOG_CHANNELS,
    find_blinks: bool = DEFAULT_FIND_BLINKS,
    interpolate_blinks: bool = DEFAULT_INTERPOLATE_BLINKS,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    blink_rel_height: float = DEFAULT_BLINK_REL_HEIGHT,
    check_and_correct_eog_polarity: bool = DEFAULT_CHECK_POLARITY,
    check_and_correct_eog_channel_assignment: bool = DEFAULT_CHECK_CHANNEL_ASSIGNMENT,
    flat_threshold_v: float = DEFAULT_FLAT_THRESHOLD_V,
    max_bad_fraction: float = DEFAULT_MAX_BAD_FRACTION,
    l_freq: float | None = DEFAULT_L_FREQ,
    h_freq: float | None = DEFAULT_H_FREQ,
) -> dict:
    """Clean continuous EOG signals and return the modified Raw object.

    Identifies unusable samples, assigns vertical and horizontal channels,
    detects blinks, optionally corrects polarity and interpolates blink
    intervals. Optional filtering can be applied to both EOG channels.

    The input Raw object is modified in place. Original channel names are
    replaced with EOG001 and EOG002, with the vertical channel first.
    Annotations are added to mark signal-quality issues and blink intervals.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording containing the EOG channels.
    eog_channels : tuple of str
        Names of the two input EOG channels.
    find_blinks : bool
        Whether to detect blinks.
    interpolate_blinks : bool
        Whether to interpolate detected blink intervals.
    peak_threshold : float
        Minimum absolute z-score for peak detection.
    blink_rel_height : float
        Relative height used to determine peak widths.
    check_and_correct_eog_polarity : bool
        Whether to correct vertical EOG polarity.
    check_and_correct_eog_channel_assignment : bool
        Whether to automatically assign vertical and horizontal channels.
    flat_threshold_v : float
        Minimum peak-to-peak amplitude in volts for signal-quality assessment.
    max_bad_fraction : float
        Maximum permitted fraction of unusable or unfilled samples.
    l_freq : float | None
        High-pass filter cutoff frequency in Hz.
    h_freq : float | None
        Low-pass filter cutoff frequency in Hz.

    Returns
    -------
    dict
        Contains the modified Raw object and EOG preprocessing information.

    Raises
    ------
    TypeError
        If the input is not an MNE Raw object.
    ValueError
        If validation fails, channel assignment or polarity is ambiguous,
        signal quality is insufficient or the excluded fraction is too high.
    """
    validate_eog_inputs(
        raw,
        eog_channels,
        find_blinks,
        interpolate_blinks,
        check_and_correct_eog_polarity,
        check_and_correct_eog_channel_assignment,
        peak_threshold,
        blink_rel_height,
        flat_threshold_v,
        max_bad_fraction,
    )
    sfreq = float(raw.info['sfreq'])
    eog = raw.load_data()
    signals = eog.get_data(picks=list(eog_channels))
    if not np.isfinite(signals).all():
        raise ValueError('EOG contains NaN or infinite values.')

    bad_mask = get_bad_mask(eog, signals, flat_threshold_v)
    add_eog_annotations(eog, bad_mask, 'BAD_EOG_quality')

    vertical_idx = 0
    assignment_info: dict[str, Any] = {'method': 'supplied_order'}
    if check_and_correct_eog_channel_assignment:
        vertical_idx, assignment_info = check_channel_assignment(
            signals,
            sfreq,
            bad_mask,
            peak_threshold,
            blink_rel_height,
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
        blink_rel_height,
        check_and_correct_eog_polarity,
        interpolate_blinks,
    )

    excluded_mask = bad_mask | unfilled_mask

    excluded_fraction = float(excluded_mask.mean())

    if excluded_fraction > max_bad_fraction:
        raise ValueError(f'EOG rejected: {excluded_fraction:.1%} of samples are excluded after cleaning.')
    if l_freq is not None or h_freq is not None:
        eog.filter(l_freq=l_freq, h_freq=h_freq, picks=['vertical', 'horizontal'], skip_by_annotation=('bad', 'edge'))

    eog.rename_channels({
    'vertical': 'EOG001',
    'horizontal': 'EOG002',
    })

    eog.reorder_channels([
        'EOG001', 'EOG002'] +
        [ch for ch in eog.ch_names if ch not in('EOG001', 'EOG002')]
    )

    return {
    'data': eog,
    'eog_info': {
        'channel_assignment': {
            'mapping': assignment_info['mapping'],
        },
        'vertical_polarity': {
            'checked': polarity_info['checked'],
            'polarity': polarity_info.get('polarity'),
            'n_positive_blinks': polarity_info.get('n_positive_blinks'),
            'n_negative_blinks': polarity_info.get('n_negative_blinks'),
        },
        'blink_events': {
            'checked': blink_info['checked'],
            'n_blinks': blink_info.get('n_blinks'),
        },
        'quality': {
            'excluded_fraction': excluded_fraction,
            'interpolated_fraction': float(interpolated_mask.mean()),
            'unfilled_blink_fraction': float(unfilled_mask.mean()),
            'max_bad_fraction': max_bad_fraction,
            'flat_threshold_v': flat_threshold_v,
        },
        'filter': {
            'l_freq': l_freq,
            'h_freq': h_freq,
        },
        'sampling_rate': sfreq,
    },
}