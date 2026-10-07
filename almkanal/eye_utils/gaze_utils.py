"""Eye-tracking preprocessing utilities.

This module provides functionality for reading and preprocessing VPixx
TRACKPixx eye-tracking recordings and converting them to MNE Raw objects.

The implementation is primarily intended for VPixx recordings exported as
MAT files, but the resulting objects are standard MNE Raw objects and can
therefore be used independently of MEG recordings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import numpy as np
from pymatreader import read_mat

from .blink_utils import call_blink_annotations, vpixx_default_blinkmap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VPixxConfig:
    # this corresponds to the standard vpixx configurations in the Lab
    # at the Christian-Doppler Klinik in Salzburg. Adjust if needed.
    """Configuration for VPixx eye-tracking preprocessing.

    Parameters
    ----------
    screen_resolution
        Screen resolution as ``(width, height)`` in pixels.
    screen_size
        Physical screen dimensions as ``(width, height)`` in metres.
    screen_distance
        Eye-to-screen distance in metres.
    calibration_model
        MNE eyetracking calibration model.
    calibration_eye
        Eye used by the calibration model.
    blink_buffer
        Time before and after a blink used during blink interpolation.
        The values are given in seconds.
    digital_output_threshold
        Threshold used when correcting VPixx digital output values.
    missing_value
        Value used by the VPixx recording to indicate missing data.
    """

    screen_resolution: tuple[int, int] = (1920, 1080)
    screen_size: tuple[float, float] = (0.61, 0.34)
    screen_distance: float = 0.82
    calibration_model: str = 'HV5'
    calibration_eye: Literal['left', 'right'] = 'right'
    blink_buffer: tuple[float, float] = (0.05, 0.2)
    digital_output_threshold: float = 256.0
    missing_value: float = 9999.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.screen_resolution) != TWO_INPUT_VALUES:
            raise ValueError('screen_resolution must contain two values.')

        if any(x <= 0 for x in self.screen_resolution):
            raise ValueError('screen_resolution values must be positive.')

        if len(self.screen_size) != TWO_INPUT_VALUES:
            raise ValueError('screen_size must contain two values.')

        if any(x <= 0 for x in self.screen_size):
            raise ValueError('screen_size values must be positive.')

        if self.screen_distance <= 0:
            raise ValueError('screen_distance must be positive.')

        if len(self.blink_buffer) != TWO_INPUT_VALUES:
            raise ValueError('blink_buffer must contain two values.')

        if any(x < 0 for x in self.blink_buffer):
            raise ValueError('blink_buffer values must be non-negative.')


TWO_INPUT_VALUES = 2
BLINK_PROBABILITY_THRESHOLD = 0.5

DEFAULT_VPIXX_CONFIG = VPixxConfig()

VPIXX_CHANNELS = (
    'Left Eye x',
    'Left Eye y',
    'Left Eye Pupil Diameter',
    'Right Eye x',
    'Right Eye y',
    'Right Eye Pupil Diameter',
    'Digital Input',
    'Left Eye Blink',
    'Right Eye Blink',
    'Digital Output',
    'Left Eye Fixation',
    'Right Eye Fixation',
    'Left Eye Saccade',
    'Right Eye Saccade',
    'Message code',
    'Left Eye Raw x',
    'Left Eye Raw y',
    'Right Eye Raw x',
    'Right Eye Raw y',
)


def read_vpixx_mat(
    filepath: str | Path,
) -> tuple[np.ndarray, float]:
    """Read a VPixx MAT file.

    Parameters
    ----------
    filepath
        Path to the VPixx ``.mat`` file.

    Returns
    -------
    data
        Eye-tracking data with shape ``(n_samples, n_channels)``.
        The original time column is removed.
    sfreq
        Sampling frequency in Hz.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If the MAT file does not contain a valid ``data`` array or if
        the sampling interval cannot be determined.
    """
    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f'Eye-tracking file does not exist: {filepath}')

    logger.info('Reading VPixx eye-tracking file: %s', filepath)

    mat = read_mat(filepath)

    if 'data' not in mat:
        raise ValueError(f"Could not find 'data' in VPixx MAT file: {filepath}")

    data = np.asarray(mat['data'], dtype=float)

    if data.ndim != TWO_INPUT_VALUES or data.shape[0] < TWO_INPUT_VALUES or data.shape[1] < TWO_INPUT_VALUES:
        raise ValueError('VPixx data must be a 2D array containing at least ' 'two samples and two columns.')

    time = data[:, 0]
    dt = np.diff(time)

    if np.any(dt <= 0):
        raise ValueError('VPixx timestamps must be strictly increasing.')

    sampling_interval = float(np.median(dt))
    sfreq = 1.0 / sampling_interval

    data = data.copy()
    data[:, 0] -= data[0, 0]

    return data[:, 1:], sfreq


def make_eye_mne(
    eye_data: np.ndarray,
    sfreq: float,
) -> mne.io.Raw:
    """Convert VPixx data to an MNE Raw object.

    Parameters
    ----------
    eye_data
        Array with shape ``(n_samples, 19)`` containing VPixx data.
    sfreq
        Sampling frequency in Hz.

    Returns
    -------
    raw
        MNE Raw object containing the raw VPixx channels.

    Raises
    ------
    ValueError
        If the input dimensions do not match the expected VPixx format.
    """
    eye_data = np.asarray(eye_data, dtype=float)

    if eye_data.ndim != TWO_INPUT_VALUES:
        raise ValueError('eye_data must be a 2D array.')

    if eye_data.shape[1] != len(VPIXX_CHANNELS):
        raise ValueError(f'Expected {len(VPIXX_CHANNELS)} VPixx channels, ' f'got {eye_data.shape[1]}.')

    if sfreq <= 0:
        raise ValueError('sfreq must be positive.')

    info = mne.create_info(
        ch_names=list(VPIXX_CHANNELS),
        sfreq=sfreq,
        ch_types=['misc'] * len(VPIXX_CHANNELS),
    )

    raw = mne.io.RawArray(eye_data.T, info)

    mne.preprocessing.eyetracking.set_channel_types_eyetrack(
        raw,
        mapping={
            'Left Eye x': ('eyegaze', 'px', 'left', 'x'),
            'Left Eye Raw x': ('eyegaze', 'px', 'left', 'x'),
            'Left Eye y': ('eyegaze', 'px', 'left', 'y'),
            'Left Eye Raw y': ('eyegaze', 'px', 'left', 'y'),
            'Right Eye x': ('eyegaze', 'px', 'right', 'x'),
            'Right Eye Raw x': ('eyegaze', 'px', 'right', 'x'),
            'Right Eye y': ('eyegaze', 'px', 'right', 'y'),
            'Right Eye Raw y': ('eyegaze', 'px', 'right', 'y'),
            'Left Eye Pupil Diameter': ('pupil', 'au', 'left'),
            'Right Eye Pupil Diameter': ('pupil', 'au', 'right'),
        },
    )

    return raw


def create_vpixx_calibration(
    config: VPixxConfig = DEFAULT_VPIXX_CONFIG,
) -> mne.preprocessing.eyetracking.Calibration:
    """Create an MNE calibration object for VPixx data.

    Parameters
    ----------
    config
        VPixx preprocessing configuration.

    Returns
    -------
    calibration
        MNE eye-tracking calibration object.
    """
    return mne.preprocessing.eyetracking.Calibration(
        onset=-10,
        model=config.calibration_model,
        eye=config.calibration_eye,
        avg_error=0,
        max_error=0,
        positions=np.array([0.0, 0.0]),
        offsets=np.array([0.0, 0.0]),
        gaze=np.array([[0.0, 0.0]]),
        screen_resolution=config.screen_resolution,
        screen_size=config.screen_size,
        screen_distance=config.screen_distance,
    )


vpixx_templatecalibration = create_vpixx_calibration


def _calculate_eye_quality(raw: mne.io.BaseRaw) -> dict[str, float]:
    """Calculate the fraction of missing gaze samples for each eye."""
    quality = {}

    for eye in ('left', 'right'):
        x = raw.get_data(picks=f'{eye.title()} Eye x')[0]
        y = raw.get_data(picks=f'{eye.title()} Eye y')[0]

        quality[eye] = float(np.mean(np.isnan(x) | np.isnan(y)))

    return quality


def _select_best_eye(
    raw: mne.io.BaseRaw,
) -> str | None:
    """Return the eye with the lowest proportion of missing samples."""
    quality = _calculate_eye_quality(raw)

    if np.isclose(quality['left'], quality['right']):
        return None

    return min(quality, key=lambda eye: quality[eye])


def _combine_eyes(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Combine two eye signals while tolerating missing values."""
    stacked = np.vstack([left, right])

    valid = np.isfinite(stacked)

    with np.errstate(invalid='ignore'):
        result = np.nanmean(stacked, axis=0)

    # Avoid warnings and preserve missing values when both eyes are absent.
    result[~valid.any(axis=0)] = np.nan

    return result


def _five_point_velocity(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate gaze velocity using a five-point derivative."""
    dt = 1.0 / sfreq

    vx = np.zeros_like(x, dtype=float)
    vy = np.zeros_like(y, dtype=float)
    min_len = 5
    if len(x) >= min_len:
        vx[2:-2] = (x[4:] + x[3:-1] - x[1:-3] - x[:-4]) / (6.0 * dt)

        vy[2:-2] = (y[4:] + y[3:-1] - y[1:-3] - y[:-4]) / (6.0 * dt)

    velocity = np.sqrt(vx**2 + vy**2)

    return vx, vy, velocity


def _make_derived_channel(
    name: str,
    data: np.ndarray,
    sfreq: float,
) -> mne.io.RawArray:
    """Create a one-channel MNE Raw object."""
    info = mne.create_info(
        ch_names=[name],
        sfreq=sfreq,
        ch_types=['misc'],
    )

    return mne.io.RawArray(
        np.asarray(data, dtype=float)[np.newaxis, :],
        info,
    )


# main preprocessing functions
def load_eyetracking_data(  # noqa PLR0915
    eye_path: str | Path,
    *,
    config: VPixxConfig = DEFAULT_VPIXX_CONFIG,
    interpolate_blinks: bool = True,
    blink_buffer: tuple[float, float] | None = None,
    convert_to_radians: bool = True,
    include_raw_vpixx_channels: bool = False,
) -> mne.io.Raw:
    """Load and preprocess VPixx eye-tracking data.

    The function performs the following operations:

    1. Read the VPixx MAT file.
    2. Correct the digital output representation when necessary.
    3. Replace VPixx missing-value markers with NaN.
    4. Convert the data to an MNE Raw object.
    5. Convert gaze coordinates to radians using the configured screen
       geometry.
    6. Detect blink annotations.
    7. Interpolate gaze data around blinks.
    8. Select the better eye based on missing-data proportion.
    9. Create combined/selected gaze, pupil and blink signals.
    10. Calculate gaze velocity.
    11. Add convenient derived channels.

    Parameters
    ----------
    eye_path
        Path to the VPixx MAT file.
    config
        VPixx configuration.
    interpolate_blinks
        Whether to interpolate gaze data around detected blinks.
    blink_buffer
        Blink interpolation buffer ``(before, after)`` in seconds.
        If None, ``config.blink_buffer`` is used.
    convert_to_radians
        If True, convert gaze coordinates from pixels to radians.
    include_raw_vpixx_channels
        If True, retain the original VPixx channels in the returned Raw.
        If False, only the processed channels plus ``Digital Output`` are
        returned.

    Returns
    -------
    raw
        Preprocessed eye-tracking recording.

        The returned object contains, at minimum:

        - ``eyetracker_x``
        - ``eyetracker_y``
        - ``pupil_diameter``
        - ``blinks``
        - ``velocity``
        - ``velocity_x``
        - ``velocity_y``
        - ``Digital Output``

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the VPixx file or configuration is invalid.

    Notes
    -----
    This function does not perform any MEG alignment. Eye tracking can
    therefore be used completely independently of MEG.
    """
    if blink_buffer is None:
        blink_buffer = config.blink_buffer

    logger.info('Loading eye-tracking data from %s', eye_path)

    eye_data, sfreq = read_vpixx_mat(eye_path)

    # VPixx digital output values are sometimes stored with an extra
    # factor of 256.
    digital_output = eye_data[:, 9]

    if np.count_nonzero(digital_output > config.digital_output_threshold) > 1:
        logger.debug('Scaling VPixx Digital Output by 1/256.')
        eye_data[:, 9] /= 256.0

    # Replace VPixx missing-value marker.
    eye_data[eye_data == config.missing_value] = np.nan

    raw = make_eye_mne(eye_data, sfreq)

    if convert_to_radians:
        calibration = create_vpixx_calibration(config)

        raw = mne.preprocessing.eyetracking.convert_units(
            raw,
            calibration=calibration,
            to='radians',
        )

    blink_map = vpixx_default_blinkmap()
    annotations = call_blink_annotations(raw, blink_map)
    raw.set_annotations(annotations)

    quality = _calculate_eye_quality(raw)

    logger.info(
        'Missing gaze data: left=%.2f%%, right=%.2f%%',
        quality['left'] * 100,
        quality['right'] * 100,
    )

    good_eye = _select_best_eye(raw)

    if good_eye is None:
        logger.info('Both eyes have equivalent data quality; combining both eyes.')
    else:
        logger.info('Selected %s eye based on data quality.', good_eye)

    if interpolate_blinks:
        raw_clean = mne.preprocessing.eyetracking.interpolate_blinks(
            raw,
            buffer=blink_buffer,
            interpolate_gaze=True,
        )
    else:
        raw_clean = raw.copy()

    sfreq = raw_clean.info['sfreq']

    left_x = raw_clean.get_data(picks='Left Eye x')[0]
    right_x = raw_clean.get_data(picks='Right Eye x')[0]
    left_y = raw_clean.get_data(picks='Left Eye y')[0]
    right_y = raw_clean.get_data(picks='Right Eye y')[0]

    left_pupil = raw_clean.get_data(picks='Left Eye Pupil Diameter')[0]
    right_pupil = raw_clean.get_data(picks='Right Eye Pupil Diameter')[0]

    left_blink = raw_clean.get_data(picks='Left Eye Blink')[0]
    right_blink = raw_clean.get_data(picks='Right Eye Blink')[0]

    if good_eye == 'left':
        x = left_x
        y = left_y
        pupil = left_pupil
        blinks = left_blink

    elif good_eye == 'right':
        x = right_x
        y = right_y
        pupil = right_pupil
        blinks = right_blink

    else:
        x = _combine_eyes(left_x, right_x)
        y = _combine_eyes(left_y, right_y)
        pupil = _combine_eyes(left_pupil, right_pupil)

        blinks = (
            np.nanmean(
                np.vstack([left_blink, right_blink]),
                axis=0,
            )
            > BLINK_PROBABILITY_THRESHOLD
        ).astype(int)

    velocity_x, velocity_y, velocity = _five_point_velocity(
        x,
        y,
        sfreq,
    )

    derived = [
        _make_derived_channel('eyetracker_x', x, sfreq),
        _make_derived_channel('eyetracker_y', y, sfreq),
        _make_derived_channel('pupil_diameter', pupil, sfreq),
        _make_derived_channel('blinks', blinks, sfreq),
        _make_derived_channel('velocity', velocity, sfreq),
        _make_derived_channel('velocity_x', velocity_x, sfreq),
        _make_derived_channel('velocity_y', velocity_y, sfreq),
    ]

    raw_clean.set_channel_types(
        {'Digital Output': 'stim'},
        on_unit_change='ignore',
    )

    raw_clean.add_channels(
        derived,
        force_update_info=True,
    )

    if include_raw_vpixx_channels:
        keep_channels = [
            *VPIXX_CHANNELS,
            'eyetracker_x',
            'eyetracker_y',
            'pupil_diameter',
            'blinks',
            'velocity',
            'velocity_x',
            'velocity_y',
        ]
    else:
        keep_channels = [
            'eyetracker_x',
            'eyetracker_y',
            'pupil_diameter',
            'blinks',
            'velocity',
            'velocity_x',
            'velocity_y',
            'Digital Output',
        ]

    raw_clean.pick(keep_channels)

    raw_clean.info['description'] = f"VPixx eye tracking; selected_eye={good_eye or 'both'}"

    logger.info(
        'Finished eye-tracking preprocessing: %d samples, %.2f Hz.',
        raw_clean.n_times,
        raw_clean.info['sfreq'],
    )

    return raw_clean


def align_eye_to_meg(
    meg_data: mne.io.BaseRaw,
    eye_data: mne.io.BaseRaw,
    *,
    meg_stim_channel: str = 'STI101',
    eye_stim_channel: str = 'Digital Output',
    meg_min_duration: float = 0.002,
    meg_max_trigger: int = 4096,
) -> mne.io.BaseRaw:
    """Align an eye-tracking Raw object to an MEG Raw object.

    Alignment is based on matching digital trigger events by trigger code
    and occurrence order.

    Parameters
    ----------
    meg_data
        MEG recording to use as the temporal reference.
    eye_data
        Preprocessed eye-tracking recording.
    meg_stim_channel
        MEG stimulus channel containing synchronization triggers.
    eye_stim_channel
        Eye-tracker stimulus channel containing synchronization triggers.
    meg_min_duration
        Minimum event duration passed to :func:`mne.find_events`.
    meg_max_trigger
        Triggers at or above this value are ignored in the MEG recording.

    Returns
    -------
    eye_data
        The same eye-tracking Raw object after temporal realignment.

    Raises
    ------
    ValueError
        If required stimulus channels are missing or if no matching events
        can be found.

    Notes
    -----
    The eye-tracking object is modified by MNE's realignment procedure.
    If you need to preserve the original, pass ``eye_data.copy()``.
    """
    if not isinstance(meg_data, mne.io.BaseRaw):
        raise TypeError('meg_data must be an MNE Raw object.')

    if not isinstance(eye_data, mne.io.BaseRaw):
        raise TypeError('eye_data must be an MNE Raw object.')

    if meg_stim_channel not in meg_data.ch_names:
        raise ValueError(f'MEG stimulus channel {meg_stim_channel!r} ' 'was not found in the recording.')

    if eye_stim_channel not in eye_data.ch_names:
        raise ValueError(f'Eye stimulus channel {eye_stim_channel!r} ' 'was not found in the recording.')

    logger.info('Finding MEG synchronization events.')

    eye_events = mne.find_events(
        eye_data,
        stim_channel=eye_stim_channel,
        initial_event=True,
    )

    meg_events = mne.find_events(
        meg_data,
        stim_channel=meg_stim_channel,
        min_duration=meg_min_duration,
    )

    eye_events = eye_events[eye_events[:, 1] == 0]

    meg_events = meg_events[meg_events[:, 1] == 0]
    meg_events = meg_events[meg_events[:, 2] < meg_max_trigger]

    if len(eye_events) == 0:
        raise ValueError('No synchronization events found in eye tracking.')

    if len(meg_events) == 0:
        raise ValueError('No synchronization events found in MEG.')

    meg_matched, eye_matched = _match_events_by_code_and_order(
        meg_events,
        eye_events,
    )

    if len(meg_matched) == 0:
        raise ValueError('No matching synchronization events were found between ' 'MEG and eye tracking.')

    logger.info(
        'Found %d matching synchronization events.',
        len(meg_matched),
    )

    eye_samples = eye_matched[:, 0]
    t_eye = eye_samples / eye_data.info['sfreq']

    meg_samples = meg_matched[:, 0] - meg_data.first_samp
    t_meg = meg_samples / meg_data.info['sfreq']

    logger.debug('MEG synchronization times: %s', t_meg)
    logger.debug('Eye synchronization times: %s', t_eye)

    mne.preprocessing.realign_raw(
        meg_data,
        eye_data,
        t_raw=t_meg,
        t_other=t_eye,
        verbose='error',
    )

    return eye_data


def _match_events_by_code_and_order(
    meg_events: np.ndarray,
    eye_events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match events by trigger code and occurrence order.

    Matching is returned in chronological order.
    """
    matched_meg: list[np.ndarray] = []
    matched_eye: list[np.ndarray] = []

    codes = np.intersect1d(
        np.unique(meg_events[:, 2]),
        np.unique(eye_events[:, 2]),
    )

    for code in codes:
        meg_code_events = meg_events[meg_events[:, 2] == code]
        eye_code_events = eye_events[eye_events[:, 2] == code]

        n = min(len(meg_code_events), len(eye_code_events))

        matched_meg.extend(meg_code_events[:n])
        matched_eye.extend(eye_code_events[:n])

    if not matched_meg:
        return (
            np.empty((0, 3), dtype=int),
            np.empty((0, 3), dtype=int),
        )

    matched_meg_array = np.asarray(matched_meg)
    matched_eye_array = np.asarray(matched_eye)

    order = np.argsort(matched_meg_array[:, 0])

    return (
        matched_meg_array[order],
        matched_eye_array[order],
    )


def add_eye_tracking_data(
    meg_data: mne.io.BaseRaw,
    eye_path: str | Path,
    *,
    align: bool = True,
    config: VPixxConfig = DEFAULT_VPIXX_CONFIG,
    interpolate_blinks: bool = True,
    blink_buffer: tuple[float, float] | None = None,
    convert_to_radians: bool = True,
    include_raw_vpixx_channels: bool = False,
    meg_stim_channel: str = 'STI101',
    eye_stim_channel: str = 'Digital Output',
) -> mne.io.BaseRaw:
    """Add preprocessed eye tracking to an MEG recording.

    Parameters
    ----------
    meg_data
        MEG Raw object to which eye-tracking channels are added.
    eye_path
        Path to the VPixx MAT file.
    align
        If True, align eye tracking to MEG using synchronization triggers.
        If False, no temporal alignment is performed.
    config
        VPixx preprocessing configuration.
    interpolate_blinks
        Whether to interpolate gaze data around blinks.
    blink_buffer
        Blink interpolation buffer in seconds.
    convert_to_radians
        Convert gaze coordinates to radians.
    include_raw_vpixx_channels
        Retain all original VPixx channels in addition to derived channels.
    meg_stim_channel
        MEG synchronization trigger channel.
    eye_stim_channel
        Eye-tracker synchronization trigger channel.

    Returns
    -------
    meg_data
        The modified MEG Raw object with eye-tracking channels appended.

    Raises
    ------
    TypeError
        If ``meg_data`` is not an MNE Raw object.

    Notes
    -----
    When ``align=False``, the eye-tracking data are appended without any
    temporal synchronization. This is useful when the eye tracker is being
    analyzed independently or when synchronization has already been
    performed elsewhere.
    """
    if not isinstance(meg_data, mne.io.BaseRaw):
        raise TypeError('meg_data must be an MNE Raw object.')

    eye_data = load_eyetracking_data(
        eye_path,
        config=config,
        interpolate_blinks=interpolate_blinks,
        blink_buffer=blink_buffer,
        convert_to_radians=convert_to_radians,
        include_raw_vpixx_channels=include_raw_vpixx_channels,
    )

    if align:
        logger.info('Aligning eye tracking to MEG.')
        align_eye_to_meg(
            meg_data,
            eye_data,
            meg_stim_channel=meg_stim_channel,
            eye_stim_channel=eye_stim_channel,
        )
    else:
        logger.info('Eye-tracking/MEG alignment disabled; ' 'adding eye tracking without synchronization.')

    meg_data.load_data()
    eye_data.load_data()

    if meg_data.info['sfreq'] != eye_data.info['sfreq']:
        raise ValueError(
            'MEG and eye-tracking sampling frequencies differ after '
            'alignment. Resample the eye-tracking data before adding it '
            'to the MEG recording.'
        )

    if meg_data.n_times != eye_data.n_times:
        raise ValueError(
            'MEG and eye-tracking recordings have different numbers of '
            'samples. They cannot be combined with Raw.add_channels(). '
            'Check temporal alignment and recording durations.'
        )

    meg_data.add_channels(
        [eye_data],
        force_update_info=True,
    )

    logger.info('Eye-tracking channels added to MEG.')

    return meg_data
