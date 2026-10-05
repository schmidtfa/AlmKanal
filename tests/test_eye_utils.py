"""Tests for VPixx eye-tracking preprocessing and MEG alignment."""

from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.io import savemat

from almkanal.eye_utils.gaze_utils import (
    add_eye_tracking_data,
    align_eye_to_meg,
    load_eyetracking_data,
    read_vpixx_mat,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_synthetic_vpixx_mat(
    path: Path,
    *,
    sfreq: float = 2000.0,
    duration: float = 5.0,
    trigger_times: tuple[float, ...] = (1.0, 2.0, 3.0),
    trigger_codes: tuple[int, ...] = (1, 2, 3),
) -> None:
    """Create a synthetic VPixx-like MAT file.

    The generated file follows the 19-channel VPixx layout expected by
    ``read_vpixx_mat()`` and ``make_eye_mne()``.

    Parameters
    ----------
    path
        Output MAT-file path.
    sfreq
        Sampling frequency in Hz.
    duration
        Recording duration in seconds.
    trigger_times
        Times at which digital-output triggers occur.
    trigger_codes
        Trigger values corresponding to ``trigger_times``.
    """
    n_samples = int(round(duration * sfreq))
    times = np.arange(n_samples) / sfreq

    # 20 columns:
    #   column 0 = timestamp
    #   columns 1:20 = the 19 VPixx channels
    data = np.zeros((n_samples, 20), dtype=float)

    data[:, 0] = times

    # ------------------------------------------------------------------
    # Synthetic gaze data
    # ------------------------------------------------------------------

    # Smooth gaze trajectory.
    gaze_x = 960.0 + 100.0 * np.sin(2 * np.pi * 0.5 * times)
    gaze_y = 540.0 + 50.0 * np.cos(2 * np.pi * 0.5 * times)

    data[:, 1] = gaze_x
    data[:, 2] = gaze_y
    data[:, 3] = 3.0

    # Right eye is similar but slightly different.
    data[:, 4] = gaze_x + 5.0
    data[:, 5] = gaze_y + 3.0
    data[:, 6] = 3.1

    # Digital input
    data[:, 7] = 0.0

    # Left/right blink channels
    data[:, 8] = 0.0
    data[:, 9] = 0.0

    # Digital output is column 10 in the original MAT array because
    # column 0 is the timestamp. After read_vpixx_mat(), it becomes
    # eye_data[:, 9].
    digital_output = np.zeros(n_samples)

    for time, code in zip(trigger_times, trigger_codes):
        sample = int(round(time * sfreq))

        if 0 <= sample < n_samples:
            digital_output[sample] = code

    data[:, 10] = digital_output

    # Remaining VPixx channels are left at zero.
    #
    # Columns:
    # 11 Left Eye Fixation
    # 12 Right Eye Fixation
    # 13 Left Eye Saccade
    # 14 Right Eye Saccade
    # 15 Message code
    # 16 Left Eye Raw x
    # 17 Left Eye Raw y
    # 18 Right Eye Raw x
    # 19 Right Eye Raw y

    savemat(path, {"data": data})


def make_synthetic_meg(
    *,
    sfreq: float = 1000.0,
    duration: float = 5.0,
    trigger_times: tuple[float, ...] = (1.0, 2.0, 3.0),
    trigger_codes: tuple[int, ...] = (1, 2, 3),
) -> mne.io.RawArray:
    """Create a synthetic MEG Raw object with synchronization triggers."""
    n_samples = int(round(duration * sfreq))

    info = mne.create_info(
        ch_names=["MEG001", "STI101"],
        sfreq=sfreq,
        ch_types=["mag", "stim"],
    )

    data = np.zeros((2, n_samples), dtype=float)

    times = np.arange(n_samples) / sfreq
    data[0] = np.sin(2 * np.pi * 10.0 * times)

    for time, code in zip(trigger_times, trigger_codes):
        sample = int(round(time * sfreq))

        if 0 <= sample < n_samples - 2:
            data[1, sample : sample + 3] = code

    return mne.io.RawArray(data, info)




def test_read_vpixx_mat(tmp_path):
    """Test reading a synthetic VPixx MAT file."""
    path = tmp_path / "synthetic_eye.mat"

    make_synthetic_vpixx_mat(
        path,
        sfreq=2000.0,
        duration=2.0,
    )

    data, sfreq = read_vpixx_mat(path)

    assert data.shape == (4000, 19)
    assert sfreq == pytest.approx(2000.0)

    # Timestamp was removed.
    assert data.shape[1] == 19


def test_load_eyetracking_data(tmp_path):
    """Test complete VPixx preprocessing."""
    path = tmp_path / "synthetic_eye.mat"

    make_synthetic_vpixx_mat(
        path,
        sfreq=2000.0,
        duration=5.0,
    )

    raw = load_eyetracking_data(path)

    assert isinstance(raw, mne.io.BaseRaw)

    assert raw.info["sfreq"] == pytest.approx(2000.0)
    assert raw.n_times == 10_000

    expected_channels = {
        "eyetracker_x",
        "eyetracker_y",
        "pupil_diameter",
        "blinks",
        "velocity",
        "velocity_x",
        "velocity_y",
        "Digital Output",
    }

    assert expected_channels.issubset(raw.ch_names)

    # Digital output should be a stimulus channel.
    assert raw.get_channel_types(picks=["Digital Output"]) == ["stim"]

    # The derived gaze signals should contain finite data.
    for channel in (
        "eyetracker_x",
        "eyetracker_y",
        "pupil_diameter",
        "velocity",
        "velocity_x",
        "velocity_y",
    ):
        data = raw.get_data(picks=[channel])

        assert data.shape == (1, raw.n_times)
        assert np.isfinite(data).all()



def test_align_eye_to_meg(tmp_path):
    """Test trigger-based eye/MEG alignment.

    The eye tracker intentionally uses a different sampling frequency
    and starts 0.5 seconds later than the MEG recording.
    """
    eye_path = tmp_path / "synthetic_eye.mat"

    meg_sfreq = 1000.0
    eye_sfreq = 2000.0

    # MEG triggers occur at these times.
    meg_trigger_times = (1.0, 2.0, 3.0)

    # Eye recording starts 0.5 seconds later.
    #
    # Therefore the same physical events occur at:
    # 1.5, 2.5, 3.5 seconds in eye-tracker time.
    eye_trigger_times = (0.5, 1.5, 2.5)
    trigger_codes = (1, 2, 3)

    meg = make_synthetic_meg(
        sfreq=meg_sfreq,
        duration=5.0,
        trigger_times=meg_trigger_times,
        trigger_codes=trigger_codes,
    )

    make_synthetic_vpixx_mat(
        eye_path,
        sfreq=eye_sfreq,
        duration=4.0,
        trigger_times=eye_trigger_times,
        trigger_codes=trigger_codes,
    )

    eye = load_eyetracking_data(
        eye_path,
        interpolate_blinks=False,
    )

    assert meg.info["sfreq"] == pytest.approx(1000.0)
    assert eye.info["sfreq"] == pytest.approx(2000.0)

    aligned_eye = align_eye_to_meg(
        meg,
        eye,
    )

    # realign_raw() should have made the sampling frequencies compatible.
    assert aligned_eye.info["sfreq"] == pytest.approx(
        meg.info["sfreq"]
    )

    # The recordings should now have compatible sample counts.
    assert aligned_eye.n_times == meg.n_times


# ---------------------------------------------------------------------------
# End-to-end public API test
# ---------------------------------------------------------------------------


def test_add_eye_tracking_data(tmp_path):
    """Test complete MEG + eye-tracking integration."""
    eye_path = tmp_path / "synthetic_eye.mat"

    meg = make_synthetic_meg(
        sfreq=1000.0,
        duration=5.0,
        trigger_times=(1.0, 2.0, 3.0),
        trigger_codes=(1, 2, 3),
    )

    # Eye tracker has:
    # - a different sampling frequency
    # - a 0.5 second temporal offset
    make_synthetic_vpixx_mat(
        eye_path,
        sfreq=2000.0,
        duration=5.0,
        trigger_times=(1.5, 2.5, 3.5),
        trigger_codes=(1, 2, 3),
    )

    original_meg_channels = list(meg.ch_names)

    result = add_eye_tracking_data(
        meg,
        eye_path,
        align=True,
        interpolate_blinks=False,
    )

    # Original MEG channels are still present.
    for channel in original_meg_channels:
        assert channel in result.ch_names

    # Eye channels were added.
    expected_eye_channels = {
        "eyetracker_x",
        "eyetracker_y",
        "pupil_diameter",
        "blinks",
        "velocity",
        "velocity_x",
        "velocity_y",
        "Digital Output",
    }

    assert expected_eye_channels.issubset(result.ch_names)

    # Sampling frequencies should now agree.
    assert result.info["sfreq"] == pytest.approx(1000.0)

    # add_channels() requires compatible sample counts.
    assert result.n_times == meg.n_times

    # The resulting object contains more channels than the original MEG.
    assert result.info["nchan"] > len(original_meg_channels)