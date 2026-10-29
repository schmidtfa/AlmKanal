from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

from almkanal.stim_utils import alignment_utils


def test_long_recording_warns_with_narrow_search_and_recovers_with_wider_search(tmp_path: Path) -> None:
    """Exercise real audio correlation at 1 kHz with 630 ms accumulated drift."""
    sfreq = 1000
    duration = 1260.0
    offset = 0.02
    drift = 500e-6
    times = np.arange(round(duration * sfreq)) / sfreq
    modulation = sosfiltfilt(butter(3, 8.0, fs=sfreq, output='sos'), np.random.default_rng(42).normal(size=len(times)))
    modulation /= np.max(np.abs(modulation))
    audio = ((0.55 + 0.4 * modulation) * np.sin(2 * np.pi * 173 * times)).astype(np.float32)
    path = tmp_path / 'long_stimulus.wav'
    wavfile.write(path, sfreq, audio)
    raw_times = np.arange(round((offset + (1 + drift) * duration + 0.1) * sfreq)) / sfreq
    recorded = np.interp((raw_times - offset) / (1 + drift), times, audio, left=0.0, right=0.0)
    raw = mne.io.RawArray(recorded[None], mne.create_info(['audio'], sfreq, ['misc']), verbose=False)

    with pytest.warns(RuntimeWarning, match='Potentially unreliable audio alignment.*long_stimulus.wav'):
        narrow = alignment_utils.estimate_raw_wav_alignment(raw, path, audio_channels=['audio'], verbose=False)
    assert narrow['quality_warnings']

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        wide = alignment_utils.estimate_raw_wav_alignment(
            raw, path, audio_channels=['audio'], max_lag_s=1.0, verbose=False
        )
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert wide['quality_warnings'] == []
    assert wide['offset_s'] == pytest.approx(offset, abs=0.003)
    assert wide['drift_us_per_s'] == pytest.approx(500.0, abs=2.0)
    assert wide['residual_rms_ms'] < 2.0


@pytest.fixture
def anchor_trial(tmp_path: Path):
    sfreq = 1000
    times = np.arange(120 * sfreq) / sfreq
    audio = ((1 + 0.3 * np.sin(2 * np.pi * 3 * times)) * np.sin(2 * np.pi * 173 * times)).astype(np.float32)
    path = tmp_path / 'anchors.wav'
    wavfile.write(path, sfreq, audio)
    raw = mne.io.RawArray(audio[None], mne.create_info(['audio'], sfreq, ['misc']), verbose=False)
    return raw, path


@pytest.mark.parametrize('sign', [-1.0, 1.0])
@pytest.mark.parametrize('case', ['boundary', 'endpoint', 'coverage'])
def test_precise_fit_can_still_warn_about_search_range_or_coverage(anchor_trial, monkeypatch, sign, case):
    raw, path = anchor_trial

    def window_lag(recorded, reference, *, center_s, **kwargs):
        if case == 'boundary':
            return sign * 0.249, 0.99
        if case == 'endpoint':
            # Good early anchors imply an end lag beyond the search range.
            return sign * 0.0025 * center_s, 0.99 if center_s < 80 else 0.0
        # A small, correct-looking lag is supported only near the start.
        return sign * (0.02 + 0.0001 * center_s), 0.99 if center_s < 40 else 0.0

    monkeypatch.setattr(alignment_utils, '_estimate_window_lag', window_lag)
    expected = 'inlier anchors span only' if case == 'coverage' else 'search limit'
    with pytest.warns(RuntimeWarning, match=expected):
        result = alignment_utils.estimate_raw_wav_alignment(raw, path, audio_channels=['audio'], verbose=False)
    assert result['residual_rms_ms'] < 1e-6
    assert any(expected in reason for reason in result['quality_warnings'])


def test_scattered_anchors_warn_and_residual_threshold_is_configurable(anchor_trial, monkeypatch):
    raw, path = anchor_trial

    def window_lag(recorded, reference, *, center_s, **kwargs):
        # High correlations alone do not imply a precise, linear timing fit.
        return 0.03 * np.sin(center_s), 0.99

    monkeypatch.setattr(alignment_utils, '_estimate_window_lag', window_lag)
    with pytest.warns(RuntimeWarning, match='inlier residual RMS.*10 ms'):
        result = alignment_utils.estimate_raw_wav_alignment(raw, path, audio_channels=['audio'], verbose=False)
    assert result['median_correlation'] == pytest.approx(0.99)
    assert result['residual_rms_ms'] > 10
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        relaxed = alignment_utils.estimate_raw_wav_alignment(
            raw, path, audio_channels=['audio'], warn_residual_rms_ms=50.0, verbose=False
        )
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert relaxed['quality_warnings'] == []
    assert relaxed['clock_slope'] == result['clock_slope']
    assert relaxed['offset_s'] == result['offset_s']


@pytest.mark.parametrize('threshold', [0.0, -1.0, np.nan, np.inf])
def test_invalid_residual_warning_threshold(anchor_trial, threshold):
    raw, path = anchor_trial
    with pytest.raises(ValueError, match='warn_residual_rms_ms must be finite and positive'):
        alignment_utils.estimate_raw_wav_alignment(raw, path, audio_channels=['audio'], warn_residual_rms_ms=threshold)


def test_zero_lag_search_warns_even_for_perfect_anchors(anchor_trial, monkeypatch):
    raw, path = anchor_trial
    monkeypatch.setattr(alignment_utils, '_estimate_window_lag', lambda *args, **kwargs: (0.0, 0.99))
    with pytest.warns(RuntimeWarning, match='no nonzero lag search'):
        result = alignment_utils.estimate_raw_wav_alignment(
            raw, path, audio_channels=['audio'], max_lag_s=0.0, verbose=False
        )
    assert result['residual_rms_ms'] == 0.0
