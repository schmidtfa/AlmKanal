from __future__ import annotations

import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import attrs
import mne
import numpy as np
import pytest
from scipy.io import wavfile

from almkanal import EpochTRF, TRFSpanSpec
from almkanal.almkanal_steps import trf_utils
from almkanal.stim_utils.audio_utils import prepare_audio


@pytest.fixture
def timing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sfreq = 500.0
    duration = 4.0
    slope = 1.2  # Large enough to distinguish delay-before from delay-after.
    offset = 0.04
    onset = 0.8
    times = np.arange(3500) / sfreq
    neural = (times - onset - offset) / slope
    raw = mne.io.RawArray(
        np.vstack([neural, np.zeros(len(times))]),
        mne.create_info(['EEG 001', 'recorded'], sfreq, ['eeg', 'misc']),
        first_samp=1234,
        verbose=False,
    )
    wav = tmp_path / 'stimulus.wav'
    wavfile.write(wav, 1000, np.sin(2 * np.pi * 100 * np.arange(4000) / 1000).astype(np.float32))
    alignment = {
        'offset_s': offset, 'clock_slope': slope, 'wav_duration_s': duration,
        'raw_sfreq': sfreq, 'drift_us_per_s': (slope - 1) * 1e6,
        'total_drift_ms': (slope - 1) * duration * 1000,
        'residual_rms_ms': 0.1, 'residual_max_ms': 0.2,
        'median_correlation': 0.99, 'n_anchor_inliers': 8,
    }

    def estimate(raw_trial, wav_file, **kwargs):
        assert raw_trial.first_samp == raw.first_samp + round(onset * sfreq)
        assert kwargs['audio_channels'] == ['recorded']
        if not Path(wav_file).exists():
            raise FileNotFoundError(wav_file)
        return alignment.copy()

    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', estimate)
    return {
        'raw': raw, 'spec': TRFSpanSpec({'stimulus': (1634, 4134)}),
        'path': tmp_path, 'wav': wav, 'alignment': alignment,
    }


@pytest.mark.parametrize('delay', [-0.2, 0.0, 0.2])
def test_physical_delay_is_applied_after_clock_correction(timing_data, delay) -> None:
    raw = timing_data['raw']
    before = raw.get_data().copy()
    warning = pytest.warns(UserWarning, match='Negative hw_delay_s delays') if delay < 0 else nullcontext()
    with warning:
        epochs = trf_utils.build_trf_epochs(
            raw, timing_data['spec'], timing_data['path'], audio_channels=['recorded'],
            hw_delay_s=delay, epoch_len_s=1.0, verbose=False,
        )
    assert len(epochs) == 4  # Both endpoints survive either sign of delay.
    neural = epochs.get_data(picks=['EEG 001'])[:, 0]
    expected = np.arange(2000).reshape(4, 500) / 500 + delay
    np.testing.assert_allclose(neural[:, 100:-100], expected[:, 100:-100], atol=0.004)
    audio, _, _, _ = prepare_audio(str(timing_data['wav']), target_fs=500)
    np.testing.assert_allclose(epochs.get_data(picks=['env_rms'])[:, 0], audio[0, :2000].reshape(4, 500))
    np.testing.assert_array_equal(raw.get_data(), before)
    np.testing.assert_allclose(epochs.metadata['wav_t_on'], [0, 1, 2, 3])
    np.testing.assert_allclose(epochs.metadata['t_on'], 0.8 + 0.04 + 1.2 * (np.arange(4) + delay))


def test_annotations_and_epoch_metadata_survive_rejection(timing_data) -> None:
    raw = timing_data['raw']
    raw.set_annotations(mne.Annotations([0.8 + 0.04 + 1.2 * 1.5], [0.12], ['BAD movement']))
    epochs = trf_utils.build_trf_epochs(
        raw, timing_data['spec'], timing_data['path'], audio_channels=['recorded'],
        hw_delay_s=0.2, epoch_len_s=1.0, verbose=False,
    )
    assert len(epochs) == 3
    assert epochs.metadata['epoch_index_in_segment'].tolist() == [0, 2, 3]
    np.testing.assert_allclose(epochs.metadata['wav_t_on'], [0, 2, 3])
    np.testing.assert_allclose(epochs.metadata['t_on'], 0.84 + 1.2 * (np.array([0, 2, 3]) + 0.2))


def test_repeated_wav_trials_keep_their_own_times(timing_data, monkeypatch) -> None:
    raw = mne.concatenate_raws([timing_data['raw'].copy(), timing_data['raw'].copy()], verbose=False)
    spec = TRFSpanSpec(
        {'first': (1634, 4134), 'repeat': (5134, 7634)},
        wav_by_label={'first': 'stimulus.wav', 'repeat': 'stimulus.wav'},
    )
    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', lambda *args, **kwargs: timing_data['alignment'])
    with pytest.warns(UserWarning, match='Negative hw_delay_s delays'):
        epochs = trf_utils.build_trf_epochs(
            raw, spec, timing_data['path'], audio_channels=['recorded'],
            hw_delay_s=-0.2, epoch_len_s=1, verbose=False,
        )
    assert len(epochs) == 8
    assert epochs.metadata['label'].tolist() == ['first'] * 4 + ['repeat'] * 4
    np.testing.assert_allclose(epochs.metadata['wav_t_on'], [0, 1, 2, 3] * 2)
    np.testing.assert_allclose(epochs.metadata['t_on'].iloc[4:].to_numpy() - epochs.metadata['t_on'].iloc[:4], 7)
    np.testing.assert_allclose(epochs.get_data()[:4], epochs.get_data()[4:])


def test_alignment_can_use_wav_duration_without_end_sample(timing_data) -> None:
    epochs = trf_utils.build_trf_epochs(
        timing_data['raw'], TRFSpanSpec({'stimulus': (1634, None)}), timing_data['path'],
        audio_channels=['recorded'], hw_delay_s=0.2, epoch_len_s=1, verbose=False,
    )
    assert len(epochs) == 4


def test_recorded_audio_takes_precedence_over_assumed_drift(timing_data) -> None:
    step = EpochTRF(
        lambda raw: timing_data['spec'], timing_data['path'],
        audio_channels=['recorded'], realign_without_audio=True,
        epoch_len_s=1, hw_delay_s=0, verbose=False,
    )
    result = step.run(timing_data['raw'], {})
    assert result['TRF_info']['alignment_method'] == 'audio'
    assert result['data'].metadata['alignment_method'].eq('audio').all()
    assert result['data'].metadata['clock_slope'].eq(1.2).all()
    # An audio-estimation error must still fail instead of silently
    # substituting the assumed rate.
    step.gen_span_spec = lambda raw: TRFSpanSpec(
        {'missing': (1634, 4134)}, wav_by_label={'missing': 'missing.wav'},
    )
    with pytest.raises(RuntimeError, match='Audio alignment failed'):
        step.run(timing_data['raw'], {})


def test_alignment_disabled_retains_fixed_delay_processing(timing_data, monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail('Disabled realignment must not estimate alignment.')

    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', unexpected)
    epochs = trf_utils.build_trf_epochs(
        timing_data['raw'], timing_data['spec'], timing_data['path'],
        epoch_len_s=1.0, verbose=False,
    )
    assert len(epochs) == 4
    expected = timing_data['raw'].get_data(picks=['EEG 001'], start=408, stop=2408)
    np.testing.assert_array_equal(epochs.get_data(picks=['EEG 001']).reshape(-1), expected.reshape(-1))


def test_failed_alignment_can_raise_or_skip(timing_data) -> None:
    spec = TRFSpanSpec(
        {'missing': (1634, 4134), 'stimulus': (1634, 4134)},
        wav_by_label={'missing': 'missing.wav'},
    )
    step = EpochTRF(
        lambda raw: spec, timing_data['path'], audio_channels=['recorded'],
        hw_delay_s=0, epoch_len_s=1, verbose=False,
    )
    with pytest.raises(RuntimeError, match='Audio alignment failed for missing'):
        step.run(timing_data['raw'], {})
    step.on_alignment_error = 'skip'
    with pytest.warns(UserWarning, match='Skipping audio trial missing'):
        result = step.run(timing_data['raw'], {})
    alignment = result['TRF_info']['alignment_info']
    assert alignment['n_trials_found'] == 2
    assert alignment['n_trials_failed'] == 1
    assert alignment['n_trials_aligned'] == 1
    assert alignment['summary']['drift_us_per_s']['n'] == 1
    assert len(result['data']) == 4


def test_insufficient_physical_delay_margin_is_an_error(timing_data) -> None:
    with pytest.warns(UserWarning, match='Negative hw_delay_s delays'), pytest.raises(
        RuntimeError, match='Audio alignment failed'
    ) as exc:
        trf_utils.build_trf_epochs(
            timing_data['raw'], timing_data['spec'], timing_data['path'],
            audio_channels=['recorded'], hw_delay_s=-1.0, epoch_len_s=1, verbose=False,
        )
    assert 'before the Raw object' in str(exc.value.__cause__)


@pytest.mark.parametrize('kwargs, message', [
    ({'audio_channels': []}, 'audio_channels'),
    ({'audio_channels': 'recorded'}, 'audio_channels'),
    ({'alignment_kwargs': {'min_corr': 0.9}}, 'requires audio_channels'),
    ({'hw_delay_s': np.nan}, 'must be finite'),
    ({'epoch_len_s': 0}, 'must be finite'),
    ({'on_alignment_error': 'ignore'}, 'on_alignment_error'),
])
def test_invalid_configuration(timing_data, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        trf_utils.build_trf_epochs(timing_data['raw'], timing_data['spec'], timing_data['path'], **kwargs)


@pytest.fixture(params=[1.0015, 1.2], ids=['1500ppm', '20percent'])
def meg_pulse_data(timing_data, monkeypatch, request) -> dict[str, Any]:
    # At 2 kHz, 16.5 ms is exactly 33 samples. Check both a realistic clock
    # slope and the deliberately exaggerated drift used in the ordering test.
    sfreq = 2000.0
    alignment = {**timing_data['alignment'], 'raw_sfreq': sfreq, 'clock_slope': request.param}
    times = np.arange(14000) / sfreq
    wav_times_at_raw = (times - 0.8 - alignment['offset_s']) / alignment['clock_slope']
    # A WAV event at 1.5 s reaches the ear 16.5 ms later, followed by a
    # neural response with 100 ms latency. Compensation must preserve that
    # neural latency while removing only the acoustic transit time.
    response_time = 1.5 + 0.0165 + 0.100
    pulse = 1e-12 * np.exp(-0.5 * ((wav_times_at_raw - response_time) / 0.005) ** 2)
    raw = mne.io.RawArray(
        np.vstack([pulse, np.zeros(len(times))]),
        mne.create_info(['MEG 001', 'recorded'], sfreq, ['mag', 'misc']),
        first_samp=1234,
        verbose=False,
    )
    # Time-varying amplitude and frequency give both RMS and flux a meaningful
    # waveform, so a wrongly shifted feature cannot pass through being constant.
    wav_times = np.arange(32000) / 8000
    amplitude = 0.1 + np.exp(-((wav_times - 1.5) / 0.04) ** 2) + 0.5 * np.exp(-((wav_times - 2.7) / 0.1) ** 2)
    audio = amplitude * np.sin(2 * np.pi * (300 * wav_times + 150 * wav_times**2))
    wavfile.write(timing_data['wav'], 8000, audio.astype(np.float32))
    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', lambda *args, **kwargs: alignment.copy())
    return {**timing_data, 'raw': raw, 'spec': TRFSpanSpec({'stimulus': (2834, 12834)}), 'alignment': alignment}


@pytest.mark.parametrize('feature, channel', [('envelope', 'env_rms'), ('flux', 'flux')])
@pytest.mark.parametrize('realign', [True, False])
def test_default_advances_only_meg_and_negative_delay_warns(meg_pulse_data, feature, channel, realign) -> None:
    raw = meg_pulse_data['raw']
    step = EpochTRF(
        gen_span_spec=lambda raw: meg_pulse_data['spec'],
        base_audio_path=meg_pulse_data['path'],
        feature=feature,
        audio_channels=['recorded'] if realign else None,
        epoch_len_s=1.0,
        verbose=False,
    )
    assert step.hw_delay_s == 0.0165
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        advanced = step.run(raw, {})
        reference = attrs.evolve(step, hw_delay_s=0).run(raw, {})
    assert not any('hw_delay_s' in str(warning.message) for warning in caught)
    with pytest.warns(UserWarning, match='Negative hw_delay_s delays') as caught_negative:
        delayed = attrs.evolve(step, hw_delay_s=-0.0165).run(raw, {})
    assert len(caught_negative) == 1

    expected_feature = prepare_audio(str(meg_pulse_data['wav']), feature=feature, target_fs=2000)[0][0, :8000]
    assert np.ptp(expected_feature) > 0.01
    peaks = []
    for result in (reference, advanced, delayed):
        epochs = result['data']
        assert len(epochs) == 4
        # Exact equality: hardware delay never changes the feature samples.
        actual_feature = epochs.get_data(picks=[channel]).ravel()
        np.testing.assert_array_equal(actual_feature, expected_feature)
        peaks.append(int(np.argmax(epochs.get_data(picks=['MEG 001']).ravel())))
    # Independently realigning different padded intervals with a 20% drift
    # introduces up to one sample of MNE crop/resample rounding. At 1500 ppm
    # (and without realignment), require the exact 33-sample hardware shift.
    tolerance = 1 if realign and meg_pulse_data['alignment']['clock_slope'] == 1.2 else 0
    assert peaks[1] - peaks[0] == pytest.approx(-33, abs=tolerance)  # Default: compensate tube delay.
    assert peaks[2] - peaks[0] == pytest.approx(33, abs=tolerance)  # Negative: add lag.
    if realign:
        assert peaks[1] - 3000 == pytest.approx(200, abs=tolerance)  # Preserve 100 ms neural latency.
    assert advanced['TRF_info']['applied_hw_delay_s'] == 0.0165
