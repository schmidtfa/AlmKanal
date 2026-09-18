from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mne
import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt

from almkanal import AlmKanal, EpochTRF, TRFSpanSpec, preprocessing_report
from almkanal.stim_utils.alignment_utils import (
    apply_raw_wav_alignment,
    estimate_raw_wav_alignment,
    find_audio_trials,
)

RAW_SFREQ = 500.0
WAV_SFREQ = 1_000
WAV_DURATION_S = 8.0
OFFSET_S = 0.040
CLOCK_SLOPE = 1.0015

ALIGNMENT_KWARGS: dict[str, Any] = {
    'sync_sfreq': 500.0,
    'audio_band': (50.0, 220.0),
    'envelope_lowpass': 20.0,
    'window_s': 1.5,
    'step_s': 0.75,
    'max_lag_s': 0.1,
    'min_corr': 0.5,
    'min_anchors': 6,
}


@pytest.fixture(scope='module')
def synthetic_audio_alignment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    wav_times = np.arange(round(WAV_DURATION_S * WAV_SFREQ)) / WAV_SFREQ
    modulation = sosfiltfilt(
        butter(3, 8.0, fs=WAV_SFREQ, output='sos'),
        rng.normal(size=len(wav_times)),
    )
    modulation /= np.max(np.abs(modulation))
    wav = (0.55 + 0.4 * modulation) * (
        np.sin(2 * np.pi * 110 * wav_times) + 0.35 * np.sin(2 * np.pi * 173 * wav_times)
    )
    wav = (wav / np.max(np.abs(wav))).astype(np.float32)

    trigger_time_s = 0.5
    raw_duration_s = trigger_time_s + OFFSET_S + CLOCK_SLOPE * WAV_DURATION_S + 0.5
    raw_times = np.arange(round(raw_duration_s * RAW_SFREQ)) / RAW_SFREQ
    wav_time_at_raw_samples = (raw_times - trigger_time_s - OFFSET_S) / CLOCK_SLOPE
    recorded = np.interp(wav_time_at_raw_samples, wav_times, wav, left=0.0, right=0.0)
    recorded += rng.normal(scale=0.005, size=len(recorded))

    onset_sample = round(trigger_time_s * RAW_SFREQ)
    end_sample = round((trigger_time_s + OFFSET_S + CLOCK_SLOPE * WAV_DURATION_S + 0.05) * RAW_SFREQ)
    stim = np.zeros(len(raw_times))
    stim[onset_sample : onset_sample + 2] = 11
    stim[end_sample : end_sample + 2] = 99
    info = mne.create_info(
        ['EEG 001', 'MISC flat', 'audio', 'stim'],
        RAW_SFREQ,
        ['eeg', 'misc', 'misc', 'stim'],
    )
    raw = mne.io.RawArray(
        np.vstack([rng.normal(scale=1e-6, size=len(raw_times)), np.zeros(len(raw_times)), recorded, stim]),
        info,
        verbose=False,
    )
    annotation_onset = trigger_time_s + OFFSET_S + CLOCK_SLOPE * 2.0
    raw.set_annotations(mne.Annotations([annotation_onset], [0.2 * CLOCK_SLOPE], ['inside trial']))

    wav_path = Path(tmp_path_factory.mktemp('audio_alignment')) / 'stimulus.wav'
    wavfile.write(wav_path, WAV_SFREQ, wav)
    raw_trial = raw.copy().crop(trigger_time_s, end_sample / RAW_SFREQ)
    alignment = estimate_raw_wav_alignment(
        raw_trial,
        wav_path,
        audio_channels=['MISC flat', 'audio'],
        verbose=False,
        **ALIGNMENT_KWARGS,
    )
    return {
        'raw': raw,
        'wav': wav,
        'wav_times': wav_times,
        'wav_path': wav_path,
        'onset_sample': onset_sample,
        'alignment': alignment,
    }


def test_find_audio_trials_handles_initial_and_unrelated_events() -> None:
    stim = np.array([11, 11, 0, 7, 0, 99, 99, 0], dtype=float)
    raw = mne.io.RawArray(
        stim[np.newaxis],
        mne.create_info(['trigger'], 100.0, ['stim']),
        verbose=False,
    )

    trials = find_audio_trials(raw, {11: 'trial.wav'}, cast('int', np.int64(99)))

    assert trials == [
        {
            'onset_code': 11,
            'wav_file': 'trial.wav',
            'onset_sample': 0,
            'end_code': 99,
            'end_sample': 5,
        }
    ]


def test_estimation_and_application_correct_known_drift(synthetic_audio_alignment: dict[str, Any]) -> None:
    alignment = synthetic_audio_alignment['alignment']
    assert alignment['selected_recorded_channel'] == 'audio'
    assert alignment['offset_s'] == pytest.approx(OFFSET_S, abs=0.002)
    assert alignment['clock_slope'] == pytest.approx(CLOCK_SLOPE, abs=0.0003)
    assert alignment['median_correlation'] > 0.95
    assert alignment['n_anchor_inliers'] >= ALIGNMENT_KWARGS['min_anchors']
    assert 'candidate_times_s' not in alignment

    aligned = apply_raw_wav_alignment(
        synthetic_audio_alignment['raw'],
        synthetic_audio_alignment['onset_sample'],
        alignment,
        verbose=False,
    )

    assert aligned.ch_names == synthetic_audio_alignment['raw'].ch_names
    assert aligned.n_times == round(WAV_DURATION_S * RAW_SFREQ)
    annotation_index = list(aligned.annotations.description).index('inside trial')
    annotation_onset = aligned.annotations.onset[annotation_index] - aligned.first_time
    assert annotation_onset == pytest.approx(2.0, abs=0.004)
    assert aligned.annotations.duration[annotation_index] == pytest.approx(0.2, abs=0.002)

    recorded = aligned.get_data(picks=['audio'])[0]
    wav_at_raw_sfreq = np.interp(
        np.arange(aligned.n_times) / RAW_SFREQ,
        synthetic_audio_alignment['wav_times'],
        synthetic_audio_alignment['wav'],
    )
    bandpass = butter(3, [50.0, 220.0], btype='bandpass', fs=RAW_SFREQ, output='sos')
    recorded_envelope = np.abs(hilbert(sosfiltfilt(bandpass, recorded)))
    wav_envelope = np.abs(hilbert(sosfiltfilt(bandpass, wav_at_raw_sfreq)))
    assert np.corrcoef(recorded_envelope, wav_envelope)[0, 1] > 0.75


@pytest.mark.parametrize('missing_end', [False, True])
def test_epoch_trf_realigns_audio_and_records_diagnostics(
    synthetic_audio_alignment: dict[str, Any],
    tmp_path: Path,
    missing_end: bool,
) -> None:
    wav_path = synthetic_audio_alignment['wav_path']
    step = EpochTRF(
        gen_span_spec=lambda raw: TRFSpanSpec.from_events(
            raw, {11: wav_path.name}, 99, stim_channel='stim',
            infer_missing_ends=missing_end, base_audio_path=wav_path.parent,
        ),
        base_audio_path=wav_path.parent,
        audio_channels=['MISC flat', 'audio'],
        alignment_kwargs=ALIGNMENT_KWARGS,
        epoch_len_s=1.0,
        verbose=False,
    )
    raw = synthetic_audio_alignment['raw'].copy()
    if missing_end:
        raw._data[-1, raw._data[-1] == 99] = 0
    original = raw.get_data().copy()
    pipeline = AlmKanal(steps=[step])
    epochs, report = pipeline.run(raw)
    trf_info = pipeline.info['steps_info']['EpochTRF']['TRF_info']
    alignment = trf_info['alignment_info']

    assert len(epochs) == 8
    assert epochs.get_data().shape[-1] == 500
    assert 'env_rms' in epochs.ch_names
    assert alignment['n_trials_aligned'] == 1
    assert alignment['n_trials_failed'] == 0
    assert alignment['n_trials_end_inferred'] == int(missing_end)
    assert alignment['trials'][0]['drift_us_per_s'] == pytest.approx(1500, abs=300)
    assert alignment['summary']['offset_ms']['mean'] == pytest.approx(40, abs=2)
    assert trf_info['hw_delay_s'] == 0.0165
    assert trf_info['applied_hw_delay_s'] == 0.016
    assert trf_info['alignment_kwargs']['min_anchors'] == 6
    assert epochs.metadata['stimulus'].unique().tolist() == ['stimulus']
    np.testing.assert_array_equal(raw.get_data(), original)

    assert 'TRF audio realignment' in report.get_contents()[0]
    json_path = tmp_path / 'pipeline.json'
    pipeline.generate_json(str(json_path))
    methods = preprocessing_report([json_path], tmp_path / 'methods.md').read_text()
    assert '1 of 1 trials were successfully aligned' in methods
    assert 'physical-delay correction of 16.500 ms was applied after realignment' in methods
    assert f"signed clock drift {alignment['trials'][0]['drift_us_per_s']:.3f}" in methods
    if missing_end:
        trial = alignment['trials'][0]
        assert trial['original_end_sample'] is None
        assert trial['inferred_end_sample'] == synthetic_audio_alignment['onset_sample'] + 4002
        assert trial['end_inference_drift_us_per_s'] == 499
        assert epochs.metadata['end_inferred'].all()
        assert epochs.metadata['end_code'].isna().all()
        assert epochs.metadata['end_inference_drift_us_per_s'].eq(499).all()
        # The known synthetic drift is 1500 us/s: inference must not impose 499.
        assert epochs.metadata['drift_us_per_s'].iloc[0] == pytest.approx(1500, abs=300)
        assert 'Trial endpoints were inferred for 1 trial without end triggers' in methods
        assert 'assumed rates: 499.000 µs/s for 1 trial' in methods
        assert 'actual alignment offset and drift were subsequently estimated from the audio' in methods
        assert '499.000 ± 0.000 µs/s' not in methods


def test_span_spec_preserves_repeated_wavs_and_event_order() -> None:
    stim = np.zeros(30)
    stim[[0, 10, 20]] = [11, 12, 11]
    stim[[8, 18, 28]] = 99
    raw = mne.io.RawArray(
        stim[None], mne.create_info(['trigger'], 100.0, ['stim']), first_samp=100, verbose=False,
    )
    spec = TRFSpanSpec.from_events(raw, {11: 'same.wav', 12: 'other.wav'}, 99)
    assert list(spec.spans_by_label.values()) == [(100, 108), (110, 118), (120, 128)]
    assert list(spec.wav_by_label.values()) == ['same.wav', 'other.wav', 'same.wav']
    assert [meta['onset_code'] for meta in spec.metadata_by_label.values()] == [11, 12, 11]
