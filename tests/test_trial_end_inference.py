import mne
import numpy as np
import pytest
from scipy.io import wavfile

from almkanal import TRFSpanSpec
from almkanal.stim_utils.alignment_utils import find_audio_trials


@pytest.fixture
def onset_only_trials(tmp_path):
    # Initial onset, unrelated event, repeated WAV, and a nonzero first_samp.
    stim = np.zeros(20_000)
    stim[[0, 2000, 10_000]] = [11, 7, 11]
    raw = mne.io.RawArray(
        stim[None],
        mne.create_info(['trigger'], 1000, ['stim']),
        first_samp=1234,
        verbose=False,
    )
    path = tmp_path / 'trial.wav'
    # Stereo float WAV: duration is frames / sampling rate, independent of channels.
    wavfile.write(path, 8000, np.zeros((64_000, 2), dtype=np.float32))
    return raw, path


@pytest.mark.parametrize('end_triggers', [None, (), (2, 102)])
def test_infers_all_missing_ends_without_changing_events(onset_only_trials, end_triggers):
    raw, path = onset_only_trials
    original = raw.get_data().copy()
    spec = TRFSpanSpec.from_events(
        raw,
        {11: path.name},
        end_triggers,
        infer_missing_ends=True,
        base_audio_path=path.parent,
    )
    # 8 seconds at +499 us/s -> 8004 raw samples, after rounding at 1000 Hz.
    assert spec.spans_by_label == {'trial_000': (1234, 9238), 'trial_001': (11234, 19238)}
    assert list(spec.wav_by_label.values()) == [path.name, path.name]
    for metadata in spec.metadata_by_label.values():
        assert metadata == {
            'stimulus': 'trial',
            'onset_code': 11,
            'end_code': None,
            'end_inferred': True,
            'end_inference_drift_us_per_s': 499.0,
            'end_inference_wav_duration_s': 8.0,
        }
    np.testing.assert_array_equal(raw.get_data(), original)


@pytest.mark.parametrize('drift, duration_samples', [(0.0, 8000), (1000.0, 8008), (-1000.0, 7992)])
def test_custom_signed_drift_and_absolute_wav_paths(onset_only_trials, drift, duration_samples):
    raw, path = onset_only_trials
    spec = TRFSpanSpec.from_events(
        raw,
        {11: path},
        infer_missing_ends=True,
        fallback_drift_us_per_s=drift,
    )
    assert all(off - on == duration_samples for on, off in spec.spans_by_label.values())


@pytest.mark.parametrize('missing_first', [False, True])
def test_actual_end_triggers_take_precedence_in_mixed_trials(onset_only_trials, missing_first):
    raw, path = onset_only_trials
    offset = 10_000 if missing_first else 0
    raw._data[0, 8050 + offset] = 99
    spec = TRFSpanSpec.from_events(
        raw,
        {11: path},
        99,
        infer_missing_ends=True,
    )
    actual_index = int(missing_first)
    actual_label = f'trial_{actual_index:03d}'
    inferred_label = f'trial_{1 - actual_index:03d}'
    assert spec.spans_by_label[actual_label] == (1234 + offset, 9284 + offset)
    assert spec.metadata_by_label[actual_label]['end_code'] == 99
    assert spec.metadata_by_label[actual_label]['end_inferred'] is False
    assert spec.metadata_by_label[actual_label]['end_inference_drift_us_per_s'] is None
    assert spec.metadata_by_label[inferred_label]['end_inferred'] is True


def test_complete_trials_do_not_need_wav_files_to_find_spans(onset_only_trials):
    raw, _ = onset_only_trials
    raw._data[0, [8050, 18050]] = 99
    trials = find_audio_trials(raw, {11: 'does-not-exist.wav'}, 99, infer_missing_ends=True)
    assert [trial['end_sample'] for trial in trials] == [9284, 19284]


def test_existing_strict_behavior_remains_the_default(onset_only_trials):
    raw, _ = onset_only_trials
    with pytest.raises(RuntimeError, match='New audio onset before'):
        find_audio_trials(raw, {11: 'trial.wav'}, 99)
    with pytest.warns(UserWarning, match='final audio trial has no end trigger and was ignored'):
        assert find_audio_trials(raw.copy().crop(tmax=9), {11: 'trial.wav'}, 99) == []
    with pytest.raises(ValueError, match='unless infer_missing_ends=True'):
        find_audio_trials(raw, {11: 'trial.wav'})


@pytest.mark.parametrize('drift', [np.nan, np.inf, -np.inf, -1e6, -2e6])
def test_invalid_drift_is_rejected(onset_only_trials, drift):
    raw, path = onset_only_trials
    with pytest.raises(ValueError, match='fallback_drift_us_per_s'):
        find_audio_trials(raw, {11: path}, infer_missing_ends=True, fallback_drift_us_per_s=drift)


def test_missing_wav_or_base_path_is_an_explicit_error(onset_only_trials):
    raw, path = onset_only_trials
    with pytest.raises(ValueError, match='base_audio_path is required'):
        find_audio_trials(raw, {11: path.name}, infer_missing_ends=True)
    with pytest.raises(FileNotFoundError, match='WAV file does not exist'):
        find_audio_trials(raw, {11: path.with_name('missing.wav')}, infer_missing_ends=True)


@pytest.mark.parametrize('boundary', ['next-onset', 'recording-end'])
def test_inference_does_not_extend_past_available_trial(onset_only_trials, boundary):
    raw, path = onset_only_trials
    if boundary == 'next-onset':
        raw._data[0, 4000] = 11
    else:
        raw = raw.copy().crop(tmax=4)
    with pytest.raises(RuntimeError, match='exceeds the next audio onset or recording end'):
        find_audio_trials(raw, {11: path}, infer_missing_ends=True)


def test_inference_at_exclusive_recording_boundary_is_crop_safe(onset_only_trials):
    raw, path = onset_only_trials
    raw.crop(tmax=8.003)
    spec = TRFSpanSpec.from_events(raw, {11: path}, infer_missing_ends=True)
    on, off = spec.spans_by_label['trial_000']
    assert (on, off) == (1234, 9237)
    trial = raw.copy().crop((on - raw.first_samp) / 1000, (off - raw.first_samp) / 1000)
    assert trial.n_times == 8004


@pytest.mark.parametrize('n_samples, drift', [(0, 499), (1, -999_999)])
def test_empty_or_subsample_wav_duration_is_rejected(onset_only_trials, n_samples, drift):
    raw, path = onset_only_trials
    wavfile.write(path, 8000, np.zeros(n_samples, dtype=np.float32))
    with pytest.raises(ValueError, match='duration'):
        find_audio_trials(raw, {11: path}, infer_missing_ends=True, fallback_drift_us_per_s=drift)
