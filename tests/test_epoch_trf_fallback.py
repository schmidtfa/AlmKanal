from __future__ import annotations

from contextlib import nullcontext

import attrs
import mne
import numpy as np
import pytest
from scipy.io import wavfile

from almkanal import AlmKanal, EpochTRF, TRFSpanSpec, preprocessing_report
from almkanal.almkanal_steps import trf_utils
from almkanal.report.methods_context import build_context_from_files
from almkanal.stim_utils.alignment_utils import assume_raw_wav_alignment
from almkanal.stim_utils.audio_utils import prepare_audio


@pytest.fixture
def no_audio_trial(tmp_path, monkeypatch):
    sfreq = 1000.0
    duration = 20.0
    onset = 0.5
    wav = tmp_path / 'original.wav'
    wav_times = np.arange(40000) / 2000
    signal = (0.5 + 0.4 * np.sin(2 * np.pi * 2 * wav_times)) * np.sin(2 * np.pi * 170 * wav_times)
    # Stereo duration must be derived from frames, not individual samples.
    wavfile.write(wav, 2000, np.column_stack([signal, signal * 0.5]).astype(np.float32))

    def unexpected(*args, **kwargs):
        pytest.fail('No recorded audio is available for cross-correlation.')

    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', unexpected)

    def make_trial(rate=499.0, repeated=False):
        slope = 1.0 + rate / 1e6
        raw_times = np.arange(round((onset + slope * duration + 1.0) * sfreq)) / sfreq
        neural_times = (raw_times - onset) / slope
        # A ramp exposes accumulated timing error, while a localized response
        # confirms that the MEG samples really move during resampling.
        response = 1e-12 * np.exp(-0.5 * ((neural_times - 18.5) / 0.008) ** 2)
        raw = mne.io.RawArray(
            np.vstack([neural_times * 1e-12, response]),
            mne.create_info(['MEG 001', 'MEG 002'], sfreq, ['mag', 'mag']),
            first_samp=1234, verbose=False,
        )
        first_on = raw.first_samp + round(onset * sfreq)
        spans = {'first': (first_on, None)}
        if repeated:
            second_on = first_on + raw.n_times
            raw = mne.concatenate_raws([raw.copy(), raw.copy()], verbose=False)
            spans['repeat'] = (second_on, None)
        spec = TRFSpanSpec(spans, wav_by_label={label: wav for label in spans})
        step = EpochTRF(
            lambda raw: spec, tmp_path, realign_without_audio=True,
            fallback_drift_us_per_s=rate, epoch_len_s=1, verbose=False,
        )
        return raw, spec, step

    return make_trial, wav


@pytest.mark.parametrize('rate', [499.0, -499.0, 0.0, 200000.0])
@pytest.mark.parametrize('delay', [0.0, 0.0165, -0.0165])
def test_assumed_drift_resamples_meg_before_delay(no_audio_trial, rate, delay):
    make_trial, wav = no_audio_trial
    raw, spec, step = make_trial(rate)
    original = raw.get_data().copy()
    step = attrs.evolve(step, hw_delay_s=delay)
    with pytest.warns(UserWarning, match='Negative hw_delay_s delays') if delay < 0 else nullcontext():
        result = step.run(raw, {})
    epochs = result['data']
    applied_delay = round(delay * 1000) / 1000
    assert len(epochs) == 20
    assert epochs.info['sfreq'] == 1000
    expected = np.arange(20000) / 1000 + applied_delay
    actual = epochs.get_data(picks=['MEG 001']).ravel() / 1e-12
    np.testing.assert_allclose(actual[100:-100], expected[100:-100], atol=0.0015)
    peak = np.argmax(epochs.get_data(picks=['MEG 002']).ravel())
    assert peak == pytest.approx(round((18.5 - applied_delay) * 1000), abs=1)
    feature = prepare_audio(str(wav), target_fs=1000)[0][0, :20000]
    np.testing.assert_array_equal(epochs.get_data(picks=['env_rms']).ravel(), feature)
    np.testing.assert_array_equal(raw.get_data(), original)
    slope = 1 + rate / 1e6
    np.testing.assert_allclose(epochs.metadata['t_on'], 0.5 + slope * (np.arange(20) + applied_delay))
    assert epochs.metadata['alignment_method'].eq('assumed_drift').all()
    assert epochs.metadata['alignment_offset_s'].eq(0).all()
    assert epochs.metadata['drift_us_per_s'].eq(rate).all()
    info = result['TRF_info']
    trial = info['alignment_info']['trials'][0]
    assert info['realign_audio'] is True
    assert info['alignment_method'] == 'assumed_drift'
    assert trial['wav_duration_s'] == 20
    assert trial['total_drift_ms'] == pytest.approx(rate * 20 / 1000)
    assert trial['residual_rms_ms'] is None
    assert trial['median_correlation'] is None
    assert trial['n_anchor_inliers'] == 0
    assert 'residual_rms_ms' not in info['alignment_info']['summary']


def test_public_builder_default_rate_and_repeated_trials(no_audio_trial):
    make_trial, wav = no_audio_trial
    raw, spec, step = make_trial(repeated=True)
    epochs = trf_utils.build_trf_epochs(
        raw, spec, wav.parent, realign_without_audio=True,
        hw_delay_s=0, epoch_len_s=1, verbose=False,
    )
    assert EpochTRF(lambda raw: spec, wav.parent).fallback_drift_us_per_s == 499
    assert len(epochs) == 40
    assert epochs.metadata['drift_us_per_s'].eq(499).all()
    np.testing.assert_array_equal(epochs.get_data()[:20], epochs.get_data()[20:])
    np.testing.assert_array_equal(epochs.metadata['wav_t_on'], list(range(20)) * 2)
    second_onset = (spec.spans_by_label['repeat'][0] - raw.first_samp) / 1000
    assert epochs.metadata.iloc[20]['t_on'] == second_onset


@pytest.mark.parametrize('preserve', [True, False])
def test_fallback_annotations_and_inferred_end_metadata(no_audio_trial, preserve):
    make_trial, wav = no_audio_trial
    raw, spec, step = make_trial()
    slope = 1.000499
    raw.set_annotations(mne.Annotations([0.5 + slope * 10.5], [0.1], ['BAD movement']))
    # Simulate an inferred end from the event helper, preserving its provenance.
    end = spec.spans_by_label['first'][0] + round(20 * slope * 1000)
    spec.spans_by_label['first'] = (spec.spans_by_label['first'][0], end)
    spec.metadata_by_label['first'] = {
        'end_inferred': True, 'end_inference_drift_us_per_s': 499.0,
        'end_inference_wav_duration_s': 20.0,
    }
    result = attrs.evolve(step, hw_delay_s=0, preserve_annotations=preserve).run(raw, {})
    epochs = result['data']
    kept = [index for index in range(20) if not preserve or index != 10]
    assert epochs.metadata['epoch_index_in_segment'].tolist() == kept
    np.testing.assert_array_equal(epochs.metadata['wav_t_on'], kept)
    assert epochs.metadata['end_inferred'].all()
    trial = result['TRF_info']['alignment_info']['trials'][0]
    assert trial['original_end_sample'] is None
    assert trial['inferred_end_sample'] == end
    assert len(raw.annotations) == 1


@pytest.mark.parametrize('failure', ['missing_wav', 'short_recording'])
def test_fallback_errors_can_raise_or_skip(no_audio_trial, failure):
    make_trial, wav = no_audio_trial
    raw, spec, step = make_trial()
    spec.spans_by_label['bad'] = (raw.first_samp + (21000 if failure == 'short_recording' else 500), None)
    spec.wav_by_label['bad'] = wav if failure == 'short_recording' else wav.with_name('missing.wav')
    with pytest.raises(RuntimeError, match='Audio alignment failed for bad'):
        step.run(raw, {})
    with pytest.warns(UserWarning, match='Skipping audio trial bad'):
        result = attrs.evolve(step, on_alignment_error='skip').run(raw, {})
    info = result['TRF_info']['alignment_info']
    assert info['n_trials_aligned'] == 1
    assert info['n_trials_failed'] == 1
    assert len(result['data']) == 20


@pytest.mark.parametrize('rate', [np.nan, np.inf, -np.inf, -1e6, -2e6])
def test_invalid_fallback_rate(no_audio_trial, rate):
    make_trial, _ = no_audio_trial
    raw, spec, step = make_trial()
    with pytest.raises(ValueError, match='fallback_drift_us_per_s'):
        attrs.evolve(step, fallback_drift_us_per_s=rate).run(raw, {})


def test_empty_wav_is_rejected(tmp_path):
    wav = tmp_path / 'empty.wav'
    wavfile.write(wav, 1000, np.empty(0, dtype=np.float32))
    with pytest.raises(ValueError, match='WAV duration'):
        assume_raw_wav_alignment(wav, 1000)


@pytest.mark.parametrize('inferred_end', [False, True])
def test_fallback_reports_and_json_identify_assumptions(no_audio_trial, tmp_path, inferred_end):
    make_trial, _ = no_audio_trial
    raw, spec, step = make_trial()
    if inferred_end:
        spec.metadata_by_label['first'] = {
            'end_inferred': True, 'end_inference_drift_us_per_s': 499.0,
            'end_inference_wav_duration_s': 20.0,
        }
        spec.spans_by_label['first'] = (1734, 21744)
    pipeline = AlmKanal(steps=[step])
    epochs, report = pipeline.run(raw)
    assert 'TRF audio realignment' in report.get_contents()[0]
    html_path = tmp_path / 'report.html'
    report.save(html_path, open_browser=False, overwrite=True)
    html = html_path.read_text()
    assert 'Assumed drift correction: 499' in html
    assert 'drift and offset were not measured' in html
    json_path = tmp_path / 'pipeline.json'
    pipeline.generate_json(str(json_path))
    context = build_context_from_files([json_path])
    assert context.steps[0].settings['alignment_method'] == 'assumed_drift'
    assert context.steps[0].results['metrics']['drift_us_per_s']['mean'] == 499
    methods = preprocessing_report([json_path], tmp_path / 'methods.md').read_text()
    assert 'assumed signed drift of 499.000 µs/s' in methods
    assert 'zero offset' in methods
    assert '1 of 1 trials were successfully aligned' in methods
    assert 'physical-delay correction of 16.500 ms was applied after realignment' in methods
    assert 'normalized cross-correlation' not in methods
    assert 'mean ± population SD' not in methods
    assert 'actual alignment offset and drift were subsequently estimated from the audio' not in methods
    if inferred_end:
        assert 'Trial endpoints were inferred for 1 trial without end triggers' in methods
