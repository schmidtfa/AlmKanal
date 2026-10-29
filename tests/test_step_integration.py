"""Exercise step combinations and profile-sensitive paths on small recordings."""

from unittest.mock import Mock

import mne
import numpy as np
import pytest

import almkanal as ak
from almkanal.info import AlmKanalInfo
from almkanal.report.methods_context import build_context_from_files
from almkanal.report.rendering import render_methods
import pandas as pd

@pytest.fixture(autouse=True)
def isolated_defaults():
    with ak.use_defaults():
        yield


def test_removed_epoch_decimation_is_rejected():
    original = ak.get_defaults()
    with pytest.raises(ValueError, match='decim'):
        ak.set_defaults(epochs={'decim': 2})
    assert ak.get_defaults() is original


@pytest.mark.parametrize('method', ['fft', 'polyphase'])
def test_events_epochs_and_resampling_use_captured_settings(tmp_path, method):
    sfreq = 200
    samples = 2000
    rng = np.random.default_rng(42)
    signals = rng.normal(scale=1e-6, size=(5, samples))
    signals[-1] = 0
    onsets = [400, 800, 1200, 1600]
    for onset in onsets:
        signals[-1, onset : onset + 5] = 1
    raw = mne.io.RawArray(
        signals,
        mne.create_info(['F3', 'F4', 'P3', 'P4', 'TRIGGER'], sfreq, ['eeg'] * 4 + ['stim']),
        first_samp=100,
    )
    raw.set_montage('standard_1020')

    with ak.use_defaults(
        name='study-events',
        filter={'highpass': 0.5, 'lowpass': 20.0, 'method': 'iir'},
        events={'stim_channel': 'TRIGGER', 'min_duration': 0.05},
        epochs={'tmin': -0.1, 'tmax': 0.3, 'baseline': (None, 0.0)},
        resample={'method': method},
    ):
        pipeline = ak.AlmKanal(
            [
                ak.Filter(),
                ak.Events(min_duration=0),
                ak.Epochs(baseline=None),
                ak.Resample(sfreq=100),
            ]
        )

    ak.set_defaults('salzburg')
    processed, report = pipeline.run(raw)
    expected_events = np.column_stack([np.array(onsets) + 100, np.zeros(4, dtype=int), np.ones(4, dtype=int)])
    np.testing.assert_array_equal(processed.events, expected_events)
    assert len(processed) == len(onsets)
    assert processed.info['sfreq'] == 100
    assert processed.baseline is None
    assert processed.tmin == pytest.approx(-0.1)
    assert ak.get_defaults().name == 'salzburg'
    assert len(report) > 0

    path = tmp_path / 'pipeline.json'
    saved = pipeline.generate_json(str(path))['processing_history']
    assert all(entry['info']['defaults']['profile'] == 'study-events' for entry in saved)
    assert saved[1]['info']['event_info']['min_duration'] == 0
    assert saved[2]['info']['epochs_info']['baseline'] is None
    assert saved[3]['info']['resample_info']['method'] == method
    text = render_methods(build_context_from_files([path]))
    assert 'Epoching' in text
    assert '100' in text


@pytest.mark.parametrize('destination', [None, [0.0, 0.0, 0.04]], ids=['median', 'explicit'])
@pytest.mark.parametrize('compensation', [False, 'external'], ids=['disabled', 'external'])
def test_multiblock_maxwell_defaults_concatenation_and_methods(monkeypatch, tmp_path, destination, compensation):
    blocks = []
    for index, position in enumerate([0.01, 0.03]):
        raw = mne.io.RawArray(
            np.full((2, 200), (index + 1) * 1e-12),
            mne.create_info(['MEG001', 'MEG002'], 100, 'mag'),
        )
        transform = np.eye(4)
        transform[2, 3] = position
        raw.info['dev_head_t'] = mne.transforms.Transform('meg', 'head', transform)
        blocks.append(raw)
    expected_data = np.concatenate([block.get_data() / 2 for block in blocks], axis=1)
    expected_destination = [0.0, 0.0, 0.02] if destination is None else destination
    calibration = 'calibration.dat' if compensation else False
    cross_talk = 'cross-talk.fif' if compensation else False
    calls = []

    def maxwell_filter(raw, **kwargs):
        calls.append(kwargs)
        result = raw.copy().apply_function(lambda values: values / 2)
        transform = np.eye(4)
        transform[:3, 3] = kwargs['destination']
        result.info['dev_head_t'] = mne.transforms.Transform('meg', 'head', transform)
        return result

    # The single-block test exercises MNE's numerical SSS implementation. Here
    # isolate it while exercising block alignment, forwarding, concatenation,
    # pipeline history, and methods rendering together.
    monkeypatch.setattr(mne.preprocessing, 'find_bad_channels_maxwell', lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(mne.preprocessing, 'maxwell_filter', maxwell_filter)
    monkeypatch.setattr('almkanal.almkanal.mne.Report', Mock())
    with ak.use_defaults(
        name='study-blocks',
        maxwell={
            'mw_calibration_file': calibration,
            'mw_cross_talk_file': cross_talk,
            'mw_st_duration': 1.0,
            'mw_st_correlation': 0.9,
        },
    ):
        step = ak.MultiBlockMaxwell(mw_destination=destination, mw_st_correlation=0.95)
        pipeline = ak.AlmKanal([step])

    processed, _ = pipeline.run(blocks)
    np.testing.assert_array_equal(processed.get_data(), expected_data)
    assert len(calls) == 2
    for call in calls:
        np.testing.assert_allclose(call['destination'], expected_destination)
        assert call['calibration'] == calibration
        assert call['cross_talk'] == cross_talk
        assert call['st_duration'] == 1.0
        assert call['st_correlation'] == 0.95

    path = tmp_path / 'blocks.json'
    saved = pipeline.generate_json(str(path))['processing_history'][0]['info']
    assert saved['defaults']['profile'] == 'study-blocks'
    assert saved['maxwell_info']['n_blocks'] == 2
    context = build_context_from_files([path])
    assert context.steps[0].settings['calibration_applied'] is bool(compensation)
    assert context.steps[0].settings['cross_talk_applied'] is bool(compensation)
    text = render_methods(context)
    assert '0.95' in text
    assert ('Fine-calibration information was applied.' in text) is bool(compensation)
    assert ('Cross-talk compensation was applied.' in text) is bool(compensation)


@pytest.mark.parametrize('fit_only', [False, True])
def test_salzburg_train_ica_runs_on_synthetic_recording(fit_only):
    sfreq = 200
    times = np.arange(24 * sfreq) / sfreq
    rng = np.random.default_rng(42)
    sources = np.array(
        [
            np.sin(2 * np.pi * 16.666 * times),
            np.sin(2 * np.pi * 9 * times),
            rng.laplace(size=len(times)),
        ]
    )
    mixing = np.array([[1.0, 0.3, 0.2], [0.2, 1.0, 0.4], [0.4, 0.2, 1.0]])
    raw = mne.io.RawArray(mixing @ sources * 1e-6, mne.create_info(['F3', 'F4', 'P3'], sfreq, 'eeg'))
    original = raw.get_data().copy()
    disabled_raw = raw.copy()
    # Two-second windows give the pure sinusoid a spectral peak within the
    # detector's accepted width range. This helper-only option must also remain
    # active during run(), after leaving the profile's scope.
    with ak.use_defaults(
        'salzburg',
        ica={'n_components': 3, 'eog': False, 'ecg': False, 'emg': False},
        ica_train={'duration': 2},
    ):
        step = ak.ICA(fit_only=fit_only)
        disabled = ak.ICA(train=False, fit_only=True)

    result = step.run(raw, AlmKanalInfo())
    assert result['ica_info']['components_dict']['train']
    assert result['defaults']['profile'] == 'salzburg'
    assert ak.get_defaults().name == 'generic'
    if fit_only:
        np.testing.assert_array_equal(result['data'].get_data(), original)
        assert result['ica_info']['applied_exclude'] == []
    else:
        assert result['ica_info']['applied_exclude']
        train_wave = sources[0]
        before = np.linalg.norm(original @ train_wave)
        after = np.linalg.norm(result['data'].get_data() @ train_wave)
        assert after < before * 0.1

    result = disabled.run(disabled_raw, AlmKanalInfo())
    assert 'train' not in result['ica_info']['components_dict']
    assert result['ica_info']['applied_exclude'] == []
    np.testing.assert_array_equal(result['data'].get_data(), original)
