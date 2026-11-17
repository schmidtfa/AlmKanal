"""Regression tests for replaying processing history on empty-room recordings.

These tests use synthetic data; they do not require a MEG dataset or calibration
files. The Maxwell and ICA-specific operations are isolated with test doubles.
"""

from copy import deepcopy
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from almkanal.almkanal_steps import spatial_filter_utils
from almkanal.info import AlmKanalInfo, StepInfo


def _raw() -> mne.io.BaseRaw:
    """Return deterministic synthetic MEG plus a channel to be dropped."""
    sfreq = 200.0
    times = np.arange(4000) / sfreq
    samples = np.array(
        [
            1e-12 * (np.sin(2 * np.pi * 10 * times) + 0.5 * np.sin(2 * np.pi * 70 * times)),
            1e-12 * (np.cos(2 * np.pi * 8 * times) + 0.3 * np.sin(2 * np.pi * 60 * times)),
            np.zeros(len(times)),
        ]
    )
    info = mne.create_info(['MEG0111', 'MEG0112', 'STI101'], sfreq, ['mag', 'mag', 'stim'])
    return mne.io.RawArray(samples, info, verbose=False)


def _filter_history(l_freq: float | None, h_freq: float | None) -> StepInfo:
    """Model the effective arguments and report-only fields produced by Filter.run."""
    return StepInfo(
        step='Filter',
        info={
            'filter_info': {
                'l_freq': l_freq,
                'h_freq': h_freq,
                'method': 'iir',
                'iir_params': {'order': 4, 'ftype': 'butter', 'output': 'sos'},
                'picks': None,
                'applied_to': 'continuous',
                'filter_order': None,
                'filter_length_seconds': None,
                'filter_length_samples': None,
            }
        },
    )


def test_empty_room_replays_repeated_filters_and_resampling_in_order() -> None:
    """Only executable kwargs may reach MNE; steps must retain their order."""
    raw_er = _raw()
    recording = _raw()
    picks = ['MEG0111', 'MEG0112']
    history = [
        StepInfo('Events', {'events_info': {'n_events': 10}}),
        _filter_history(l_freq=None, h_freq=40.0),
        StepInfo('Epochs', {'epochs_info': {'n_epochs': 5}}),
        StepInfo(
            'Resample',
            {
                'resample_info': {
                    'sfreq': 100.0,
                    'npad': 'auto',
                    'window': 'boxcar',
                    'pad': 'reflect_limited',
                    'method': 'fft',
                    'original_sfreq': 200.0,
                    'applied_to': 'continuous',
                }
            },
        ),
        _filter_history(l_freq=1.0, h_freq=None),
        StepInfo('ForwardModel', {'fwd_info': {'created': True}}),
    ]
    history_before = deepcopy(history)

    # Independent reference pipeline: do not derive arguments by applying the
    # same exclusion logic as preproc_empty_room, which could hide a regression.
    expected = raw_er.copy().pick(picks)
    iir_params = {'order': 4, 'ftype': 'butter', 'output': 'sos'}
    expected.filter(None, 40.0, method='iir', iir_params=iir_params, verbose=False)
    expected.resample(100.0, npad='auto', window='boxcar', pad='reflect_limited', method='fft', verbose=False)
    expected.filter(1.0, None, method='iir', iir_params=iir_params, verbose=False)

    result = spatial_filter_utils.preproc_empty_room(
        raw_er=raw_er,
        data=recording,
        preproc_info=history,
        picks=picks,
    )

    assert result is raw_er
    assert result.ch_names == picks
    assert result.info['sfreq'] == pytest.approx(100.0)
    assert result.info['highpass'] == pytest.approx(1.0)
    assert result.info['lowpass'] == pytest.approx(40.0)
    np.testing.assert_allclose(result.get_data(), expected.get_data(), rtol=1e-10, atol=1e-20)
    assert history == history_before  # Replay must not remove fields from the record.


@pytest.mark.parametrize('use_epochs', [False, True])
def test_empty_room_maxwell_discards_only_reporting_fields(monkeypatch, use_epochs: bool) -> None:
    """Maxwell receives valid settings, with either raw or epoched study data."""
    raw_er = _raw()
    recording: mne.io.BaseRaw | mne.BaseEpochs = _raw().pick(['MEG0111', 'MEG0112'])
    if use_epochs:
        recording = mne.EpochsArray(
            np.zeros((2, 2, 100)),
            info=recording.info,
            verbose=False,
        )
    maxwell_info = {
        'coord_frame': 'meg',
        'st_duration': None,
        'calibration_file': 'fine_cal.dat',
        'cross_talk_file': 'crosstalk.fif',
        'calibration_applied': True,
        'cross_talk_applied': True,
    }
    history = [StepInfo('Maxwell', {'maxwell_info': maxwell_info})]
    passed_to_prepare = []
    passed_to_maxwell = []

    def fake_prepare(*, raw_er, raw):
        passed_to_prepare.append((raw_er, raw))
        return raw_er

    def fake_run_maxwell(raw, **kwargs):
        passed_to_maxwell.append((raw, kwargs))
        return raw

    monkeypatch.setattr(mne.preprocessing, 'maxwell_filter_prepare_emptyroom', fake_prepare)
    monkeypatch.setattr(spatial_filter_utils, 'run_maxwell', fake_run_maxwell)

    result = spatial_filter_utils.preproc_empty_room(
        raw_er=raw_er,
        data=recording,
        preproc_info=history,
        picks=['MEG0111', 'MEG0112'],
    )

    assert result is raw_er
    assert result.ch_names == ['MEG0111', 'MEG0112']
    assert passed_to_prepare[0][0] is raw_er
    reference_raw = passed_to_prepare[0][1]
    if use_epochs:
        assert isinstance(reference_raw, mne.io.BaseRaw)
        assert reference_raw.ch_names == recording.ch_names
        assert reference_raw.n_times == 1
        np.testing.assert_array_equal(reference_raw.get_data(), 0.0)
    else:
        assert reference_raw is recording
    expected_kwargs = {
        key: value
        for key, value in maxwell_info.items()
        if key not in ('calibration_applied', 'cross_talk_applied')
    }
    assert passed_to_maxwell == [(raw_er, expected_kwargs)]
    assert history[0].info['maxwell_info'] == maxwell_info


@pytest.mark.parametrize('fit_only', [False, True])
def test_empty_room_ica_uses_applied_exclusions_without_modifying_model(fit_only: bool) -> None:
    """Only actually-applied ICA cleaning is replayed, using a copied model."""
    raw_er = _raw().pick(['MEG0111', 'MEG0112'])
    fitted_ica = Mock()
    fitted_ica.exclude = [0]
    ica_copy = Mock()
    ica_copy.exclude = [0]
    fitted_ica.copy.return_value = ica_copy
    history = [
        StepInfo('ICA', {'ica_info': {'ica': fitted_ica, 'fit_only': fit_only, 'applied_exclude': [1, 2]}})
    ]

    result = spatial_filter_utils.preproc_empty_room(
        raw_er=raw_er, data=_raw(), preproc_info=history, picks=None
    )

    assert result is raw_er
    assert fitted_ica.exclude == [0]
    if fit_only:
        fitted_ica.copy.assert_not_called()
        ica_copy.apply.assert_not_called()
    else:
        fitted_ica.copy.assert_called_once_with()
        assert ica_copy.exclude == [1, 2]
        ica_copy.apply.assert_called_once_with(raw_er)


def test_spatial_filter_passes_full_ordered_history_to_covariance_computation(monkeypatch) -> None:
    """The SpatialFilter boundary must pass its recorded history, not a summary."""
    history = AlmKanalInfo(pick_params={'meg': True})
    history.add(_filter_history(l_freq=None, h_freq=40.0))
    history.add(StepInfo('Resample', {'resample_info': {'sfreq': 100.0}}))
    history.add(_filter_history(l_freq=1.0, h_freq=None))
    captured = []
    beamformer = object()
    data_cov = object()

    def fake_comp_spatial_filters(**kwargs):
        captured.append(kwargs)
        return beamformer, {}, None, data_cov

    monkeypatch.setattr(spatial_filter_utils, 'comp_spatial_filters', fake_comp_spatial_filters)
    step = spatial_filter_utils.SpatialFilter(
        fwd=object(), pick_dict={'meg': True}, empty_room=_raw()
    )
    recording = _raw()

    result = step.run(recording, history)

    assert len(captured) == 1
    assert captured[0]['preproc_info'] is history.processing_history
    assert [entry.step for entry in captured[0]['preproc_info']] == ['Filter', 'Resample', 'Filter']
    assert result['spatial_filter_info']['filters'] is beamformer
    assert result['spatial_filter_info']['data_cov'] is data_cov
