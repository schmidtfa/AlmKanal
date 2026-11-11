import warnings

import mne
import numpy as np
import pytest

from almkanal import Resample
from almkanal.info import AlmKanalInfo, StepInfo


@pytest.fixture
def stimulus_epochs():
    values = np.stack([np.sin(np.arange(100)), np.tile(np.repeat([0.0, 1.0], 25), 2)])
    epochs = mne.EpochsArray(
        values[None, :, :],
        mne.create_info(['EEG', 'binary_feature'], 100, ['eeg', 'misc']),
        verbose=False,
    )
    # Permit downsampling under the existing explicit anti-aliasing policy.
    with epochs.info._unlock():
        epochs.info['lowpass'] = 20.0
    return epochs


@pytest.mark.parametrize('method', ['fft', 'polyphase'])
@pytest.mark.parametrize('sfreq', [50, 200], ids=['downsample', 'upsample'])
def test_resampling_after_stimulus_epochs_warns(stimulus_epochs, method, sfreq):
    info = AlmKanalInfo()
    info.add(StepInfo('StimulusEpochs', {'TRF_info': {'feature_channels': ['binary_feature']}}))
    info.add(StepInfo('Filter', {'filter_info': {}}))  # Not necessarily the immediately preceding step.

    with pytest.warns(
        UserWarning,
        match=r'Resampling after StimulusEpochs.*binary \(0/1\).*StimulusEpochs\(resample_sfreq=',
    ) as caught:
        result = Resample(sfreq=sfreq, method=method).run(stimulus_epochs, info)

    assert len(caught) == 1
    assert result['data'] is stimulus_epochs
    assert result['data'].info['sfreq'] == sfreq
    assert result['resample_info']['original_sfreq'] == 100


@pytest.mark.parametrize('method', ['fft', 'polyphase'])
@pytest.mark.parametrize('after_stimulus', [False, True], ids=['no-stimulus-step', 'unchanged-rate'])
def test_resampling_without_feature_risk_does_not_warn(stimulus_epochs, method, after_stimulus):
    info = AlmKanalInfo()
    if after_stimulus:
        info.add(StepInfo('StimulusEpochs', {'TRF_info': {'feature_channels': ['binary_feature']}}))
    sfreq = 100 if after_stimulus else 200

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = Resample(sfreq=sfreq, method=method).run(stimulus_epochs, info)

    assert not caught
    assert result['data'].info['sfreq'] == sfreq
