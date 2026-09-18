from unittest.mock import Mock

import mne
import numpy as np
import pandas as pd
import pytest

from almkanal import Filter, Resample


@pytest.mark.parametrize('step', [Filter(), Resample(sfreq=100)], ids=['filter', 'resample'])
@pytest.mark.parametrize(
    ('tmin', 'baseline_samples'),
    [(0.0, 0), (0.1, 0), (-0.2, 21), (-0.4, 41), (-0.5, 0)],
    ids=['zero-start', 'positive-start', 'pre-zero-baseline', 'ends-at-zero', 'ends-before-zero'],
)
def test_epoch_reports_baseline_and_input_preservation(step, tmin, baseline_samples):
    samples = np.random.default_rng(42).normal(size=(4, 2, 41)) + 10
    samples[:, 0, :] *= 1e-12
    events = np.column_stack((np.arange(4) * 100, np.zeros(4, dtype=int), [1, 2, 1, 2]))
    metadata = pd.DataFrame({'condition': ['SingleSpeaker', 'SpeechInNoise'] * 2})
    epochs = mne.EpochsArray(
        samples,
        mne.create_info(['MEG001', 'env_rms'], sfreq=100, ch_types=['mag', 'misc']),
        events=events,
        event_id={'trial_000': 1, 'trial_001': 2},
        tmin=tmin,
        baseline=None,
        metadata=metadata,
        verbose=False,
    )
    original = epochs.copy()
    # Exercise MNE's baseline correction and averaging; only rendering is mocked.
    report = Mock(spec=mne.Report)

    step.reports(epochs, report, {})

    report.add_evokeds.assert_called_once()
    assert report.add_evokeds.call_args.kwargs == {'n_time_points': 5}
    evokeds = report.add_evokeds.call_args.args[0]
    assert len(evokeds) == 2
    for evoked, code in zip(evokeds, (1, 2), strict=True):
        expected = original.get_data(copy=True)[events[:, 2] == code, :1, :].mean(axis=0)
        if baseline_samples:
            expected -= expected[:, :baseline_samples].mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(evoked.data, expected, rtol=1e-12, atol=1e-25)
        assert evoked.baseline == ((tmin, 0.0) if baseline_samples else None)
        assert evoked.nave == 2

    # Reporting must preserve MEG, the WAV feature channel, and epoch metadata.
    np.testing.assert_array_equal(epochs.get_data(copy=True), original.get_data(copy=True))
    np.testing.assert_array_equal(epochs.times, original.times)
    np.testing.assert_array_equal(epochs.events, original.events)
    pd.testing.assert_frame_equal(epochs.metadata, original.metadata)
    assert epochs.event_id == original.event_id
    assert epochs.baseline is None
