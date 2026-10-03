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

        np.testing.assert_allclose(evoked.data, expected, rtol=1e-12, atol=1e-25)
        assert evoked.baseline is None
        assert evoked.nave == 2

    # Reporting must preserve MEG, the WAV feature channel, and epoch metadata.
    np.testing.assert_array_equal(epochs.get_data(copy=True), original.get_data(copy=True))
    np.testing.assert_array_equal(epochs.times, original.times)
    np.testing.assert_array_equal(epochs.events, original.events)
    pd.testing.assert_frame_equal(epochs.metadata, original.metadata)
    assert epochs.event_id == original.event_id
    assert epochs.baseline is None



from unittest.mock import Mock

import mne
import numpy as np
import pytest

from almkanal.almkanal_steps.filter_utils import Filter, Resample


@pytest.fixture
def raw_small():
    info = mne.create_info(
        ['Cz'],
        sfreq=100.0,
        ch_types='eeg',
    )

    values = (
        np.random.default_rng(42).normal(size=(1, 2000))
        * 1e-6
    )

    return mne.io.RawArray(
        values,
        info,
        verbose=False,
    )


@pytest.mark.parametrize(
    'settings, expected_widths',
    [
        (
            {
                'highpass': 1.234,
                'lowpass': 30.123,
            },
            (1.234, 7.53075),
        ),
        (
            {
                'highpass': 1.0,
                'lowpass': 30.0,
                'l_trans_bandwidth': 0.8765,
                'h_trans_bandwidth': 4.5678,
            },
            (0.8765, 4.5678),
        ),
        (
            {
                'highpass': None,
                'lowpass': 30.123,
            },
            (None, 7.53075),
        ),
    ],
    ids=[
        'auto-bandpass',
        'explicit-bandpass',
        'lowpass-only',
    ],
)
def test_filter_preserves_bandwidth_precision(
    raw_small,
    monkeypatch,
    settings,
    expected_widths,
):
    reference = raw_small.copy()

    # Record the call without replacing MNE's filtering.
    spy = Mock(wraps=raw_small.filter)
    monkeypatch.setattr(raw_small, 'filter', spy)

    result = Filter(**settings).run(
        raw_small,
        info={},
    )

    recorded = result['filter_info']

    spy.assert_called_once()
    passed = spy.call_args.kwargs

    for key, expected in zip(
        ('l_trans_bandwidth', 'h_trans_bandwidth'),
        expected_widths,
    ):
        if expected is None:
            assert recorded[key] is None
        else:
            assert recorded[key] == pytest.approx(expected)

            # Metadata must retain the exact value passed to MNE.
            assert recorded[key] == passed[key]

    # The returned signal must match a direct MNE call.
    reference.filter(**passed)

    np.testing.assert_array_equal(
        result['data'].get_data(),
        reference.get_data(),
    )


@pytest.mark.parametrize(
    'step_class',
    [Filter, Resample],
)
@pytest.mark.parametrize(
    'tmin, baseline',
    [
        (-0.2, None),
        (-0.2, (-0.2, -0.1)),
        (0.0, None),
    ],
    ids=[
        'no-baseline',
        'custom-baseline',
        'zero-start',
    ],
)
def test_epoch_reports_preserve_the_supplied_data(
    step_class,
    tmin,
    baseline,
):
    info = mne.create_info(
        ['Cz'],
        sfreq=100.0,
        ch_types='eeg',
    )

    # A nonzero, changing signal makes an extra baseline
    # correction detectable.
    ramp = np.linspace(2e-6, 5e-6, 61)
    values = np.stack(
        [ramp, ramp + 1e-6]
    )[:, None, :]

    epochs = mne.EpochsArray(
        values,
        info,
        events=np.array([
            [0, 0, 1],
            [100, 0, 2],
        ]),
        event_id={'a': 1, 'b': 2},
        tmin=tmin,
        baseline=baseline,
        verbose=False,
    )

    original = epochs.get_data(copy=True)
    expected = epochs.copy().average(by_event_type=True)

    report = Mock(spec=mne.Report)
    step = (
        Filter()
        if step_class is Filter
        else Resample(sfreq=50)
    )

    step.reports(
        epochs,
        report,
        info={},
    )

    report.add_evokeds.assert_called_once()
    actual = report.add_evokeds.call_args.args[0]

    assert len(actual) == len(expected)

    for observed, wanted in zip(actual, expected):
        np.testing.assert_allclose(
            observed.data,
            wanted.data,
            rtol=0,
            atol=1e-18,
        )

        assert observed.baseline == wanted.baseline
        assert observed.comment == wanted.comment

    # Reporting must also leave the input epochs unchanged.
    np.testing.assert_array_equal(
        epochs.get_data(),
        original,
    )


@pytest.mark.parametrize(
    'step_class, title',
    [
        (Filter, 'Raw (filtered)'),
        (Resample, 'RawResample'),
    ],
)
def test_raw_reporting_is_unchanged(
    raw_small,
    step_class,
    title,
):
    report = Mock(spec=mne.Report)
    step = (
        Filter()
        if step_class is Filter
        else Resample(sfreq=50)
    )

    step.reports(
        raw_small,
        report,
        info={},
    )

    report.add_raw.assert_called_once_with(
        raw_small,
        butterfly=False,
        psd=True,
        title=title,
    )

    report.add_evokeds.assert_not_called()