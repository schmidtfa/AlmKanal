from unittest.mock import Mock

import mne
import numpy as np
import pytest

from almkanal.almkanal import (
    AlmKanal,
    AlmKanalStep,
)
import almkanal.almkanal as core


class DummyStep(AlmKanalStep):

    def run(
        self,
        data,
        info,
    ):
        return {
            'data': data,
            'dummy_info': {
                'ran': True,
            },
        }

    def reports(
        self,
        data,
        report,
        info,
    ):
        pass


class TargetStep(DummyStep):
    pass


class BadStep(AlmKanalStep):

    def run(
        self,
        data,
        info,
    ):
        return {
            'not_data': True,
        }

    def reports(
        self,
        data,
        report,
        info,
    ):
        pass


@pytest.fixture
def raw_small():
    info = mne.create_info(
        [
            'MEG 0111',
            'STI 014',
        ],
        sfreq=100,
        ch_types=[
            'mag',
            'stim',
        ],
    )

    return mne.io.RawArray(
        np.zeros((2, 100)),
        info,
        verbose=False,
    )


@pytest.fixture
def epochs_small():
    info = mne.create_info(
        ['Cz'],
        sfreq=100,
        ch_types='eeg',
    )

    return mne.EpochsArray(
        np.zeros((2, 1, 20)),
        info,
        verbose=False,
    )


@pytest.fixture
def mock_report(
    monkeypatch,
):
    report = Mock(
        spec=mne.Report
    )

    monkeypatch.setattr(
        core.mne,
        'Report',
        Mock(
            return_value=report
        ),
    )

    return report

def test_dependency_must_be_before():
    step = DummyStep()
    step.must_be_before = ('TargetStep',)

    with pytest.raises(ValueError, match='should follow'):
        AlmKanal(steps=[TargetStep(), step])


def test_dependency_must_be_after():
    step = DummyStep()
    step.must_be_after = ('TargetStep',)

    with pytest.raises(ValueError, match='should precede'):
        AlmKanal(steps=[step, TargetStep()])


def test_base_step_methods_raise():
    step = AlmKanalStep()

    with pytest.raises(NotImplementedError):
        step.run(None, {})

    with pytest.raises(NotImplementedError):
        step.reports(None, None, {})


def test_invalid_pipeline_input(mock_report):
    with pytest.raises(ValueError, match='Input data'):
        AlmKanal(steps=[]).run(np.zeros(10))


def test_bad_step_return_raises(raw_small, mock_report):
    with pytest.raises(ValueError, match="'data' key"):
        AlmKanal(steps=[BadStep()]).run(raw_small)


def test_pipeline_callable(raw_small, mock_report):
    processed, report = AlmKanal(steps=[])(raw_small)

    assert processed is raw_small
    assert report is mock_report


def test_raw_list_and_picking(raw_small, mock_report):
    data = [raw_small.copy(), raw_small.copy()]

    pipeline = AlmKanal(
        steps=[],
        pick_params={'meg': True, 'stim': False},
    )

    processed, report = pipeline.run(data)

    assert report is mock_report
    assert mock_report.add_raw.call_count == 2
    mock_report.add_epochs.assert_not_called()
    assert len(processed) == 2

    for block in processed:
        assert block.ch_names == ['MEG 0111']


def test_epochs_list_is_reported(epochs_small, mock_report):
    data = [epochs_small.copy(), epochs_small.copy()]

    processed, report = AlmKanal(steps=[]).run(data)

    assert processed is data
    assert report is mock_report
    assert mock_report.add_epochs.call_count == 2
    mock_report.add_raw.assert_not_called()