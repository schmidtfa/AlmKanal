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