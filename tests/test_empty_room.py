import os
from datetime import UTC, datetime

import mne
import pytest

import numpy as np

from almkanal.almkanal_steps.spatial_filter_utils import (
    _get_empty_room_picks,
    get_nearest_empty_room,
    preproc_empty_room,
)
from almkanal.info import StepInfo


@pytest.fixture
def setup_empty_room_dir(tmp_path):
    """
    Creates three date-labeled subfolders (YYMMDD). Two contain only 'supine',
    and one contains only 'empty_room_68.fif' (the target).
    """
    d1 = tmp_path / '240101'
    d1.mkdir()
    (d1 / 'empty_room_supine.fif').touch()

    d2 = tmp_path / '240110'
    d2.mkdir()
    (d2 / 'empty_room_68.fif').touch()

    d3 = tmp_path / '240115'
    d3.mkdir()
    (d3 / 'empty_room_supine.fif').touch()

    return tmp_path


@pytest.fixture
def mock_info():
    """
    Creates a real MNE Info object with a UTC-aware datetime for meas_date.
    """
    info = mne.create_info(
        ch_names=['MEG 001'],
        sfreq=1000,
        ch_types=['mag'],
    )
    info.set_meas_date(
        datetime(
            2024,
            1,
            11,
            tzinfo=UTC,
        )
    )
    return info


def test_get_nearest_empty_room(
    setup_empty_room_dir,
    mock_info,
):
    print(
        'Test directories:',
        os.listdir(setup_empty_room_dir),
    )

    for date_dir in os.listdir(setup_empty_room_dir):
        print(f'  {date_dir} -> ' f'{os.listdir(setup_empty_room_dir / date_dir)}')

    result_path = get_nearest_empty_room(
        mock_info,
        str(setup_empty_room_dir),
    )

    expected = setup_empty_room_dir / '240110' / 'empty_room_68.fif'

    assert result_path == expected


def _create_test_raw(ch_names, ch_types):
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=1000,
        ch_types=ch_types,
    )
    return mne.io.RawArray(
        np.zeros((len(ch_names), 100)),
        info,
        verbose='ERROR',
    )


def test_get_empty_room_picks():
    """Test that MEG and EEG channels are correctly selected."""
    data = _create_test_raw(
        ['MEG 001', 'MEG 002', 'EEG 001'],
        ['mag', 'grad', 'eeg'],
    )
    raw_er = _create_test_raw(
        ['MEG 001', 'MEG 002', 'EEG 001'],
        ['mag', 'grad', 'eeg'],
    )

    picks = ['MEG 001', 'MEG 002', 'EEG 001']

    result = _get_empty_room_picks(data, raw_er, picks)

    assert result == picks


def test_get_empty_room_picks_missing_channels():
    """Test that unavailable channels are excluded with a warning."""
    data = _create_test_raw(
        ['MEG 001', 'MEG 002', 'EEG 001'],
        ['mag', 'grad', 'eeg'],
    )
    raw_er = _create_test_raw(
        ['MEG 001', 'EEG 001'],
        ['mag', 'eeg'],
    )

    picks = ['MEG 001', 'MEG 002', 'EEG 001']

    with pytest.warns(UserWarning, match='not available'):
        result = _get_empty_room_picks(data, raw_er, picks)

    assert result == ['MEG 001', 'EEG 001']


def test_get_empty_room_picks_no_meg_or_eeg():
    """Test that an error is raised if no MEG/EEG channels are selected."""
    data = _create_test_raw(
        ['STI 001'],
        ['stim'],
    )
    raw_er = data.copy()

    with pytest.raises(ValueError, match='No MEG or EEG channels'):
        _get_empty_room_picks(data, raw_er, ['STI 001'])


def test_get_empty_room_picks_no_matching_channels():
    """Test that an error is raised if no selected channels exist in empty room."""
    data = _create_test_raw(
        ['MEG 001'],
        ['mag'],
    )
    raw_er = _create_test_raw(
        ['MEG 002'],
        ['mag'],
    )

    with pytest.warns(UserWarning, match='not available'):
        with pytest.raises(ValueError, match='No matching MEG/EEG channels'):
            _get_empty_room_picks(data, raw_er, ['MEG 001'])


def test_empty_room_filtering(
    gen_mne_data_raw,
    monkeypatch,
):
    raw, _ = gen_mne_data_raw

    data = raw.copy()
    raw_er = raw.copy()

    called = {}

    def fake_filter(
        l_freq,
        h_freq,
        **kwargs,
    ):
        called['l_freq'] = l_freq
        called['h_freq'] = h_freq
        return raw_er

    monkeypatch.setattr(
        raw_er,
        'filter',
        fake_filter,
    )

    preproc_info = [
        StepInfo(
            step='Filter',
            info={
                'filter_info': {
                    'l_freq': 1.0,
                    'h_freq': 40.0,
                }
            },
        )
    ]

    preproc_empty_room(
        raw_er=raw_er,
        data=data,
        preproc_info=preproc_info,
        picks=None,
    )

    assert called == {
        'l_freq': 1.0,
        'h_freq': 40.0,
    }


def test_empty_room_resampling(
    gen_mne_data_raw,
    monkeypatch,
):
    raw, _ = gen_mne_data_raw

    data = raw.copy()
    raw_er = raw.copy()

    called = {}

    def fake_resample(
        sfreq,
        **kwargs,
    ):
        called['sfreq'] = sfreq
        return raw_er

    monkeypatch.setattr(
        raw_er,
        'resample',
        fake_resample,
    )

    preproc_info = [
        StepInfo(
            step='Resample',
            info={
                'resample_info': {
                    'sfreq': data.info['sfreq'],
                }
            },
        )
    ]

    preproc_empty_room(
        raw_er=raw_er,
        data=data,
        preproc_info=preproc_info,
        picks=None,
    )

    assert called['sfreq'] == data.info['sfreq']
