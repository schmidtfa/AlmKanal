import os
from datetime import datetime, timezone

import mne
import pytest

from almkanal.almkanal_steps.spatial_filter_utils import (
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
            tzinfo=timezone.utc,
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
        print(
            f'  {date_dir} -> '
            f'{os.listdir(setup_empty_room_dir / date_dir)}'
        )

    result_path = get_nearest_empty_room(
        mock_info,
        str(setup_empty_room_dir),
    )

    expected = (
        setup_empty_room_dir
        / '240110'
        / 'empty_room_68.fif'
    )

    assert result_path == expected


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