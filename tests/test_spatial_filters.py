from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from almkanal.almkanal_steps import (
    spatial_filter_utils as sfu,
)

def test_nearest_empty_room_skips_invalid_and_supine(
    tmp_path,
):
    # Invalid directory name should be ignored.
    (
        tmp_path / 'notes'
    ).mkdir()

    nearest = (
        tmp_path / '260920'
    )
    nearest.mkdir()

    (
        nearest
        / 'empty_supine.fif'
    ).touch()

    fallback = (
        tmp_path / '260922'
    )
    fallback.mkdir()

    expected = (
        fallback
        / 'empty_room.fif'
    )
    expected.touch()

    info = mne.create_info(
        ['MEG 0111'],
        100,
        'mag',
    )

    info.set_meas_date(
        datetime(
            2026,
            9,
            20,
            tzinfo=timezone.utc,
        )
    )

    result = sfu.get_nearest_empty_room(
        info,
        str(tmp_path),
    )

    assert result == expected


def test_nearest_empty_room_raises_without_dates(
    tmp_path,
):
    (
        tmp_path
        / 'not_a_date'
    ).mkdir()

    info = mne.create_info(
        ['MEG 0111'],
        100,
        'mag',
    )

    info.set_meas_date(
        datetime(
            2026,
            9,
            20,
            tzinfo=timezone.utc,
        )
    )

    with pytest.raises(
        ValueError,
        match='No valid empty room',
    ):
        sfu.get_nearest_empty_room(
            info,
            str(tmp_path),
        )

def test_nearest_empty_room_skips_empty_directory(
    tmp_path,
):
    empty = tmp_path / '260920'
    empty.mkdir()

    usable = tmp_path / '260921'
    usable.mkdir()

    expected = usable / 'empty.fif'
    expected.touch()

    info = mne.create_info(
        ['MEG 0111'],
        100,
        'mag',
    )

    info.set_meas_date(
        datetime(
            2026,
            9,
            20,
            tzinfo=timezone.utc,
        )
    )

    assert (
        sfu.get_nearest_empty_room(
            info,
            str(tmp_path),
        )
        == expected
    )


def _raw(
    sfreq=100,
    ch_types=('mag',),
    ):
    names = [
        f'CH{ix}'
        for ix in range(
            len(ch_types)
        )
    ]

    info = mne.create_info(
        names,
        sfreq,
        list(ch_types),
    )

    return mne.io.RawArray(
        np.zeros(
            (
                len(ch_types),
                500,
            )
        ),
        info,
        verbose=False,
    )


def _set_filter_info(
    raw,
    highpass,
    lowpass,
):
    with raw.info._unlock():
        raw.info[
            'highpass'
        ] = highpass

        raw.info[
            'lowpass'
        ] = lowpass


@pytest.mark.parametrize(
    'empty_settings, expected',
    [
        (
            (0.1, 40.0),
            {
                'l_freq': 1.0,
                'h_freq': None,
            },
        ),
        (
            (1.0, 45.0),
            {
                'l_freq': None,
                'h_freq': 40.0,
            },
        ),
    ],
)
def test_preproc_empty_room_single_filter_edge(
    monkeypatch,
    empty_settings,
    expected,
):
    data = _raw()
    raw_er = _raw()

    _set_filter_info(
        data,
        1.0,
        40.0,
    )

    _set_filter_info(
        raw_er,
        *empty_settings,
    )

    filter_mock = Mock(
        return_value=raw_er
    )

    monkeypatch.setattr(
        raw_er,
        'filter',
        filter_mock,
    )

    result = sfu.preproc_empty_room(
        raw_er=raw_er,
        data=data,
        preproc_info={},
        picks=raw_er.ch_names,
    )

    filter_mock.assert_called_once_with(
        **expected
    )

    assert result is raw_er


@pytest.mark.parametrize(
    'use_epochs',
    [False, True],
)
def test_preproc_empty_room_maxwell_and_ica(
    monkeypatch,
    use_epochs,
):
    raw = _raw()
    raw_er = _raw()

    if use_epochs:
        data = mne.EpochsArray(
            np.zeros((2, 1, 20)),
            raw.info.copy(),
            verbose=False,
        )
    else:
        data = raw

    prepare_mock = Mock(
        name='prepare_emptyroom',
        return_value=raw_er,
    )

    maxwell_mock = Mock(
        name='run_maxwell',
        return_value=raw_er,
    )

    ica = Mock(name='ica')
    ica_copy = Mock(name='ica_copy')
    ica_copy.exclude = []

    ica.copy.return_value = ica_copy

    monkeypatch.setattr(
        mne.preprocessing,
        'maxwell_filter_prepare_emptyroom',
        prepare_mock,
    )

    monkeypatch.setattr(
        sfu,
        'run_maxwell',
        maxwell_mock,
    )

    preproc_info = {
        'Maxwell': {
            'maxwell_info': {
                'coord_frame': 'head',
            },
        },
        'ICA': {
            'ica_info': {
                'ica': ica,
                'fit_only': False,
                'applied_exclude': [1, 3],
            },
        },
    }

    result = sfu.preproc_empty_room(
        raw_er=raw_er,
        data=data,
        preproc_info=preproc_info,
        picks=raw_er.ch_names,
    )

    assert result is raw_er

    prepare_mock.assert_called_once()
    maxwell_mock.assert_called_once()

    ica.copy.assert_called_once()
    assert ica_copy.exclude == [1, 3]

    ica_copy.apply.assert_called_once_with(
        raw_er
    )

@pytest.mark.parametrize(
    'nearest',
    [False, True],
)
def test_process_empty_room_from_path(
    monkeypatch,
    nearest,
):
    data = _raw()
    raw_er = _raw()

    read_raw = Mock(
        return_value=raw_er
    )

    monkeypatch.setattr(
        mne.io,
        'read_raw',
        read_raw,
    )

    if nearest:
        monkeypatch.setattr(
            sfu,
            'get_nearest_empty_room',
            lambda *args, **kwargs:
                'nearest.fif',
        )

    monkeypatch.setattr(
        sfu,
        'preproc_empty_room',
        lambda **kwargs: raw_er,
    )

    fake_cov = object()

    monkeypatch.setattr(
        mne,
        'compute_raw_covariance',
        Mock(
            return_value=fake_cov
        ),
    )

    monkeypatch.setattr(
        mne,
        'compute_rank',
        Mock(
            return_value={
                'meg': 1,
            }
        ),
    )

    rank, cov = (
        sfu.process_empty_room(
            data=data,
            info=data.info,
            picks=None,
            preproc_info={},
            empty_room='empty.fif',
            get_nearest=nearest,
        )
    )

    assert rank == {
        'meg': 1,
    }

    assert cov is fake_cov


def test_comp_spatial_filters_uses_supplied_noise_cov(
    monkeypatch,
):
    data = _raw(
        ch_types=(
            'mag',
            'grad',
        )
    )

    data_cov = object()
    noise_cov = object()

    compute_rank = Mock(
        return_value={
            'meg': 2,
        }
    )

    make_lcmv = Mock(
        return_value='filters'
    )

    monkeypatch.setattr(
        mne,
        'compute_rank',
        compute_rank,
    )

    monkeypatch.setattr(
        mne.beamformer,
        'make_lcmv',
        make_lcmv,
    )

    filters, settings, returned_noise, returned_data = (
        sfu.comp_spatial_filters(
            data=data,
            fwd=object(),
            pick_dict=None,
            preproc_info={},
            data_cov=data_cov,
            noise_cov=noise_cov,
        )
    )

    assert filters == 'filters'
    assert returned_noise is noise_cov
    assert returned_data is data_cov

    compute_rank.assert_called_with(
        noise_cov,
        info=data.info,
    )

    assert (
        settings['noise_cov']
        is noise_cov
    )


def test_comp_spatial_filters_uses_empty_room(
    monkeypatch,
):
    data = _raw(
        ch_types=(
            'mag',
            'grad',
        )
    )

    data_cov = object()
    noise_cov = object()

    process = Mock(
        return_value=(
            {'meg': 2},
            noise_cov,
        )
    )

    monkeypatch.setattr(
        sfu,
        'process_empty_room',
        process,
    )

    monkeypatch.setattr(
        mne.beamformer,
        'make_lcmv',
        Mock(
            return_value='filters'
        ),
    )

    result = (
        sfu.comp_spatial_filters(
            data=data,
            fwd=object(),
            pick_dict=None,
            preproc_info={},
            data_cov=data_cov,
            empty_room='empty.fif',
        )
    )

    process.assert_called_once()

    assert result[2] is noise_cov


def test_comp_spatial_filters_adhoc_covariance(
    monkeypatch,
):
    data = _raw(
        ch_types=(
            'mag',
            'grad',
        )
    )

    data_cov = object()
    noise_cov = object()

    monkeypatch.setattr(
        mne,
        'make_ad_hoc_cov',
        Mock(
            return_value=noise_cov
        ),
    )

    monkeypatch.setattr(
        mne,
        'compute_rank',
        Mock(
            return_value={
                'meg': 2,
            }
        ),
    )

    monkeypatch.setattr(
        mne.beamformer,
        'make_lcmv',
        Mock(
            return_value='filters'
        ),
    )

    with pytest.warns(
        UserWarning,
        match='ad-hoc',
    ):
        result = (
            sfu.comp_spatial_filters(
                data=data,
                fwd=object(),
                pick_dict=None,
                preproc_info={},
                data_cov=data_cov,
                noise_cov=None,
                empty_room=None,
            )
        )

    assert result[2] is noise_cov


def test_spatial_filter_requires_picks():
    data = _raw()

    step = sfu.SpatialFilter(
        fwd=object(),
    )

    with pytest.raises(
        ValueError,
        match='pick_dict',
    ):
        step.run(
            data,
            info={
                'Picks': None,
            },
        )


def test_spatial_filter_report_with_noise_cov():
    data = _raw()

    report = Mock(
        spec=mne.Report
    )

    data_cov = object()

    noise_cov = Mock()
    noise_cov._as_square.return_value = (
        'noise-square'
    )

    info = {
        'SpatialFilter': {
            'spatial_filter_info': {
                'data_cov': data_cov,
                'noise_cov': noise_cov,
            }
        }
    }

    sfu.SpatialFilter().reports(
        data,
        report,
        info,
    )

    assert (
        report.add_covariance.call_count
        == 2
    )

    noise_cov._as_square.assert_called_once()