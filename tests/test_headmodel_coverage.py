"""Additional cheap coverage tests for headmodel_utils."""

from pathlib import Path
from unittest.mock import Mock
import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from almkanal.almkanal_steps import headmodel_utils as hm
from almkanal.info import AlmKanalInfo, StepInfo


@pytest.fixture
def raw_small():
    info = mne.create_info(['MEG 0111'], 100.0, ch_types='mag')
    return mne.io.RawArray(
        np.zeros((1, 500)),
        info,
        verbose=False,
    )


def make_subject(path: Path) -> Path:
    for folder in ('mri', 'surf', 'bem', 'label'):
        (path / folder).mkdir(parents=True, exist_ok=True)
    return path


def test_forward_model_rejects_invalid_source(tmp_path):
    step = hm.ForwardModel(
        'recording',
        tmp_path,
        source='invalid',
    )

    with pytest.raises(
        ValueError,
        match="source must be 'surface' or 'volume'",
    ):
        step.run(Mock(), AlmKanalInfo())


def test_forward_model_requires_three_layer_bem_for_eeg(
    raw_small,
    tmp_path,
):
    step = hm.ForwardModel(
        'recording',
        tmp_path,
        eeg=True,
        meg=False,
        bem_conductivity=(0.3,),
    )

    with pytest.raises(
        ValueError,
        match='EEG forward models require a three-layer BEM',
    ):
        step.run(raw_small, AlmKanalInfo())


def test_forward_model_requires_saved_transform_when_not_redoing(
    raw_small,
    tmp_path,
    monkeypatch,
):
    fs_dir = tmp_path / 'freesurfer'
    fsaverage = make_subject(fs_dir / 'fsaverage')
    (
        fsaverage
        / 'bem'
        / 'fsaverage-ico-4-src.fif'
    ).touch()

    monkeypatch.setattr(
        mne.datasets,
        'fetch_fsaverage',
        Mock(),
    )

    step = hm.ForwardModel(
        'recording',
        tmp_path,
        use_template_mri=True,
        redo_hdm=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match='Run with redo_hdm=True first',
    ):
        step.run(raw_small, AlmKanalInfo())


@pytest.mark.parametrize(
    ('meg', 'expected_fig'),
    [
        (True, 'figure'),
        (False, None),
    ],
)
def test_forward_model_reuses_saved_transform(
    raw_small,
    tmp_path,
    monkeypatch,
    meg,
    expected_fig,
):
    fs_dir = tmp_path / 'freesurfer'
    fsaverage = make_subject(fs_dir / 'fsaverage')
    (
        fsaverage
        / 'bem'
        / 'fsaverage-ico-4-src.fif'
    ).touch()

    cache_id = 'recording_from_template'
    trans_dir = tmp_path / 'headmodels' / cache_id
    trans_dir.mkdir(parents=True)
    trans_file = trans_dir / f'{cache_id}-trans.fif'
    trans_file.touch()

    monkeypatch.setattr(
        mne.datasets,
        'fetch_fsaverage',
        Mock(),
    )

    read_trans = Mock(return_value='trans')
    plot = Mock(return_value='figure')
    forward = Mock(
        return_value={
            'nsource': 4,
            'src': [
                {'nuse': 2},
                {'nuse': 2},
            ],
        }
    )

    monkeypatch.setattr(mne, 'read_trans', read_trans)
    monkeypatch.setattr(hm, 'plot_head_model', plot)
    monkeypatch.setattr(hm, 'make_fwd', forward)

    step = hm.ForwardModel(
        'recording',
        tmp_path,
        use_template_mri=True,
        redo_hdm=False,
        meg=meg,
    )

    result = step.run(raw_small, AlmKanalInfo())

    read_trans.assert_called_once_with(trans_file)

    if meg:
        plot.assert_called_once_with(
            'trans',
            raw_small.info,
            subject_id=cache_id,
            subjects_dir=fs_dir,
        )
    else:
        plot.assert_not_called()

    assert result['fwd_info']['coreg_fig'] == expected_fig


@pytest.mark.parametrize('fig', [None, 'figure'])
def test_forward_model_reports_optional_coregistration(
    raw_small,
    tmp_path,
    fig,
):
    report = Mock()

    info = AlmKanalInfo()
    info.add(
        StepInfo(
            step='ForwardModel',
            info={
                'fwd_info': {
                    'coreg_fig': fig,
                    'fwd': 'forward',
                }
            },
        )
    )

    step = hm.ForwardModel(
        'recording',
        tmp_path,
    )
    step.reports(raw_small, report, info)

    if fig is None:
        report.add_figure.assert_not_called()
    else:
        report.add_figure.assert_called_once_with(
            fig='figure',
            title='Coregistration',
            image_format='PNG',
            caption='',
        )

    report.add_forward.assert_called_once_with(
        'forward',
        title='ForwardModel',
    )
