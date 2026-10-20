"""Coverage tests for headmodel_utils.plot_head_model.

These tests exercise AlmKanal's plotting and scalp-surface selection logic
without running real FreeSurfer/BEM geometry or MNE coregistration.
"""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from almkanal.almkanal_steps import headmodel_utils as hm


@pytest.fixture
def plot_info():
    """Minimal MEG info with deterministic sensor and head-shape locations."""
    info = mne.create_info(
        ['MEG 0111'],
        sfreq=100.0,
        ch_types='mag',
    )
    info['chs'][0]['loc'][:3] = np.array([0.02, 0.00, 0.04])

    with info._unlock():
        info['dig'] = [
            {
                'kind': 4,
                'r': np.array([0.00, 0.00, 0.08]),
                'ident': 1,
                'coord_frame': 4,
            },
            {
                'kind': 4,
                'r': np.array([0.01, 0.00, 0.08]),
                'ident': 2,
                'coord_frame': 4,
            },
        ]

    return info


@pytest.fixture
def mock_plot_transforms(monkeypatch):
    """Keep plot_head_model entirely in synthetic identity coordinates."""
    get_trans = Mock(return_value=(object(), None))
    get_to_frame = Mock(
        return_value={
            'meg': np.eye(4),
            'head': np.eye(4),
            'mri': np.eye(4),
        }
    )

    monkeypatch.setattr(
        hm.mne.transforms,
        '_get_trans',
        get_trans,
    )
    monkeypatch.setattr(
        hm.mne.transforms,
        '_get_transforms_to_coord_frame',
        get_to_frame,
    )

    return get_trans, get_to_frame


def _bem_surfaces():
    """Two simple surfaces so both first/else plotting branches execute."""
    return [
        {
            'rr': np.array(
                [
                    [-0.02, -0.02, 0.00],
                    [0.02, -0.02, 0.00],
                    [0.00, 0.02, 0.04],
                ]
            )
        },
        {
            'rr': np.array(
                [
                    [-0.03, -0.03, -0.01],
                    [0.03, -0.03, -0.01],
                    [0.00, 0.03, 0.05],
                ]
            )
        },
    ]


def test_plot_head_model_uses_bem_head_file(
    plot_info,
    mock_plot_transforms,
    tmp_path,
    monkeypatch,
):
    subject_id = 'recording'
    bem_path = (
        tmp_path
        / subject_id
        / 'bem'
        / f'{subject_id}-head.fif'
    )
    bem_path.parent.mkdir(parents=True)
    bem_path.touch()

    read_bem = Mock(return_value=_bem_surfaces())
    read_surface = Mock()
    get_head_surf = Mock()

    monkeypatch.setattr(mne, 'read_bem_surfaces', read_bem)
    monkeypatch.setattr(mne, 'read_surface', read_surface)
    monkeypatch.setattr(mne, 'get_head_surf', get_head_surf)

    fig = hm.plot_head_model(
        object(),
        plot_info,
        subject_id=subject_id,
        subjects_dir=tmp_path,
    )

    try:
        assert len(fig.axes) == 4
        assert [ax.get_title() for ax in fig.axes] == [
            'Axial View',
            'Coronal View',
            'Sagittal View',
            '3D View',
        ]
        assert fig._suptitle.get_text() == (
            f'MEG - DIG Coregistration of {subject_id}'
        )

        read_bem.assert_called_once_with(bem_path)
        read_surface.assert_not_called()
        get_head_surf.assert_not_called()
    finally:
        plt.close(fig)


def test_plot_head_model_falls_back_to_freesurfer_scalp_surface(
    plot_info,
    mock_plot_transforms,
    tmp_path,
    monkeypatch,
):
    subject_id = 'recording'
    scalp_path = (
        tmp_path
        / subject_id
        / 'bem'
        / 'outer_skin.surf'
    )
    scalp_path.parent.mkdir(parents=True)
    scalp_path.touch()

    # FreeSurfer surface coordinates are in millimetres in this branch.
    rr_mm = np.array(
        [
            [-20.0, -20.0, 0.0],
            [20.0, -20.0, 0.0],
            [0.0, 20.0, 40.0],
        ]
    )

    read_bem = Mock()
    read_surface = Mock(
        return_value=(
            rr_mm,
            np.array([[0, 1, 2]]),
        )
    )
    get_head_surf = Mock()

    monkeypatch.setattr(mne, 'read_bem_surfaces', read_bem)
    monkeypatch.setattr(mne, 'read_surface', read_surface)
    monkeypatch.setattr(mne, 'get_head_surf', get_head_surf)

    fig = hm.plot_head_model(
        object(),
        plot_info,
        subject_id=subject_id,
        subjects_dir=tmp_path,
    )

    try:
        read_bem.assert_not_called()
        read_surface.assert_called_once_with(scalp_path)
        get_head_surf.assert_not_called()

        assert len(fig.axes) == 4
    finally:
        plt.close(fig)


def test_plot_head_model_falls_back_to_mne_head_surface(
    plot_info,
    mock_plot_transforms,
    tmp_path,
    monkeypatch,
):
    subject_id = 'recording'

    read_bem = Mock()
    read_surface = Mock()
    get_head_surf = Mock(return_value=_bem_surfaces()[0])

    monkeypatch.setattr(mne, 'read_bem_surfaces', read_bem)
    monkeypatch.setattr(mne, 'read_surface', read_surface)
    monkeypatch.setattr(mne, 'get_head_surf', get_head_surf)

    fig = hm.plot_head_model(
        object(),
        plot_info,
        subject_id=subject_id,
        subjects_dir=tmp_path,
    )

    try:
        read_bem.assert_not_called()
        read_surface.assert_not_called()
        get_head_surf.assert_called_once_with(
            subject_id,
            source=('head-sparse', 'head-dense', 'bem'),
            subjects_dir=tmp_path,
        )

        assert len(fig.axes) == 4
    finally:
        plt.close(fig)
