from types import SimpleNamespace
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from almkanal import (
    AlmKanal,
    EEGRANSAC,
    Filter,
    ForwardModel,
    ReReference,
    Resample,
    SourceReconstruction,
    SpatialFilter,
)
from .settings import SOURCE_SURF, SOURCE_VOL


class _DummyReport:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture(autouse=True)
def _disable_reports(monkeypatch):
    """Raw smoke tests exercise processing, not MNE report rendering."""
    monkeypatch.setattr('almkanal.almkanal.mne.Report', _DummyReport)

    for step_cls in (
        EEGRANSAC,
        Filter,
        ReReference,
        Resample,
        ForwardModel,
        SpatialFilter,
        SourceReconstruction,
    ):
        monkeypatch.setattr(step_cls, 'reports', lambda *args, **kwargs: None)


def _mock_forward_step(monkeypatch, subjects_dir):
    """Provide ForwardModel metadata without running FreeSurfer/MNE anatomy work."""
    def fake_run(self, data, info):
        return {
            'data': data,
            'fwd_info': {
                'fwd': {'src': SimpleNamespace(kind=self.source)},
                'subject_id_freesurfer': self.subject_id,
                'subjects_dir': str(subjects_dir),
            },
        }

    monkeypatch.setattr(ForwardModel, 'run', fake_run)


def _mock_source_application(monkeypatch):
    """Replace MNE beamformer application/parcellation, not AlmKanal glue."""
    monkeypatch.setattr(
        mne.beamformer,
        'apply_lcmv_raw',
        lambda data, filters: object(),
    )


def test_ransac(gen_mne_data_raw_eeg):
    raw, _ = gen_mne_data_raw_eeg
    raw = raw.copy().crop(tmax=10)

    ak = AlmKanal(
        steps=[
            EEGRANSAC(),
            Filter(),
            ReReference(),
            Resample(100),
        ]
    )
    ak.run(raw)


@pytest.mark.parametrize('source', [source for source, _ in SOURCE_SURF], scope='session')
def test_surface_source_pipeline(gen_mne_data_raw, source, monkeypatch, tmp_path):
    raw, _ = gen_mne_data_raw
    raw = raw.copy().crop(tmax=1)

    _mock_forward_step(monkeypatch, tmp_path)
    _mock_source_application(monkeypatch)

    fake_filters = object()
    monkeypatch.setattr(
        'almkanal.almkanal_steps.spatial_filter_utils.comp_spatial_filters',
        lambda **kwargs: (fake_filters, {}, None, object()),
    )

    pick_dict = {
        'meg': 'mag',
        'eog': False,
        'ecg': False,
        'eeg': False,
        'stim': False,
    }

    ak = AlmKanal(
        steps=[
            ForwardModel(
                pick_dict=pick_dict,
                subject_id='sample',
                subjects_dir=tmp_path,
                source=source,
                use_template_mri=False,
            ),
            SpatialFilter(pick_dict=pick_dict),
            SourceReconstruction(
                morph2fsaverage=False,
            ),
        ]
    )

    ak.run(raw)


@pytest.mark.parametrize('source', [source for source, _ in SOURCE_VOL], scope='session')
def test_ad_hoc_cov(gen_mne_data_raw, source, monkeypatch, tmp_path):
    raw, _ = gen_mne_data_raw
    raw = raw.copy().crop(tmax=1)

    _mock_forward_step(monkeypatch, tmp_path)
    _mock_source_application(monkeypatch)

    data_cov = object()
    noise_cov = object()
    fake_filters = object()
    make_ad_hoc_cov = Mock(return_value=noise_cov)

    monkeypatch.setattr(mne, 'compute_raw_covariance', lambda *args, **kwargs: data_cov)
    monkeypatch.setattr(mne, 'compute_rank', lambda *args, **kwargs: {'mag': 1, 'grad': 1})
    monkeypatch.setattr(mne, 'make_ad_hoc_cov', make_ad_hoc_cov)
    monkeypatch.setattr(mne.beamformer, 'make_lcmv', lambda *args, **kwargs: fake_filters)

    pick_dict = {
        'meg': True,
        'eog': False,
        'ecg': False,
        'eeg': False,
        'stim': False,
    }

    ak = AlmKanal(
        steps=[
            ForwardModel(
                pick_dict=pick_dict,
                subject_id='sample',
                subjects_dir=tmp_path,
                source=source,
                use_template_mri=False,
            ),
            SpatialFilter(pick_dict=pick_dict),
            SourceReconstruction(
                morph2fsaverage=False,
            ),
        ]
    )

    ak.run(raw)

    make_ad_hoc_cov.assert_called_once()


def test_rereference_constructor():
    step = ReReference(
        ref_channels=['Cz'],
        projection=False,
        ch_type='eeg',
    )

    assert step.ref_channels == ['Cz']
    assert step.ch_type == 'eeg'
