"""Stimulus feature channels survive sensor selection and source reconstruction."""

from types import SimpleNamespace
from unittest.mock import Mock

import mne
import numpy as np
import pytest
from scipy.io import wavfile

import almkanal as ak
from almkanal.almkanal_steps import spatial_filter_utils as sfu
from almkanal.info import AlmKanalInfo, StepInfo


@pytest.mark.parametrize('chans2keep', [None, [], ['EOG']])
@pytest.mark.parametrize(
    'feature, expected_count',
    [('envelope', 1), (['envelope', 'flux'], 2), (['envelope', 'rectify'], 3), (['mel', 'mel_onsets'], 64)],
)
def test_stimulus_features_reach_source_output(tmp_path, monkeypatch, feature, expected_count, chans2keep):
    if chans2keep and feature != ['mel', 'mel_onsets']:
        chans2keep = [*chans2keep, 'envelope']  # Explicit and automatic names can overlap.
    requested_channels = None if chans2keep is None else list(chans2keep)
    sfreq = 200
    rng = np.random.default_rng(42)
    raw = mne.io.RawArray(
        rng.normal(size=(4, 4 * sfreq)),
        mne.create_info(['MEG001', 'MEG002', 'EOG', 'unrequested'], sfreq, ['mag', 'mag', 'eog', 'misc']),
        verbose=False,
    )
    wavfile.write(
        tmp_path / 'trial.wav',
        1000,
        np.sin(2 * np.pi * 100 * np.arange(4000) / 1000).astype(np.float32),
    )
    stimulus = ak.StimulusEpochs(
        lambda raw: ak.StimulusSpanSpec({'trial': (0, raw.n_times)}),
        tmp_path,
        feature=feature,
        resample_sfreq=400,
        epoch_len_s=1,
        verbose=False,
    )
    spatial = ak.SpatialFilter(pick_dict={'meg': True}, chans2keep=chans2keep, data_cov=mne.make_ad_hoc_cov(raw.info))
    fwd = {'src': SimpleNamespace(kind='surface')}

    # Exercise real epoching, resampling, sensor picking, and pipeline history;
    # replace only the anatomy and beamformer computations.
    monkeypatch.setattr(
        ak.ForwardModel,
        'run',
        lambda self, data, info: {
            'data': data,
            'fwd_info': {'fwd': fwd, 'subject_id_freesurfer': 'subject', 'subjects_dir': str(tmp_path)},
        },
    )
    monkeypatch.setattr(ak.ForwardModel, 'reports', lambda *args: None)
    monkeypatch.setattr('almkanal.almkanal.mne.Report', Mock())
    monkeypatch.setattr(mne.beamformer, 'make_lcmv', Mock(return_value={'test_filter': True}))
    apply = Mock(return_value=['source estimate'])
    monkeypatch.setattr(mne.beamformer, 'apply_lcmv_epochs', apply)

    expected = {}
    stimulus_run = ak.StimulusEpochs.run

    def epoch_stimulus(self, data, info):
        result = stimulus_run(self, data, info)
        expected.update({ch: result['data'].get_data(picks=[ch]) for ch in result['data'].ch_names})
        return result

    monkeypatch.setattr(ak.StimulusEpochs, 'run', epoch_stimulus)
    pipeline = ak.AlmKanal(
        [
            stimulus,
            ak.ForwardModel('subject', tmp_path),
            spatial,
            ak.SourceReconstruction(morph2fsaverage=False),
        ],
        pick_params=None,
    )
    result, _ = pipeline.run(raw)

    feature_channels = pipeline.info.get_step_info('StimulusEpochs', required=True)['TRF_info']['feature_channels']
    assert len(feature_channels) == expected_count
    assert feature_channels == list(expected)[4:]
    # Only explicitly requested recording channels join the generated features.
    expected_channels = list(dict.fromkeys([*(chans2keep or []), *feature_channels]))
    assert list(result['extra_data']) == expected_channels
    for ch in expected_channels:
        np.testing.assert_array_equal(result['extra_data'][ch], expected[ch])
        assert result['extra_data'][ch].shape == (4, 1, 400)
    assert apply.call_args.args[0].ch_names == ['MEG001', 'MEG002']
    assert result['fs'] == 400
    assert len(result['metadata']) == 4
    assert spatial.chans2keep == requested_channels


@pytest.mark.parametrize('chans2keep', [None, [], ['CH1']])
@pytest.mark.parametrize('stimulus_info', [None, {'TRF_info': {}}])
def test_spatial_filter_without_recorded_feature_channels(monkeypatch, chans2keep, stimulus_info):
    data = mne.io.RawArray(
        np.arange(200).reshape(2, 100),
        mne.create_info(['CH0', 'CH1'], 100, ['mag', 'eog']),
        verbose=False,
    )
    expected = data.get_data(picks=['CH1'])
    info = AlmKanalInfo()
    if stimulus_info is not None:
        info.add(StepInfo('StimulusEpochs', stimulus_info))
    monkeypatch.setattr(sfu, 'comp_spatial_filters', Mock(return_value=('filters', {}, None, 'cov')))
    result = ak.SpatialFilter(fwd={}, pick_dict={'meg': True}, chans2keep=chans2keep).run(data, info)
    extra_data = result['spatial_filter_info']['extra_data']
    if chans2keep is None:
        assert extra_data is None
    elif not chans2keep:
        assert extra_data == {}
    else:
        assert list(extra_data) == chans2keep
        np.testing.assert_array_equal(extra_data['CH1'], expected)


def test_missing_explicit_channel_still_raises():
    data = mne.io.RawArray(np.zeros((1, 100)), mne.create_info(['MEG'], 100, 'mag'), verbose=False)
    with pytest.raises(ValueError, match='missing'):
        ak.SpatialFilter(fwd={}, pick_dict={'meg': True}, chans2keep=['missing']).run(data, AlmKanalInfo())
