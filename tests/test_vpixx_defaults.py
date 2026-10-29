"""Defaults integration for VPixx loading, alignment, and pipeline steps."""

from unittest.mock import Mock

import mne
import numpy as np
import pytest

import almkanal as ak
from almkanal.almkanal_steps import eye_utils
from almkanal.eye_utils import gaze_utils
from almkanal.eye_utils.gaze_utils import VPixxConfig
from almkanal.info import AlmKanalInfo
from tests.test_eye_utils import make_synthetic_meg, make_synthetic_vpixx_mat


@pytest.fixture(autouse=True)
def isolated_defaults():
    with ak.use_defaults():
        yield


def test_generic_geometry_is_optional_only_for_pixel_processing(tmp_path):
    path = tmp_path / 'eye.mat'
    make_synthetic_vpixx_mat(path)

    assert ak.VPixxCleaner(path).screen_resolution is None
    assert VPixxConfig().screen_size is None
    with pytest.raises(ValueError, match='must be configured to convert gaze to radians'):
        gaze_utils.load_eyetracking_data(path)

    ak.set_defaults(vpixx={'convert_to_radians': False, 'include_raw_vpixx_channels': True})
    raw = gaze_utils.load_eyetracking_data(path)
    assert 'Left Eye x' in raw.ch_names
    assert np.max(raw.get_data(picks='eyetracker_x')) > 900  # Still pixels.

    raw = gaze_utils.load_eyetracking_data(path, include_raw_vpixx_channels=False)
    assert 'Left Eye x' not in raw.ch_names


def test_vpixx_config_captures_defaults_and_explicit_geometry_wins():
    ak.set_defaults('salzburg', vpixx={'blink_buffer_vpixx': (0.1, 0.3)})
    config = VPixxConfig(screen_distance=0.7)
    step = ak.VPixxCleaner('eye.mat', interpolate_blinks=False, blink_buffer=None)

    ak.set_defaults()
    calibration = gaze_utils.create_vpixx_calibration(config)
    assert calibration['screen_distance'] == 0.7
    assert calibration['screen_size'] == (0.61, 0.34)
    assert config.blink_buffer == (0.1, 0.3)
    assert step.screen_distance == 0.82
    assert step.interpolate_blinks is False
    assert step.blink_buffer is None
    with pytest.raises(ValueError, match='must be configured'):
        gaze_utils.create_vpixx_calibration()


def test_standalone_loader_resolves_current_profile_and_respects_config(monkeypatch, tmp_path):
    path = tmp_path / 'eye.mat'
    make_synthetic_vpixx_mat(path)
    observed = []
    buffers = []

    def convert(raw, *, calibration, to):
        observed.append(calibration['screen_distance'])
        return raw

    def interpolate(raw, *, buffer, interpolate_gaze):
        buffers.append(buffer)
        return raw

    monkeypatch.setattr(mne.preprocessing.eyetracking, 'convert_units', convert)
    monkeypatch.setattr(mne.preprocessing.eyetracking, 'interpolate_blinks', interpolate)

    ak.set_defaults('salzburg', vpixx={'blink_buffer': (0.2, 0.4)})
    explicit_config = VPixxConfig(screen_distance=0.6, blink_buffer=(0.0, 0.0))
    gaze_utils.load_eyetracking_data(path)
    ak.set_defaults('salzburg', vpixx={'screen_distance': 0.9, 'blink_buffer': (0.3, 0.5)})
    gaze_utils.load_eyetracking_data(path)
    # Explicit None bypasses the profile's interpolation-buffer override and
    # requests the buffer from this explicit configuration.
    gaze_utils.load_eyetracking_data(path, config=explicit_config, blink_buffer=None)
    assert observed == [0.82, 0.9, 0.6]
    assert buffers == [(0.2, 0.4), (0.3, 0.5), (0.0, 0.0)]


@pytest.mark.parametrize('concatenate', [False, True])
def test_step_captures_profile_for_blocks_and_records_settings(monkeypatch, tmp_path, concatenate):
    observed = []
    buffers = []
    original_loader = eye_utils.load_eyetracking_data
    original_interpolate = mne.preprocessing.eyetracking.interpolate_blinks

    def load(*args, **kwargs):
        observed.append((ak.get_defaults().name, kwargs['config'].screen_size))
        return original_loader(*args, **kwargs)

    def interpolate(raw, **kwargs):
        buffers.append(kwargs['buffer'])
        return original_interpolate(raw, **kwargs)

    monkeypatch.setattr(eye_utils, 'load_eyetracking_data', load)
    monkeypatch.setattr(mne.preprocessing.eyetracking, 'interpolate_blinks', interpolate)
    monkeypatch.setattr('almkanal.almkanal.mne.Report', Mock())
    paths = [tmp_path / 'first.mat', tmp_path / 'second.mat']
    for path in paths:
        make_synthetic_vpixx_mat(path)

    with ak.use_defaults(
        'salzburg',
        name='study-eye',
        version='2',
        vpixx={'concatenate_raws': concatenate, 'blink_buffer_vpixx': (0.1, 0.3)},
    ):
        step = ak.VPixxCleaner(paths, screen_size=(0.53, 0.30))
        pipeline = ak.AlmKanal([step])

    # Run after leaving the profile scope: both the step and its helpers must
    # continue to use the construction-time settings.
    processed, _ = pipeline.run([make_synthetic_meg(), make_synthetic_meg()])
    assert isinstance(processed, mne.io.BaseRaw if concatenate else list)
    if concatenate:
        assert processed.n_times == 10_000
    else:
        assert len(processed) == 2
        assert all('eyetracker_x' in raw.ch_names for raw in processed)
    assert observed == [('study-eye', (0.53, 0.30))] * 2
    assert buffers == [(0.1, 0.3)] * 2
    saved = pipeline.generate_json()['processing_history'][0]
    assert saved['step'] == 'VPixxCleaner'
    assert saved['info']['defaults']['profile'] == 'study-eye'
    assert saved['info']['defaults']['version'] == '2'
    assert saved['info']['defaults']['parameters']['concatenate_raws'] is concatenate
    assert saved['info']['defaults']['parameters']['screen_size'] == [0.53, 0.30]
    assert ak.get_defaults().name == 'generic'


def test_single_raw_and_explicit_concatenation_override(tmp_path):
    path = tmp_path / 'eye.mat'
    make_synthetic_vpixx_mat(path)
    with ak.use_defaults('salzburg', vpixx={'concatenate_raws': True}):
        single = ak.VPixxCleaner(path)
        blocks = ak.VPixxCleaner([path], concatenate_raws=False)
    assert isinstance(single.run(make_synthetic_meg(), AlmKanalInfo())['data'], mne.io.BaseRaw)
    assert isinstance(blocks.run([make_synthetic_meg()], AlmKanalInfo())['data'], list)


def test_alignment_profile_and_explicit_none_zero_overrides(monkeypatch):
    observed = []
    matched = []
    events = np.array([[10, 0, 1], [20, 0, 2], [30, 0, 9000]])

    def find_events(raw, **kwargs):
        observed.append(kwargs)
        return events.copy()

    def realign(raw, other, *, t_raw, t_other, **kwargs):
        matched.append(len(t_raw))

    monkeypatch.setattr(mne, 'find_events', find_events)
    monkeypatch.setattr(mne.preprocessing, 'realign_raw', realign)
    meg = make_synthetic_meg()
    eye = make_synthetic_meg().rename_channels({'STI101': 'Digital Output'})
    with ak.use_defaults('salzburg'):
        gaze_utils.align_eye_to_meg(meg, eye)
        gaze_utils.align_eye_to_meg(meg, eye, meg_stim_channel=None, meg_min_duration=0, meg_max_trigger=None)
    assert observed[1]['stim_channel'] == 'STI101'
    assert observed[1]['min_duration'] == 0.002
    assert observed[3]['stim_channel'] is None
    assert observed[3]['min_duration'] == 0
    assert matched == [2, 3]


def test_add_helper_shares_alignment_and_loading_defaults(monkeypatch, tmp_path):
    path = tmp_path / 'eye.mat'
    # Binary-exact timestamps keep sampling rates identical with alignment
    # deliberately disabled/mocked in this test.
    make_synthetic_vpixx_mat(path, sfreq=1024)
    observed = []
    monkeypatch.setattr(gaze_utils, 'align_eye_to_meg', lambda raw, eye, **kwargs: observed.append(kwargs))
    with ak.use_defaults('salzburg', vpixx={'meg_max_trigger': 512, 'meg_min_duration': 0.01, 'align': False}):
        gaze_utils.add_eye_tracking_data(make_synthetic_meg(sfreq=1024), path)
        assert observed == []
        gaze_utils.add_eye_tracking_data(make_synthetic_meg(sfreq=1024), path, align=True, meg_max_trigger=None)
    assert observed == [
        {
            'meg_stim_channel': 'STI101',
            'eye_stim_channel': 'Digital Output',
            'meg_min_duration': 0.01,
            'meg_max_trigger': None,
        }
    ]


@pytest.mark.parametrize(
    'settings',
    [
        {'screen_resolution': (1920,)},
        {'screen_size': (0.5, 'wide')},
        {'screen_distance': True},
        {'calibration_eye': 'both'},
        {'meg_max_trigger': 4.5},
    ],
)
def test_invalid_profile_settings_do_not_change_active_profile(settings):
    current = ak.get_defaults()
    with pytest.raises(TypeError):
        ak.set_defaults('salzburg', vpixx=settings)
    assert ak.get_defaults() is current


@pytest.mark.parametrize(
    'settings',
    [
        {'screen_size': (0.0, 0.3)},
        {'screen_distance': float('nan')},
        {'screen_resolution': (1920, -1)},
        {'blink_buffer': (-0.1, 0.2)},
    ],
)
def test_invalid_geometry_and_buffers_are_rejected(settings):
    with pytest.raises(ValueError):
        VPixxConfig(**settings)


def test_step_failure_restores_active_profile():
    with ak.use_defaults('salzburg'):
        step = ak.VPixxCleaner(['first.mat', 'second.mat'])
    with pytest.raises(ValueError, match='Each MEG recording'):
        step.run(make_synthetic_meg(), AlmKanalInfo())
    assert ak.get_defaults().name == 'generic'
