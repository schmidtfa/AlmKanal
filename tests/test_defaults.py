from __future__ import annotations

import pickle
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from almkanal import (
    ICA,
    Defaults,
    EpochTRF,
    Events,
    Filter,
    Maxwell,
    TRFSpanSpec,
    configure,
    get_defaults,
    set_defaults,
    use_defaults,
)
from almkanal.almkanal_steps import trf_utils
from almkanal.stim_utils import alignment_utils


@pytest.fixture(autouse=True)
def isolated_defaults():
    """Each test restores the calling context even if configure() is used."""
    with use_defaults(Defaults.generic()):
        yield


def test_generic_and_salzburg_profiles_have_distinct_hardware_assumptions():
    generic = get_defaults()
    assert generic.name == Defaults.generic().name
    assert ICA().train is False
    assert generic.trf.hw_delay_s == 0.0
    assert generic.trf.fallback_drift_us_per_s == 0.0
    assert generic.trf.audio_channels is None

    with use_defaults(Defaults.salzburg()):
        salzburg = get_defaults()
        assert ICA().train is True
        assert salzburg.trf.hw_delay_s == 0.0165
        assert salzburg.trf.fallback_drift_us_per_s == 499.0
        assert salzburg.trf.audio_channels == ('MISC007', 'MISC008')
        step = EpochTRF(lambda raw: TRFSpanSpec({}), '.')
        assert step.audio_channels == ('MISC007', 'MISC008')
        assert EpochTRF(step.gen_span_spec, '.', audio_channels=None).audio_channels is None
        assert EpochTRF(step.gen_span_spec, '.', audio_channels=['recorded']).audio_channels == ['recorded']
        assert salzburg.eye.distance == 82
        assert salzburg.eye.screen_width == 63
        assert list(salzburg.eye.screen_rect) == [0, 0, 1920, 1080]


def test_configure_affects_new_steps_and_explicit_arguments_take_precedence():
    profile = Defaults.salzburg().with_overrides(
        name='another_lab',
        filter={'highpass': 1.5, 'lowpass': 30.0, 'n_jobs': 2},
        ica={'train': True, 'eog': True},
        events={'stim_channel': 'TRIGGER', 'min_duration': 0.005},
        maxwell={'mw_st_duration': 10.0},
    )
    configure(profile)

    assert get_defaults() is profile
    filtered = Filter()
    assert (filtered.highpass, filtered.lowpass, filtered.n_jobs) == (1.5, 30.0, 2)
    assert Events().stim_channel == 'TRIGGER'
    assert Maxwell().mw_st_duration == 10.0
    assert Filter(lowpass=None).lowpass is None
    assert ICA(train=False, eog=False).train is False
    assert ICA(train=False, eog=False).eog is False
    assert Events(min_duration=0).min_duration == 0
    assert Filter(0.0, None).highpass == 0.0
    assert Filter(0.0, None).lowpass is None

    configure(Defaults.generic())
    assert filtered.lowpass == 30.0
    assert filtered._defaults_profile is profile
    assert Filter().lowpass == Defaults.generic().filter.lowpass


def test_nested_scopes_restore_after_configuration_changes_and_exceptions():
    original = get_defaults()
    outer = Defaults.generic().with_overrides(name='outer', filter={'lowpass': 25.0})
    inner = Defaults.generic().with_overrides(name='inner', filter={'lowpass': 35.0})

    with use_defaults(outer):
        assert get_defaults() is outer
        with pytest.raises(RuntimeError, match='processing failed'):
            with use_defaults(inner):
                assert Filter().lowpass == 35.0
                configure(Defaults.salzburg())
                raise RuntimeError('processing failed')
        assert get_defaults() is outer
        assert Filter().lowpass == 25.0

    assert get_defaults() is original


def test_configuration_is_local_to_the_current_context():
    original = get_defaults()
    other = copy_context()
    profile = Defaults.salzburg()

    other.run(configure, profile)

    assert other.run(get_defaults) is profile
    assert get_defaults() is original


@pytest.mark.parametrize('invalid', [None, {}, 42])
def test_configuration_requires_a_defaults_object(invalid):
    original = get_defaults()
    with pytest.raises(TypeError):
        configure(invalid)
    assert get_defaults() is original
    with pytest.raises(TypeError):
        with use_defaults(invalid):
            pytest.fail('Invalid defaults must be rejected before entering the scope.')
    assert get_defaults() is original


@pytest.mark.parametrize(
    'overrides',
    [
        {'missing_section': {}},
        {'filter': {'lowpas': 30.0}},
        {'ica': {'trains': True}},
    ],
)
def test_unknown_sections_and_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        Defaults.generic().with_overrides(**overrides)


@pytest.mark.parametrize(
    'overrides',
    [
        {'filter': {'lowpass': 'thirty'}},
        {'ica': {'train': 'yes'}},
        {'trf': {'fallback_drift_us_per_s': 'slow'}},
    ],
)
def test_obviously_invalid_setting_types_are_rejected(overrides):
    with pytest.raises(TypeError):
        Defaults.generic().with_overrides(**overrides)


def test_profile_settings_cannot_be_changed_through_mutable_aliases():
    original = Defaults.salzburg()
    filters = {'method': 'iir', 'iir_params': {'order': 4, 'ftype': 'butter'}}
    geometry = {'screen_rect': [0, 0, 1280, 720]}
    profile = original.with_overrides(name='custom', filter=filters, eye=geometry)

    filters['iir_params']['order'] = 99
    geometry['screen_rect'][2] = 1
    section = profile.section('filter')
    section['iir_params']['order'] = 98
    exported = profile.to_dict()
    exported['filter']['iir_params']['order'] = 97
    public_parameters = profile.filter.iir_params
    public_parameters['order'] = 96

    assert profile.filter.iir_params['order'] == 4
    assert list(profile.eye.screen_rect) == [0, 0, 1280, 720]
    assert original.filter.method == 'fir'
    assert list(original.eye.screen_rect) == [0, 0, 1920, 1080]
    assert profile.name == 'custom'
    assert profile.version == original.version
    with pytest.raises((AttributeError, TypeError)):
        profile.name = 'changed'
    with pytest.raises((AttributeError, TypeError)):
        profile.filter.lowpass = 1
    with pytest.raises(ValueError):
        profile.section('missing_section')


def test_steps_receive_independent_copies_of_mutable_defaults():
    profile = Defaults.generic().with_overrides(
        filter={'method': 'iir', 'iir_params': {'order': 4, 'ftype': 'butter'}},
    )
    with use_defaults(profile):
        first, second = Filter(), Filter()

    first.iir_params['order'] = 8

    assert second.iir_params['order'] == 4
    assert profile.filter.iir_params['order'] == 4


def test_standalone_step_run_uses_its_captured_profile_and_records_resolved_values():
    profile = Defaults.generic().with_overrides(name='construction', filter={'lowpass': 30.0})
    with use_defaults(profile):
        step = Filter(highpass=None)
    observed = []

    def filter_data(**kwargs):
        observed.append((get_defaults(), kwargs))

    raw = SimpleNamespace(info={'sfreq': 200.0}, filter=filter_data)
    active = get_defaults()
    result = step.run(raw, {})

    assert result['data'] is raw
    assert observed[0][0] is profile
    assert observed[0][1]['l_freq'] is None
    assert observed[0][1]['h_freq'] == 30.0
    assert get_defaults() is active
    assert result['defaults']['profile'] == profile.name
    assert result['defaults']['version'] == profile.version
    assert result['defaults']['parameters']['highpass'] is None
    assert result['defaults']['parameters']['lowpass'] == 30.0


@pytest.mark.parametrize('explicit_step_drift', [False, True])
@pytest.mark.parametrize('explicit_callback_drift', [False, True])
def test_trf_callback_and_helpers_keep_the_step_profile_after_scope_exit(
    monkeypatch,
    tmp_path,
    explicit_step_drift,
    explicit_callback_drift,
):
    profile = Defaults.generic().with_overrides(
        name='recording_setup',
        events={'stim_channel': 'TRIGGER'},
        trf={'fallback_drift_us_per_s': 125.0, 'hw_delay_s': 0.012},
    )
    observed = {}
    raw = SimpleNamespace(info={'sfreq': 1000.0})
    epochs = object()

    def find_trials(data, onset_trigger_to_wav, end_triggers=None, **kwargs):
        observed['trial_profile'] = get_defaults()
        observed['trial_kwargs'] = kwargs
        return [
            {'wav_file': tmp_path / 'sound.wav', 'onset_sample': 0, 'end_sample': 100, 'onset_code': 1, 'end_code': 2}
        ]

    def span_callback(data):
        observed['callback_profile'] = get_defaults()
        kwargs = {'fallback_drift_us_per_s': 0.0} if explicit_callback_drift else {}
        return TRFSpanSpec.from_events(data, {1: 'sound.wav'}, 2, **kwargs)

    def build_epochs(**kwargs):
        observed['build_profile'] = get_defaults()
        observed['build_kwargs'] = kwargs
        return epochs, {}

    monkeypatch.setattr(trf_utils, 'find_audio_trials', find_trials)
    monkeypatch.setattr(trf_utils, '_build_trf_epochs', build_epochs)
    with use_defaults(profile):
        kwargs = {'fallback_drift_us_per_s': 75.0} if explicit_step_drift else {}
        step = EpochTRF(span_callback, tmp_path, **kwargs)

    active = get_defaults()
    result = step.run(raw, {})

    step_drift = 75.0 if explicit_step_drift else 125.0
    callback_drift = 0.0 if explicit_callback_drift else step_drift
    assert result['data'] is epochs
    assert observed['callback_profile'].name == profile.name
    assert observed['callback_profile'].trf.fallback_drift_us_per_s == step_drift
    assert observed['trial_profile'] is observed['callback_profile']
    if not explicit_step_drift:
        assert observed['callback_profile'] is profile
    assert observed['build_profile'] is profile
    assert observed['trial_kwargs']['stim_channel'] == 'TRIGGER'
    assert observed['trial_kwargs']['fallback_drift_us_per_s'] == callback_drift
    assert observed['build_kwargs']['fallback_drift_us_per_s'] == step_drift
    assert observed['build_kwargs']['hw_delay_s'] == 0.012
    assert get_defaults() is active


def test_standalone_step_run_restores_defaults_after_processing_failure():
    with use_defaults(Defaults.salzburg()):
        step = Filter()
    active = get_defaults()

    def broken_filter(**kwargs):
        assert get_defaults() is step._defaults_profile
        raise RuntimeError('processing failed')

    raw = SimpleNamespace(info={'sfreq': 200.0}, filter=broken_filter)
    with pytest.raises(RuntimeError, match='processing failed'):
        step.run(raw, {})

    assert get_defaults() is active


def test_standalone_alignment_resolves_current_defaults_and_preserves_zero(monkeypatch, tmp_path):
    wav_file = tmp_path / 'sound.wav'
    wav_file.touch()
    monkeypatch.setattr(alignment_utils.librosa, 'get_duration', lambda **kwargs: 2.0)
    profile = Defaults.generic().with_overrides(trf={'fallback_drift_us_per_s': 125.0})

    with use_defaults(profile):
        inferred = alignment_utils.assume_raw_wav_alignment(wav_file, 1000.0)
        explicit = alignment_utils.assume_raw_wav_alignment(wav_file, 1000.0, fallback_drift_us_per_s=0.0)

    assert inferred['drift_us_per_s'] == 125.0
    assert inferred['clock_slope'] == pytest.approx(1.000125)
    assert explicit['drift_us_per_s'] == 0.0
    assert explicit['clock_slope'] == 1.0


def test_profiles_and_constructed_steps_survive_pickle_roundtrips():
    profile = Defaults.salzburg().with_overrides(name='stored_lab', filter={'lowpass': 32.0})
    with use_defaults(profile):
        step = Filter(highpass=None)

    restored_profile = pickle.loads(pickle.dumps(profile))
    restored_step = pickle.loads(pickle.dumps(step))

    assert restored_profile.to_dict() == profile.to_dict()
    assert restored_profile.name == profile.name
    assert restored_profile.version == profile.version
    assert restored_step.lowpass == 32.0
    assert restored_step.highpass is None
    assert restored_step._defaults_profile.to_dict() == profile.to_dict()


def test_set_defaults_combines_named_profile_and_user_settings():
    profile = set_defaults(
        'salzburg',
        name='our-lab',
        version='2',
        ica={'train': False, 'resample_freq': None},
        filter={'lowpass': 80.0},
    )
    assert get_defaults() is profile
    assert profile.name == 'our-lab'
    assert profile.version == '2'
    assert Filter().lowpass == 80.0
    assert ICA().train is False
    assert ICA().resample_freq is None
    assert profile.trf.hw_delay_s == 0.0165
    assert Filter(lowpass=None).lowpass is None


def test_set_defaults_resets_to_selected_profile_and_can_explicitly_extend_current():
    set_defaults('salzburg', filter={'lowpass': 80.0})
    set_defaults(get_defaults(), filter={'highpass': 2.0})
    assert Filter().lowpass == 80.0
    assert Filter().highpass == 2.0
    assert ICA().train is True

    set_defaults(filter={'highpass': 1.0})
    assert Filter().highpass == 1.0
    assert Filter().lowpass == 40.0
    assert ICA().train is False

    set_defaults()
    assert Filter().highpass == 0.1
    assert get_defaults().name == 'generic'


def test_scoped_shortcut_accepts_the_same_named_profile_and_overrides():
    original = set_defaults(filter={'lowpass': 30.0})
    with use_defaults('salzburg', name='temporary', filter={'lowpass': 80.0}) as selected:
        assert get_defaults() is selected
        assert ICA().train is True
        assert Filter().lowpass == 80.0
        with use_defaults(filter={'lowpass': None}):
            assert Filter().lowpass is None
            assert ICA().train is False
        assert get_defaults() is selected
    assert get_defaults() is original


@pytest.mark.parametrize(
    'kwargs, error',
    [
        ({'profile': 'missing'}, ValueError),
        ({'profile': None}, TypeError),
        ({'filter': {'lowpas': 20}}, ValueError),
        ({'ica': {'train': 'no'}}, TypeError),
    ],
)
def test_shortcuts_validate_before_changing_active_defaults(kwargs, error):
    original = get_defaults()
    with pytest.raises(error):
        set_defaults(**kwargs)
    assert get_defaults() is original
    with pytest.raises(error):
        with use_defaults(**kwargs):
            pytest.fail('Invalid profile was activated.')
    assert get_defaults() is original


def test_eye_hardware_must_be_configured_without_a_lab_profile():
    from almkanal.almkanal_steps.eye_utils import clean_pixx_eye_data

    with pytest.raises(ValueError, match='tpixx_fs, distance, and screen_width'):
        clean_pixx_eye_data(None)
    with use_defaults('salzburg'):
        # Explicit None must not silently pick a hardware value from the profile.
        with pytest.raises(ValueError, match='tpixx_fs, distance, and screen_width'):
            clean_pixx_eye_data(None, distance=None)


def test_standalone_maxwell_preserves_positional_overrides(monkeypatch):
    from almkanal.almkanal_steps.channel_utils import run_maxwell
    import mne

    observed = []
    monkeypatch.setattr(mne.preprocessing, 'find_bad_channels_maxwell', lambda *args, **kwargs: ([], []))

    def maxwell(raw, **kwargs):
        observed.append(kwargs)
        return raw

    monkeypatch.setattr(mne.preprocessing, 'maxwell_filter', maxwell)
    raw = SimpleNamespace(info={'bads': []})
    with use_defaults('generic', maxwell={'mw_coord_frame': 'meg', 'mw_st_duration': 10.0}):
        run_maxwell(raw)
        run_maxwell(raw, 'head', None, None, None, None)
    assert observed[0]['coord_frame'] == 'meg'
    assert observed[0]['st_duration'] == 10.0
    assert observed[1]['coord_frame'] == 'head'
    assert observed[1]['st_duration'] is None


def test_pipeline_json_records_selected_profile_and_explicit_parameters(monkeypatch, tmp_path):
    import json
    from unittest.mock import Mock

    import mne
    import numpy as np
    from almkanal import AlmKanal

    monkeypatch.setattr('almkanal.almkanal.mne.Report', Mock())
    monkeypatch.setattr(Filter, 'reports', lambda *args: None)
    raw = mne.io.RawArray(np.zeros((1, 1000)), mne.create_info(['Cz'], 100, 'eeg'), verbose=False)
    with use_defaults('generic', name='lab-a', version='2', filter={'highpass': None, 'lowpass': 30.0}):
        pipeline = AlmKanal(steps=[Filter(lowpass=20.0), Filter(lowpass=10.0)])
    set_defaults('salzburg')
    pipeline.run(raw)
    output = tmp_path / 'pipeline.json'
    pipeline.generate_json(str(output))
    history = json.loads(output.read_text())['processing_history']
    assert [entry['step'] for entry in history] == ['Filter', 'Filter']
    saved = history[0]['info']
    assert saved['defaults']['profile'] == 'lab-a'
    assert saved['defaults']['version'] == '2'
    assert saved['defaults']['parameters']['lowpass'] == 20.0
    assert saved['filter_info']['h_freq'] == 20.0
    assert history[1]['info']['defaults']['parameters']['lowpass'] == 10.0
    assert history[1]['info']['filter_info']['h_freq'] == 10.0
    assert get_defaults().name == 'salzburg'


def test_profiles_include_new_main_parameters():
    from almkanal import EEGRANSAC, ForwardModel, MultiBlockMaxwell, Resample

    assert ICA().train_freq == 16.666
    assert ICA().train_thresh == 6.0
    assert get_defaults().ica_train.duration == 8
    with use_defaults(
        'generic',
        maxwell={'mw_st_correlation': 0.95, 'mw_calibration_file': False},
        ransac={'random_state': 12},
        forward_model={'redo_bem': True},
        resample={'window': ('kaiser', 5.0), 'method': 'polyphase'},
    ):
        assert Maxwell().mw_st_correlation == 0.95
        assert MultiBlockMaxwell().mw_st_correlation == 0.95
        assert Maxwell().mw_calibration_file is False
        assert EEGRANSAC().random_state == 12
        assert ForwardModel('subject', '.').redo_bem is True
        assert Resample(100).window == ('kaiser', 5.0)
