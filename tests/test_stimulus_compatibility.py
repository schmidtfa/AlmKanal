"""Compatibility contracts for the stimulus-epoching API and module names."""

import pickle
from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.io import wavfile

import almkanal
from almkanal import AlmKanal, AlmKanalStep, Defaults, EpochStimuls, EpochTRF, StimulusSpanSpec, use_defaults
from almkanal._defaults_schema import EpochStimulsDefaults, EpochTRFDefaults
from almkanal.almkanal_steps.trf_utils import EpochTRF as LegacyModuleStep
from almkanal.info import AlmKanalInfo, StepInfo


FIXTURES = Path(__file__).parent / 'fixtures'


def test_old_and_new_exports_are_the_same_step():
    assert EpochTRF is EpochStimuls is LegacyModuleStep
    assert EpochStimuls.__name__ == 'EpochStimuls'
    assert {'EpochStimuls', 'EpochTRF'} <= set(almkanal.__all__)
    assert EpochTRFDefaults is EpochStimulsDefaults
    assert type(Defaults.generic().stimuli) is EpochStimulsDefaults


def test_pre_rename_pickles_load_and_roundtrip():
    step = pickle.loads((FIXTURES / 'legacy_epoch_trf.pickle').read_bytes())
    assert type(step) is EpochStimuls
    assert step.gen_span_spec is dict
    assert step.resample_sfreq == 125
    assert step.feature == ['envelope', 'rectify']
    assert step._defaults_profile.name == 'salzburg'
    assert step.audio_channels == ('MISC007', 'MISC008')
    assert step.hw_delay_s == 0.0165
    assert step._defaults_profile.stimuli.fallback_drift_us_per_s == 499
    restored = pickle.loads(pickle.dumps(step))
    assert type(restored) is EpochStimuls
    assert restored._defaults_profile.to_dict() == step._defaults_profile.to_dict()
    assert restored.resample_sfreq == step.resample_sfreq
    defaults = pickle.loads((FIXTURES / 'legacy_epoch_trf_defaults.pickle').read_bytes())
    assert type(defaults) is EpochStimulsDefaults
    assert defaults == Defaults.generic().stimuli


@pytest.mark.parametrize('name', ['EpochTRF', 'EpochStimuls'])
@pytest.mark.parametrize('direction', ['before', 'after'])
def test_ordering_constraints_resolve_both_names(name, direction):
    stimuli = EpochStimuls(dict, '.')
    other = AlmKanalStep()
    if direction == 'before':
        other.must_be_before = (name,)
        valid = [other, stimuli]
    else:
        other.must_be_after = (name,)
        valid = [stimuli, other]
    AlmKanal(valid)
    with pytest.raises(ValueError, match='Dependency Error'):
        AlmKanal(list(reversed(valid)))


@pytest.mark.parametrize('first, second', [(EpochTRF, EpochStimuls), (EpochStimuls, EpochTRF)])
def test_mixed_aliases_do_not_bypass_duplicate_step_check(first, second):
    with pytest.raises(ValueError, match='EpochStimuls can only occur once'):
        AlmKanal([first(dict, '.'), second(dict, '.')])


@pytest.mark.parametrize('requested', ['EpochTRF', 'EpochStimuls'])
def test_history_lookup_resolves_legacy_and_current_names(requested):
    history = AlmKanalInfo()
    legacy = StepInfo('EpochTRF', {'TRF_info': {'feature': ['envelope']}})
    current = StepInfo('EpochStimuls', {'TRF_info': {'feature': ['rectify']}})
    history.add(legacy)
    history.add(current)
    assert history.get_steps(requested) == [legacy, current]
    assert history.get_step_info(requested, occurrence=0, required=True) is legacy.info
    assert history.get_step_info(requested, required=True) is current.info
    assert [entry.step for entry in history.processing_history] == ['EpochTRF', 'EpochStimuls']


def test_new_and_legacy_entry_points_preserve_data_defaults_and_reports(tmp_path):
    sfreq = 500
    times = np.arange(2000) / sfreq
    raw = mne.io.RawArray(
        np.sin(2 * np.pi * 5 * times)[None], mne.create_info(['EEG'], sfreq, ['eeg']),
        first_samp=1234, verbose=False,
    )
    wav = tmp_path / 'stimulus.wav'
    wavfile.write(wav, 2000, np.sin(2 * np.pi * 100 * np.arange(4000) / 2000).astype(np.float32))
    spec = StimulusSpanSpec({'stimulus': (raw.first_samp + 250, None)})
    profile = Defaults.generic().with_overrides(
        stimuli={'feature': ['envelope', 'rectify'], 'resample_sfreq': 125, 'epoch_len_s': 1, 'verbose': False},
    )
    with use_defaults(profile):
        new_step = EpochStimuls(lambda raw: spec, tmp_path)
        legacy_step = EpochTRF(lambda raw: spec, tmp_path)
    # Compare the renamed and legacy APIs, including independently constructed
    # steps with a captured profile after its context has exited.
    current = new_step.run(raw, AlmKanalInfo())
    legacy = legacy_step.run(raw, AlmKanalInfo())
    np.testing.assert_array_equal(current['data'].get_data(), legacy['data'].get_data())
    assert current['data'].metadata.equals(legacy['data'].metadata)
    assert current['TRF_info'] == legacy['TRF_info']
    assert current['defaults'] == legacy['defaults']
    assert current['defaults']['parameters']['resample_sfreq'] == 125
    assert new_step._defaults_profile is profile
    assert new_step.create_info({'TRF_info': current['TRF_info']}).step == 'EpochStimuls'


def test_span_spec_and_builder_compatibility_aliases():
    from almkanal import StimulusSpanSpec, TRFSpanSpec, build_stimulus_epochs, build_trf_epochs
    from almkanal.almkanal_steps import trf_utils

    assert StimulusSpanSpec is TRFSpanSpec is trf_utils.TRFSpanSpec
    assert StimulusSpanSpec.__name__ == 'StimulusSpanSpec'
    assert build_stimulus_epochs is build_trf_epochs is trf_utils.build_trf_epochs
    assert build_stimulus_epochs.__name__ == 'build_stimulus_epochs'
    assert {'StimulusSpanSpec', 'build_stimulus_epochs', 'TRFSpanSpec', 'build_trf_epochs'} <= set(almkanal.__all__)


def test_pre_rename_span_spec_pickle_loads_and_roundtrips():
    from almkanal import StimulusSpanSpec

    spec = pickle.loads((FIXTURES / 'legacy_trf_span_spec.pickle').read_bytes())
    assert type(spec) is StimulusSpanSpec
    assert spec.spans_by_label == {'story': (1234, None)}
    assert spec.metadata_by_label == {'story': {'condition': 'speech'}}
    assert spec.wav_by_label == {'story': 'story.wav'}
    assert pickle.loads(pickle.dumps(spec)) == spec


def test_pre_rename_profile_migrates_stimulus_defaults():
    step = pickle.loads((FIXTURES / 'legacy_epoch_trf.pickle').read_bytes())
    profile = step._defaults_profile
    assert 'stimuli' in profile.to_dict()
    assert 'trf' not in profile.to_dict()
    assert profile.stimuli == profile.trf
    assert profile.stimuli.feature == ['envelope', 'rectify']
    assert profile.section('trf') == profile.section('stimuli')
    updated = profile.with_overrides(stimuli={'resample_sfreq': 250})
    legacy = profile.with_overrides(trf={'resample_sfreq': 250})
    assert updated.to_dict() == legacy.to_dict()
    assert updated.stimuli.resample_sfreq == 250
    assert profile.stimuli.resample_sfreq == 125


@pytest.mark.parametrize('name', [
    'EpochStimuls', 'EpochTRF', 'StimulusSpanSpec', 'TRFSpanSpec',
    'build_stimulus_epochs', 'build_trf_epochs',
])
def test_legacy_module_exports_and_pickle_paths(name):
    from almkanal.almkanal_steps import stimulus_utils, trf_utils

    current = getattr(stimulus_utils, name)
    assert getattr(trf_utils, name) is current
    assert name in trf_utils.__all__
    assert current.__module__ == 'almkanal.almkanal_steps.stimulus_utils'
    # Protocol 0 GLOBAL records resolve historical imports, including the
    # intermediate names saved before the implementation module moved.
    legacy_reference = f'calmkanal.almkanal_steps.trf_utils\n{name}\n.'.encode()
    assert pickle.loads(legacy_reference) is current
    assert pickle.loads(pickle.dumps(current)) is current
