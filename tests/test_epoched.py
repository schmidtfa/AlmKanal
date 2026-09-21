import mne
from almkanal import AlmKanal, SpatialFilter, SourceReconstruction, Epochs
import numpy as np

def test_src(gen_mne_data_epochs): 

    data_path = mne.datasets.sample.data_path()
    meg_path = data_path / 'MEG' / 'sample'
    fwd_fname = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    
    pick_dict = {
    'meg': 'mag',
    'eog': False,
    'ecg': False,
    'eeg': False,
    'stim': False,
}

    ak = AlmKanal(steps=[
                         SpatialFilter(fwd=fwd, pick_dict=pick_dict),
                         SourceReconstruction(source='volume',)])

    ak.run(gen_mne_data_epochs)


def test_epochs_explicit_events(
    gen_mne_data_raw,
):
    raw, _ = gen_mne_data_raw

    events = mne.find_events(raw)

    step = Epochs(
        events=events,
        tmin=-0.1,
        tmax=0.2,
    )

    result = step.run(
        raw,
        info={},
    )

    assert isinstance(
        result['data'],
        mne.BaseEpochs,
    )

    np.testing.assert_array_equal(
        result['epochs_info']['events'],
        events,
    )


def test_epochs_from_previous_events(
    gen_mne_data_raw,
):
    raw, _ = gen_mne_data_raw

    events = mne.find_events(raw)

    info = {
        'Events': {
            'event_info': {
                'events': events,
            }
        }
    }

    step = Epochs(
        tmin=-0.1,
        tmax=0.2,
    )

    result = step.run(
        raw,
        info=info,
    )

    np.testing.assert_array_equal(
        result['epochs_info']['events'],
        events,
    )


def test_epochs_step_can_be_reused(
    gen_mne_data_raw,
):
    raw, _ = gen_mne_data_raw

    events_1 = mne.find_events(raw)
    events_2 = events_1[::2]

    step = Epochs(
        tmin=-0.1,
        tmax=0.2,
    )

    result_1 = step.run(
        raw.copy(),
        info={
            'Events': {
                'event_info': {
                    'events': events_1,
                }
            }
        },
    )

    result_2 = step.run(
        raw.copy(),
        info={
            'Events': {
                'event_info': {
                    'events': events_2,
                }
            }
        },
    )

    np.testing.assert_array_equal(
        result_1['epochs_info']['events'],
        events_1,
    )

    np.testing.assert_array_equal(
        result_2['epochs_info']['events'],
        events_2,
    )

    assert step.events is None    