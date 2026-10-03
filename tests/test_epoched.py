import mne
import numpy as np

from almkanal import Epochs, SourceReconstruction, SpatialFilter
from almkanal.info import AlmKanalInfo, StepInfo


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

    spatial_filter = SpatialFilter(
        fwd=fwd,
        pick_dict=pick_dict,
    )
    spatial_result = spatial_filter.run(
        gen_mne_data_epochs,
        info=AlmKanalInfo(),
    )

    source_reconstruction = SourceReconstruction(
        filters=spatial_result['spatial_filter_info']['filters'],
        morph2fsaverage=False,
    )

    info = AlmKanalInfo()

    info.add(
        StepInfo(
            step='ForwardModel',
            info={
                'fwd_info': {
                    'fwd': fwd,
                    'subject_id_freesurfer': 'sample',
                    'subjects_dir': str(data_path / 'subjects'),
                }
            },
        )
    )

    info.add(
        StepInfo(
            step='SpatialFilter',
            info={
                'spatial_filter_info': spatial_result[
                    'spatial_filter_info'
                ],
            },
        )
    )

    result = source_reconstruction.run(
        gen_mne_data_epochs,
        info,
    )

    assert result['data']['label_tc'] is not None


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
        info=AlmKanalInfo(),
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

    info = AlmKanalInfo()
    info.add(
        StepInfo(
            step='Events',
            info={
                'event_info': {
                    'events': events,
                }
            },
        )
    )

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

    info_1 = AlmKanalInfo()
    info_1.add(
        StepInfo(
            step='Events',
            info={
                'event_info': {
                    'events': events_1,
                }
            },
        )
    )

    info_2 = AlmKanalInfo()
    info_2.add(
        StepInfo(
            step='Events',
            info={
                'event_info': {
                    'events': events_2,
                }
            },
        )
    )

    result_1 = step.run(
        raw.copy(),
        info=info_1,
    )

    result_2 = step.run(
        raw.copy(),
        info=info_2,
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