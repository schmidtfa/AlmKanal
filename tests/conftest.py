import pytest
import mne

@pytest.fixture(scope='session')
def gen_mne_data_raw():
    data_path = mne.datasets.sample.data_path()

    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)#.crop(tmin=0, tmax=60)

    raw = raw.pick(picks=['meg', 'eog', 'stim'])

    raw.resample(sfreq=100)

    yield raw, data_path

@pytest.fixture(scope='session')
def gen_mne_data_epochs():

    # % now lets check-out the events
    event_id = {
        'Auditory/Left': 1,
        'Auditory/Right': 2,
        'Visual/Left': 3,
        'Visual/Right': 4,
    }
    tmin = -0.2
    tmax = 0.5

    data_path = mne.datasets.sample.data_path()

    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_raw.fif'
    event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'

    events = mne.read_events(event_fname)
    raw = mne.io.read_raw_fif(raw_fname)

    # Load real data as the template
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    epochs.resample(sfreq=10)

    yield epochs


