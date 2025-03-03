# %%
import mne

from almkanal import ICA, AlmKanal, ForwardModel, Maxwell, PhysioCleaner, SourceReconstruction, SpatialFilter

# %%
data_path = mne.datasets.sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# %%
pick_dict = {
    'meg': True,
    'eog': False,
    'ecg': False,
    'eeg': False,
    'stim': False,
}

# Build the pipeline with our dummy functions.
ak = AlmKanal(
    steps=[
        Maxwell(),
        ICA(
            n_components=50,
            train=False,
            eog=True,
            surrogate_eog_chs={
                'left_eog_chs': ['MEG 0121', 'MEG 0311'],
                'right_eog_chs': ['MEG 1211', 'MEG 1411'],
            },
            ecg=True,
            emg=True,
            resample_freq=100,
        ),
        ForwardModel(subject_id='sample', subjects_dir='./data_old', pick_dict=pick_dict, redo_hdm=False),
        SpatialFilter(pick_dict=pick_dict),
        SourceReconstruction(),
    ]
)

ak_physio = AlmKanal(
    steps=[
        PhysioCleaner(eog='EOG 061'),
    ]
)

# %%
proc_data, report = ak_physio.run(raw)

# %%
proc_data, report = ak.run(raw)

# %%
# make objects callable ->
report.save('report_raw.html', overwrite=True)

# %%
Maxwell().run(raw, info={})


# %%
