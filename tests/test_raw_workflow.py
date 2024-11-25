#%%
import mne
import os
import sys
sys.path.append('/Users/fabian.schmidt/git/AlmKanal')
from AlmKanal import AlmKanal

sample_data_folder = mne.datasets.sample.data_path()
meg_path = os.path.join(sample_data_folder, "MEG", "sample",)
sample_data_raw_file = os.path.join(meg_path, "sample_audvis_raw.fif")
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

#%% Lets initialize the almkanal
ak = AlmKanal(raw=raw)

#%% you can easily use common workflows like maxfiltering in one call
# default arguments have been decided upon by gianpaolo, thomas, nathan and fabian
ak.do_maxwell()
# %% you can always use common mne methods like filtering that modify 
# the raw and epoched objects in place
ak.raw.filter(l_freq=.1, h_freq=100)
#%% one shot call to ica
ak.do_ica( n_components=50,
            method="picard",
            resample_freq=100,
            eog=True,
            ecg=True,
            muscle=False,
            train=False,
            train_freq=16.6,
            threshold=0.4,)

#%% this places the ic object in your almkanal pipeline
# you access it by calling
ak.ica

#%% find events in the data
# they will be automatically added to the AlmKanal
ak.do_events()

ak.events
#%% now we want to epoch the data

event_dict = {
    "Auditory/Left": 1,
    "Auditory/Right": 2,
    "Visual/Left": 3,
    "Visual/Right": 4,
}

ak.do_epochs(event_id=event_dict)

#%% When you have the necessary components assembled you can push the data
# to src space
fwd_fname = os.path.join(meg_path, "sample_audvis-meg-vol-7-fwd.fif")
fwd = mne.read_forward_solution(fwd_fname)

ak.pick_dict['meg'] = 'mag'
ak.fwd = fwd

stc = ak.do_src()

#%%

stc_ave = mne.beamformer.apply_lcmv(ak.epoched["Auditory/Left"].average(), ak.filters) 
# %%
