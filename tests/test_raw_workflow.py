#%%
import mne
import os
import matplotlib.pyplot as plt
import scipy.signal as dsp

sample_data_folder = mne.datasets.sample.data_path()
meg_path = os.path.join(sample_data_folder, "MEG", "sample",)
sample_data_raw_file = os.path.join(meg_path, "sample_audvis_raw.fif")
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
# %%
import sys
sys.path.append('/Users/fabian.schmidt/git/AlmKanal')
from raw_cleaner import raw_cleaner

#%%

preproc_settings = {'l_pass' : None, 
                    'h_pass' : 0.1,
                    'mw' : False,
                    'ica': False,
                    }

raw = raw_cleaner(raw, **preproc_settings)

preproc_settings['ica'] = False

from utils.src_utils import raw2source
pick_dict = {'meg': 'mag', 
             'eeg': False, 
             'stim': True, 
             'exclude': "bads"}

#%% make a fwd model using either a template or real MRI
# from utils.head_model_utils import make_fwd
# source='surface'
# template_mri=True
# trans_path = '/home/schmidtfa/experiments/brain_age/data/data_cam_can/headmodels/',
# subject_id = ''
# subjects_dir = ''

# fwd = make_fwd(info, source, trans_path, subjects_dir, subject_id, template_mri)

fwd_fname = os.path.join(meg_path, "sample_audvis-meg-vol-7-fwd.fif")
fwd = mne.read_forward_solution(fwd_fname)

stc = raw2source(raw,
                 fwd,
                 pick_dict=pick_dict,
                 )
#%%
event_fname = os.path.join(meg_path, "sample_audvis_filt-0-40_raw-eve.fif")
events = mne.read_events(event_fname)

from utils.event_utils import gen_epochs

event_dict = {
    "Auditory/Left": 1,
    "Auditory/Right": 2,
    "Visual/Left": 3,
    "Visual/Right": 4,
}

epoch_settings = {
    'tmin': -.1,
    'tmax': 0.3,
    'proj': False,
    'picks':None,
    'baseline': None,
}

epochs = gen_epochs(stc, event_dict=event_dict, events=events, epoch_settings=epoch_settings)
# %%
