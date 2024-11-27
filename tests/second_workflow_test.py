#%%
import mne
import os
import sys
sys.path.append('../../AlmKanal')
from AlmKanal import AlmKanal
# %%
raw = mne.io.read_raw('../test_data/19610202mrln_resting.fif')
# %%
ak = AlmKanal(raw=raw)
# %%
ak.do_fwd_model(subject_id='19610202mrln',
                base_data_path='/home/schmidtfa/git/AlmKanal/data/',)
# %%
ak.fwd
# %%
