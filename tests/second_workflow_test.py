#%%
import mne
import sys
sys.path.append('../../AlmKanal')
from AlmKanal import AlmKanal
# %%
raw = mne.io.read_raw('../test_data/19610202mrln_resting.fif')
# %%
ak = AlmKanal(raw=raw)
# %%
ak.do_fwd_model(subject_id='19610202mrln',
                subjects_dir='/home/schmidtfa/git/AlmKanal/data_old/',
                redo_hdm=False)
# %%
stc = ak.do_src(empty_room_path='/home/schmidtfa/git/AlmKanal/test_data/empty_room_68.fif')
# %%
er = mne.io.read_raw('/home/schmidtfa/git/AlmKanal/test_data/empty_room_68.fif')
# %%
er.info['highpass']
# %%
er.info['lowpass']
# %%
