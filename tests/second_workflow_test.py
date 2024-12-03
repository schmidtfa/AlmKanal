# %%
import mne
from almkanal import AlmKanal

# %%
raw = mne.io.read_raw('../test_data/19610202mrln_resting.fif', preload=True)
# %%
ak = AlmKanal(raw=raw)
# %% do raw preproc
ak.do_maxwell()
#%%
ak.raw.filter(l_freq=0.1, h_freq=100)
#%%
ak.do_ica()

# %% do fwd model
ak.do_fwd_model(subject_id='19610202mrln', 
                subjects_dir='/home/schmidtfa/git/AlmKanal/data_old/', 
                redo_hdm=True)
# %% go 2 source
stc = ak.do_src(
    subject_id='19610202mrln',
    subjects_dir='/home/schmidtfa/git/AlmKanal/data_old/',
    return_parc=True,
    empty_room_path='/home/schmidtfa/git/AlmKanal/test_data/empty_room_68.fif',
)

# %%
import matplotlib.pyplot as plt
import scipy.signal as dsp
import numpy as np

#sns.set_context('notebook')
#sns.set_style('ticks')

# %%
fs = ak.raw.info['sfreq']
freqs, psd = dsp.welch(stc['label_tc'], fs=fs, nperseg=4 * fs, noverlap=2 * fs)

# %%


f, ax = plt.subplots(figsize=(5, 5))

fmask = freqs < 100
ax.loglog(freqs[fmask], np.mean(psd.T[fmask], axis=1))
# %%
fs = ak.raw.info['sfreq']
freqs2, psd2 = dsp.welch(ak.raw.get_data(picks='mag'), fs=fs, nperseg=4 * fs, noverlap=2 * fs)

# %%
f, ax = plt.subplots(figsize=(5, 5))

fmask = freqs2 < 100
ax.loglog(freqs2.T[fmask], np.mean(psd2.T[fmask], axis=1))

# %%
