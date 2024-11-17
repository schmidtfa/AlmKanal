#%%
import mne
import os
import matplotlib.pyplot as plt
import scipy.signal as dsp

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
# %%

# %%
import sys
sys.path.append('/Users/fabian.schmidt/git/AlmKanal')
from raw_cleaner import raw_cleaner
# %%
raw, ics = raw_cleaner(raw, l_pass=None, h_pass=0.1, ica=True, mw=False)
# %%
fs = raw.info['sfreq']
freq, psd = dsp.welch(ics['ic_src_tc'].get_data(), fs=fs, nperseg=4*fs, noverlap=2*fs)
# %%
plt.loglog(freq, psd[7].T);
# %%
ics
# %%
plt.plot(ics['ic_src_tc'].get_data()[30][:2000])
# %%
