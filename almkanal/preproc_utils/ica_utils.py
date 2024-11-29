from pathlib import Path

import mne
import numpy as np
from pyrasa.irasa import irasa
from pyrasa.utils.peak_utils import get_band_info
from scipy.stats import zscore


def plot_ica(raw, ica, components_dict, bad_ids, fname, img_path):
    """Function to plot the ICA components"""

    if len(bad_ids) > 0:
        titles = {}
        for key, vals in components_dict.items():
            for val in vals:
                titles.update({int(val): f'IC {key}'})
        # save the ica figures
        fig = ica.plot_components(picks=bad_ids, ch_type='mag', inst=raw, title=titles, show=False)
        plot_file_name = f'{fname}_ICA_summary_plot.png'
        # if the directory doesn't exist, create it
        if not Path.exists(Path(img_path) / 'ica_pngs'):
            Path.mkdir(Path(img_path) / 'ica_pngs')

        fig.savefig(Path(img_path) / 'ica_pngs' / plot_file_name)
    else:
        print('Error: No Components detected using the current settings. So no plots will be generated.')


def run_ica(
    raw,
    n_components=None,
    method='picard',
    resample_freq=None,
    eog=True,
    ecg=True,
    train=True,
    train_freq=16,
    muscle=False,
    ica_corr_thresh=0.5,
    img_path=None,
    fname=None,
):
    # we run ica on high-pass filtered data
    raw_copy = raw.copy().filter(l_freq=1, h_freq=None)
    if resample_freq is not None:
        raw_copy.resample(resample_freq)
    ica = mne.preprocessing.ICA(n_components=n_components, method=method)
    ica.fit(raw_copy)

    bads = []
    components_dict = {}
    # find which ICs match the EOG/ECG pattern using correlation
    if eog:
        eog_idcs, _ = ica.find_bads_eog(raw_copy, measure='correlation', threshold=ica_corr_thresh)
        components_dict.update({'eog': eog_idcs})
        bads.append(eog_idcs)
    if ecg:
        # take ecg based on correlation
        try:
            ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_copy)  # , picks=ecg_list, verbose=True
        except Exception as e:
            print(f'Error creating ECG epochs: {e}')
            print('Apparently the ECG003 channel is not informative, therefore')
            print('removing ECG003 channel and trying with the MEG only channels')

            if 'ECG003' in raw_copy.ch_names:
                raw_copy.drop_channels('ECG003')

            ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_copy)

        # find the ECG components,
        ecg_idcs, _ = ica.find_bads_ecg(ecg_epochs, measure='correlation', threshold=ica_corr_thresh)
        components_dict.update({'ecg': ecg_idcs})
        bads.append(ecg_idcs)
    # remove muscle based on slope increase between 9-100hz
    if muscle:
        raise NotImplementedError('muscle rejection via ICA not yet implemented')
    # remove train based
    if train:
        train_idcs = find_train_ica(raw_copy, ica, train_freq)
        components_dict.update({'train': train_idcs})
        bads.append(train_idcs)

    bad_ids = np.concatenate(bads).astype(int).tolist() if len(bads) > 0 else []

    raw.info['description'] = f'# excluded components: {len(bad_ids)}; excluded ICA: {bad_ids}'

    # plot data if wanted
    if img_path is not None:
        plot_ica(raw, ica, components_dict, bad_ids, fname, img_path)
    # % drop physiological components
    ica.apply(raw, exclude=bad_ids)

    return raw, ica, bad_ids


def find_train_ica(raw, ica, train_freq, duration=4, overlap=0.5, hmax=2, sd=2):
    """
    This function extracts independent components based on narrowband spectral peaks.
    We use this mainly to detect an artifact in our MEG data caused, by a nearby train station.
    The artifact is a narrowband 16.666Hz oscillation.
    """

    # we add a small value to the hmax to be absolutely sure i am not going into filters
    # choice is pretty arbitrary we just want to absolutely avoid runnning into errors and
    # the artifact of interest is usually not on the borders of the spectrum.
    hmax_filt = hmax + 0.5

    # get info for train timeseries
    ic_signal = ica.get_sources(raw).get_data()

    # run irasa
    fs = raw.info['sfreq']

    lower, upper = raw.info['highpass'] / hmax_filt, raw.info['lowpass'] / hmax_filt

    irasa_out = irasa(
        ic_signal,
        fs=fs,
        band=(lower, upper),
        psd_kwargs={'nperseg': int(duration * fs), 'noverlap': int(duration * fs * overlap)},
        hset_info=(1, hmax, 0.05),
    )

    train_peaks = get_band_info(
        irasa_out.get_peaks(), freq_range=(train_freq - 1, train_freq + 1), ch_names=np.arange(ic_signal.shape[0])
    ).dropna()

    # select the right number of components based on a thresholding procedure
    bad_ics = []
    if train_peaks.shape[0] > 1:
        while (zscore(train_peaks['pw']) > sd).sum() > 0:
            bad_ch = train_peaks[zscore(train_peaks['pw']) > sd]['ch_name'].values
            bad_ics.append(bad_ch)
            train_peaks = train_peaks.query(f'ch_name != {bad_ch}')
    else:
        bad_ics.append(train_peaks['ch_name'].values)

    if len(bad_ics) > 0:
        bad_ics = [int(val) for val in np.concatenate(bad_ics)]

    return bad_ics
