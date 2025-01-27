import mne
import neurokit2 as nk
import numpy as np


def run_bio_preproc(
    raw: mne.io.Raw,
    ecg: None | str | list = None,
    resp: None | str | list = None,
    eog: None | str | list = None,
    emg: None | str | list = None,
) -> mne.io.Raw:
    """This function takes a raw mne file and returns a raw object containing preprocessed
    physiological information."""

    bio_chs = {
        'ecg': ecg,
        'resp': resp,
        'eog': eog,
        'emg': emg,
    }

    mne_ch_sel = [key for key, val in bio_chs.items() if val is not None]

    bio_raw = raw.copy().pick(mne_ch_sel + ['stim'])
    bio_df = bio_raw.to_data_frame()

    bio_clean = nk.bio_process(
        ecg=bio_df[bio_chs['ecg']] if bio_chs['ecg'] is not None else None,
        eog=bio_df[bio_chs['eog']] if bio_chs['eog'] is not None else None,
        rsp=bio_df[bio_chs['resp']] if bio_chs['resp'] is not None else None,
        emg=bio_df[bio_chs['emg']] if bio_chs['emg'] is not None else None,
        sampling_rate=bio_raw.info['sfreq'],
    )[0]

    # get all the stim channels
    stim_indices = mne.pick_types(raw.info, stim=True)
    stim_channel_names = [raw.info['ch_names'][i] for i in stim_indices]

    ch_names_bio = bio_clean.columns.tolist() + stim_channel_names
    ch_types_bio = np.tile('bio', len(bio_clean.columns)).tolist() + np.tile('stim', len(stim_channel_names)).tolist()

    bio_info = mne.create_info(ch_names=ch_names_bio, sfreq=bio_raw.info['sfreq'], ch_types=ch_types_bio)

    bio_data = np.concatenate([bio_clean.to_numpy(), bio_df[stim_channel_names].to_numpy()], axis=1).T

    return mne.io.RawArray(bio_data, bio_info)
