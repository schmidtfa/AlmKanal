import mne
import neurokit2 as nk
import numpy as np
from attrs import define

from almkanal import AlmKanalStep


def run_bio_preproc(
    raw: mne.io.Raw,
    ecg: None | str | list = None,
    resp: None | str | list = None,
    eog: None | str | list = None,
    emg: None | str | list = None,
) -> mne.io.Raw:
    """
    Preprocess physiological signals (ECG, EOG, RESP, EMG) in an MNE raw object.

    This function extracts specified physiological channels, preprocesses them using `neurokit2`,
    and returns a new raw object containing the cleaned physiological data along with stimulus channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MNE object containing physiological and stimulus channels.
    ecg : str | list | None, optional
        Name(s) of the ECG channel(s) to preprocess. Defaults to None.
    resp : str | list | None, optional
        Name(s) of the respiratory channel(s) to preprocess. Defaults to None.
    eog : str | list | None, optional
        Name(s) of the EOG channel(s) to preprocess. Defaults to None.
    emg : str | list | None, optional
        Name(s) of the EMG channel(s) to preprocess. Defaults to None.

    Returns
    -------
    mne.io.Raw
        A new raw object containing the preprocessed physiological signals and stimulus channels.
    """

    bio_chs = {
        'ecg': ecg,
        'resp': resp,
        'eog': eog,
        'emg': emg,
    }

    mne_ch_sel = {key: val for key, val in bio_chs.items() if val is not None}

    for ch, ch_val in mne_ch_sel.items():
        raw.set_channel_types({ch_val: ch})

    bio_raw = raw.copy().pick(list(mne_ch_sel.keys()) + ['stim'])
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


@define
class PhysioCleaner(AlmKanalStep):
    ecg: None | str | list = None
    resp: None | str | list = None
    eog: None | str | list = None
    emg: None | str | list = None

    def run(self, data: mne.io.BaseRaw, info: dict) -> mne.io.Raw:
        """
        Preprocess physiological signals (ECG, EOG, RESP, EMG) in an MNE raw object.

        This method extracts specified physiological channels, preprocesses them using `neurokit2`,
        and returns a new raw object containing the cleaned physiological data along with stimulus channels.

        Parameters
        ----------
        ecg : str | list | None, optional
            Name(s) of the ECG channel(s) to preprocess. Defaults to None.
        resp : str | list | None, optional
            Name(s) of the respiratory channel(s) to preprocess. Defaults to None.
        eog : str | list | None, optional
            Name(s) of the EOG channel(s) to preprocess. Defaults to None.
        emg : str | list | None, optional
            Name(s) of the EMG channel(s) to preprocess. Defaults to None.

        Returns
        -------
        mne.io.Raw
            A new raw object containing the preprocessed physiological signals and stimulus channels.
        """
        data = run_bio_preproc(
            raw=data,
            ecg=self.ecg,
            resp=self.resp,
            eog=self.eog,
            emg=self.emg,
        )

        return {'data': data}

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict):
        pass  # maybe let this function plot ECG ERP etc.
