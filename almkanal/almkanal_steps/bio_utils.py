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

    # Map channel types to their provided names (if any)
    bio_chs = {'ecg': ecg, 'resp': resp, 'eog': eog, 'emg': emg}
    mne_ch_sel = {ch: name for ch, name in bio_chs.items() if name is not None}

    # Set channel types in the raw object
    for ch, name in mne_ch_sel.items():
        raw.set_channel_types({name: ch})

    # Create a copy containing only the bio channels and stim channels
    bio_raw = raw.copy().pick(list(mne_ch_sel.keys()) + ['stim'])
    bio_df = bio_raw.to_data_frame()

    # Prepare keyword arguments for neurokit2 bio_process, mapping 'resp' -> 'rsp'
    bio_proc_args = {
        'ecg': bio_df[bio_chs['ecg']] if bio_chs['ecg'] is not None else None,
        'eog': bio_df[bio_chs['eog']] if bio_chs['eog'] is not None else None,
        'rsp': bio_df[bio_chs['resp']] if bio_chs['resp'] is not None else None,
        'emg': bio_df[bio_chs['emg']] if bio_chs['emg'] is not None else None,
    }
    bio_clean = nk.bio_process(**bio_proc_args, sampling_rate=bio_raw.info['sfreq'])[0]

    # Get stimulus channel names
    stim_channel_names = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, stim=True)]

    # Combine channel names and types for the new raw object
    ch_names_all = list(bio_clean.columns) + stim_channel_names
    ch_types_all = ['bio'] * len(bio_clean.columns) + ['stim'] * len(stim_channel_names)
    bio_info = mne.create_info(ch_names=ch_names_all, sfreq=bio_raw.info['sfreq'], ch_types=ch_types_all)

    # Concatenate the processed bio data with stimulus channel data and return a new RawArray
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

        return {
            'data': data,
            'physio_info': {  # hardcoded for now
                'bio_info': {
                    'sampling_rate': 1000,
                    'ecg': {
                        'ecg': self.ecg is not None,
                        'clean_method': 'neurokit',
                        'powerline_hz': 50,
                        'rpeak_method': 'neurokit',
                        'channels': ['ECG'],
                    },
                    'eog': {
                        'eog': self.eog is not None,
                        'clean_method': 'neurokit',
                        'bandpass_hz': [0.25, 7.5],
                        'blink_method': 'mne',
                        'channels': ['VEOG', 'HEOG'],
                    },
                    'emg': {
                        'emg': self.emg is not None,
                        'clean_method': 'biosppy',
                        'amplitude': {'lowcut': 10, 'highcut': 400, 'envelope_lp': 8},
                        'activation_method': 'threshold',
                        'channels': ['EMG1', 'EMG2'],
                    },
                }
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        pass  # maybe let this function plot ECG ERP etc.
