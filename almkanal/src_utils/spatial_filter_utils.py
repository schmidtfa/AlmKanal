# %%imports
import os
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from attrs import define
from mne._fiff.pick import _contains_ch_type
from numpy.typing import NDArray

from almkanal import (AlmKanal,
                      AlmKanalStep,
                      Maxwell,
                      Filter,
                      Resample,
                      )



from almkanal.preproc_utils.maxwell_utils import run_maxwell


# %%
def get_nearest_empty_room(info: mne.Info, empty_room_dir: str) -> Path:
    """
    Find the empty room recording closest in date to the current measurement.

    This function looks for subdirectories (named as dates in '%y%m%d' format) in the given
    empty_room_dir. It then selects the directory with a date nearest to the measurement date
    (from info['meas_date']) and returns the path to its first file that meets the criteria.

    Parameters
    ----------
    info : mne.Info
        The MEG data information structure, including measurement date.
    empty_room_dir : str
        Directory containing subdirectories named with dates of empty room recordings.

    Returns
    -------
    Path
        Path to the nearest empty room file.

    Raises
    ------
    ValueError
        If no valid empty room recording is found.
    """
    # Build list of valid dates from directory names
    valid_dates = []
    for entry in os.listdir(empty_room_dir):
        try:
            valid_dates.append(datetime.strptime(entry, '%y%m%d'))
        except ValueError:
            continue  # Skip entries that don't match the date format
    if not valid_dates:
        raise ValueError(f'No valid empty room directories found in {empty_room_dir}')

    # Truncate measurement date to day resolution
    meas_date = info['meas_date']
    meas_date_trunc = datetime(meas_date.year, meas_date.month, meas_date.day)

    # Loop until a matching recording is found or no dates remain
    while valid_dates:
        # Find the date closest to the measurement date
        nearest_date = min(valid_dates, key=lambda d: abs(d - meas_date_trunc))
        date_str = nearest_date.strftime('%y%m%d')
        cur_empty_dir = Path(empty_room_dir) / date_str
        files = os.listdir(cur_empty_dir)
        if not files:
            valid_dates.remove(nearest_date)
            continue

        file_name = files[0]
        # Skip directories with unwanted file characteristics
        if 'supine' in file_name:
            valid_dates.remove(nearest_date)
            continue
        if '68' in file_name and 'sss' not in file_name.lower():
            return cur_empty_dir / file_name

        # Fallback: return the file if no specific condition applies
        return cur_empty_dir / file_name

    raise ValueError('No appropriate empty room recording found.')


def preproc_empty_room(
    raw_er: mne.io.Raw,
    data: mne.io.Raw | mne.Epochs,
    preproc_info: dict,
    picks: dict | None,
) -> mne.io.Raw:
    """
    Preprocess an empty room recording to match the preprocessing of the original MEG data.

    Parameters
    ----------
    raw_er : mne.io.Raw
        The raw empty room MEG data.
    data : mne.io.Raw | mne.Epochs
        The original MEG data or epochs for comparison and preprocessing alignment.
    preproc_info : dict
        Configuration object containing preprocessing details (e.g., Maxwell filter settings, ICA).
    pick_dict : PickDictClass
        Dictionary specifying channel selection criteria.

    Returns
    -------
    mne.io.Raw
        The preprocessed empty room MEG data.
    """

    # do channel picking here -> we need to disallow dropping bad
    # channels as this can result in problems
    raw_er.pick(picks=picks)
    if 'Maxwell' in preproc_info:
        if isinstance(data, mne.epochs.Epochs):
            raw = mne.io.RawArray(np.empty([len(data.info.ch_names), 100]), info=data.info)
        elif isinstance(data, mne.io.fiff.raw.Raw):
            raw = data

        raw_er = mne.preprocessing.maxwell_filter_prepare_emptyroom(raw_er=raw_er, raw=raw)
        raw_er = run_maxwell(raw_er, **preproc_info['Maxwell']['maxwell_info'])

    # picks = mne.pick_types(raw_er.info, **pick_dict)
    # raw_er.pick(picks=picks)
    # Add filtering here -> i.e. check if deviation between empty and real data and then filter
    if bool(
        np.logical_and(
            np.isclose(data.info['highpass'], raw_er.info['highpass'], atol=0.01) is False,
            (np.isclose(data.info['lowpass'], raw_er.info['lowpass'], atol=0.01),) is False,
        )
    ):
        raw_er.filter(l_freq=data.info['highpass'], h_freq=data.info['lowpass'])
    elif np.isclose(data.info['highpass'], raw_er.info['highpass'], atol=0.01) is False:
        raw_er.filter(
            l_freq=data.info['highpass'],
            h_freq=None,
        )
    elif np.isclose(data.info['lowpass'], raw_er.info['lowpass'], atol=0.01) is False:
        raw_er.filter(l_freq=None, h_freq=data.info['lowpass'])
    else:
        print('No filtering applied')

    # TODO: Also make sure that the sampling rate is the same
    if np.isclose(data.info['sfreq'], raw_er.info['sfreq'], atol=0.9) is False:
        # adjust for small floating point differences
        raw_er.resample(data.info['sfreq'])

    if 'ICA' in preproc_info:
        component_ids = np.concatenate(preproc_info['ICA']['ica_info']['component_ids'])
        if len(component_ids) == 0:
            component_ids = None

        preproc_info['ICA']['ica_info']['ica'].apply(raw_er, exclude=component_ids)

    return raw_er


def process_empty_room(
    data: mne.io.Raw | mne.Epochs,
    info: mne.Info,
    picks: dict | None,
    preproc_info: dict,
    empty_room: str | mne.io.Raw,
    get_nearest: bool = False,
) -> tuple[NDArray, NDArray]:
    """
    Process the empty room MEG data for noise covariance estimation.

    Parameters
    ----------
    data : mne.io.Raw | mne.Epochs
        The original MEG data or epochs for alignment with the empty room data.
    info : mne.Info
        The MEG data information structure, including measurement metadata.
    pick_dict : PickDictClass
        Dictionary specifying channel selection criteria.
    preproc_info : InfoClass
        Configuration object containing preprocessing details (e.g., Maxwell filter settings, ICA).
    empty_room : str | mne.io.Raw
        Path to the empty room recording or preloaded empty room raw data.
    get_nearest : bool, optional
        If True, finds the nearest empty room recording based on the measurement date. Defaults to False.

    Returns
    -------
    tuple[NDArray, NDArray]
        - `true_rank`: The true rank of the noise covariance matrix.
        - `noise_cov`: The computed noise covariance matrix.
    """

    if np.logical_and(get_nearest, isinstance(empty_room, str)):
        fname_empty_room = get_nearest_empty_room(info, empty_room_dir=empty_room)
        raw_er = mne.io.read_raw(fname_empty_room, preload=True)
    elif np.logical_and(not get_nearest, isinstance(empty_room, str)):
        raw_er = mne.io.read_raw(empty_room, preload=True)
    elif isinstance(empty_room, mne.io.Raw):
        raw_er = empty_room

    raw_er = preproc_empty_room(
        raw_er=raw_er,
        data=data,
        preproc_info=preproc_info,
        picks=picks,
    )

    # if isinstance(data, mne.io.fiff.raw.Raw):
    noise_cov = mne.compute_raw_covariance(raw_er, rank=None, method='auto')

    # TODO: This seems unecessary think about whether i can just drop this elif
    # elif isinstance(data, mne.epochs.Epochs):
    #     t_length = np.abs(data.epoched.tmax - data.epoched.tmin)
    #     raw_er = mne.make_fixed_length_epochs(raw_er, duration=t_length)
    #     noise_cov = mne.compute_covariance(raw_er, rank=None, method='auto')
    # when using noise cov rank should be based on noise cov
    true_rank = mne.compute_rank(noise_cov, info=raw_er.info)  # inferring true rank

    return true_rank, noise_cov


def comp_spatial_filters(
    data: mne.io.Raw | mne.Epochs,
    fwd: mne.Forward,
    pick_dict: dict | None,
    preproc_info: dict,
    data_cov: None | NDArray = None,
    noise_cov: None | NDArray = None,
    empty_room: None | str | mne.io.Raw = None,
    nearest_empty_room: bool = False,
) -> mne.beamformer.Beamformer:
    """
    Compute spatial filters for source reconstruction using LCMV beamformers.

    Parameters
    ----------
    data : mne.io.Raw | mne.Epochs
        MEG data for source reconstruction.
    fwd : mne.Forward
        The forward model.
    pick_dict : dict
        Dictionary specifying channel selection criteria.
    preproc_info : InfoClass
        Configuration object containing preprocessing details (e.g., Maxwell filter settings, ICA).
    data_cov : None | NDArray, optional
        Data covariance matrix. If None, it is computed automatically. Defaults to None.
    noise_cov : None | NDArray, optional
        Noise covariance matrix. If None, it is computed from the empty room recording or ad-hoc. Defaults to None.
    empty_room : None | str | mne.io.Raw, optional
        Path to the empty room recording or preloaded empty room raw data. Defaults to None.
    nearest_empty_room : bool, optional
        Whether to find the nearest empty room recording for noise covariance. Defaults to False.

    Returns
    -------
    mne.beamformer.Beamformer
        LCMV spatial filters for source projection.
    """

    # %Compute covariance matrices
    # data covariance based on the actual recording
    # noise covariance based on empyt rooms
    # select only meg channels from raw

    if isinstance(pick_dict, dict):
        picks = mne.pick_types(data.info, **pick_dict)
        data.pick(picks=picks)
    elif pick_dict is None:
        picks = None

    info = data.info

    # check if multiple channel types are present after picking
    n_ch_types = np.sum([_contains_ch_type(data.info, ch_type) for ch_type in ['mag', 'grad', 'eeg']])

    # compute a data covariance matrix
    if np.logical_and(data_cov is None, isinstance(data, mne.io.BaseRaw)):
        data_cov = mne.compute_raw_covariance(data, rank=None, method='auto')
    elif np.logical_and(data_cov is None, isinstance(data, mne.BaseEpochs)):
        data_cov = mne.compute_covariance(data, rank=None, method='auto')
    else:
        print('Data covariance matrix not computed as it was supplied by the analyst.')

    # if you have mixed sensor types we need a noise covariance matrix
    # per default we take this from an empty room recording
    # importantly this should be preprocessed similarly to the actual data
    if np.logical_and(
        np.logical_and(n_ch_types > 1, noise_cov is None),
        np.logical_or(isinstance(empty_room, str), isinstance(empty_room, mne.io.BaseRaw)),
    ):
        # assert np.logical_or(isinstance(empty_room, str), isinstance(empty_room, mne.io.Raw)), """Please
        # supply either a mne.io.raw object, a path that leads directly
        #  to an empty_room recording or a folder with a bunch of empty room recordings"""
        true_rank, noise_cov = process_empty_room(
            data=data,
            info=info,
            picks=picks,
            preproc_info=preproc_info,
            empty_room=empty_room,
            get_nearest=nearest_empty_room,
        )

    elif np.logical_and(
        np.logical_and(n_ch_types > 1, noise_cov is None),
        np.logical_and(empty_room is None, not nearest_empty_room),
    ):
        warnings.warn("""You have multiple sensor types, but did neither specify a noise covariance
                      matrix or supply a path to an empty room file. Computing an ad-hoc covariance matrix!""")

        noise_cov = mne.make_ad_hoc_cov(info)
        # TODO: check in with thomas if rank should be computed on data_cov if ad-hoc cov is created
        true_rank = mne.compute_rank(data_cov, info=info)

    elif n_ch_types == 1:
        # when we dont have a noise cov we just use the data cov for rank comp
        true_rank = mne.compute_rank(data_cov, info=info)
        noise_cov = None

    lcmv_settings = {
        'reg': 0.05,
        'noise_cov': noise_cov,
        'pick_ori': 'max-power',
        'weight_norm': 'nai',
        'rank': true_rank,
    }

    filters = mne.beamformer.make_lcmv(info, fwd, data_cov, **lcmv_settings)

    # build and apply filters
    return filters, lcmv_settings, noise_cov, data_cov


@define
class SpatialFilter(AlmKanalStep):
    fwd: mne.Forward = None
    pick_dict: dict | None = None
    data_cov: None | NDArray = None
    noise_cov: None | NDArray = None
    empty_room: None | str | mne.io.Raw = None
    nearest_empty_room: bool = False

    must_be_before: tuple = ('SourceReconstruction',)
    must_be_after: tuple = (
        'Maxwell',
        'ICA',
        'ForwardModel',
    )

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict) -> mne.beamformer.Beamformer:
        """
        Compute spatial filters for source projection using LCMV beamformers.

        Parameters
        ----------
        fwd : mne.Forward | None, optional
            The forward model. Defaults to None.
        data_cov : NDArray | None, optional
            Data covariance matrix. Defaults to None.
        noise_cov : NDArray | None, optional
            Noise covariance matrix. Defaults to None.
        empty_room : str | mne.io.Raw | None, optional
            Path to or preloaded empty room recording. Defaults to None.
        get_nearest_empty_room : bool, optional
            Whether to find the nearest empty room recording. Defaults to False.

        Returns
        -------
        None
        """

        if self.pick_dict is None and info['Picks'] is not None:
            self.pick_dict = info['Picks']

        elif self.pick_dict is None and info['Picks'] is None:
            raise ValueError('pick_dict must be provided for spatial filtering.')

        if self.fwd is None:
            self.fwd = info['ForwardModel']['fwd_info']['fwd']

        filters, lcmv_settings, noise_cov, data_cov = comp_spatial_filters(
            data=data,
            fwd=self.fwd,
            pick_dict=self.pick_dict,
            data_cov=self.data_cov,
            noise_cov=self.noise_cov,
            preproc_info=info,
            empty_room=self.empty_room,
            nearest_empty_room=self.nearest_empty_room,
        )
        return {
            'data': data,
            'spatial_filter_info': {
                'filters': filters,
                'lcmv_settings': lcmv_settings,
                'data_cov': data_cov,
                'noise_cov': noise_cov,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        report.add_covariance(
            info['SpatialFilter']['spatial_filter_info']['data_cov'],
            info=data.info,
            title='Data Covariance Matrix',
        )

        if info['SpatialFilter']['spatial_filter_info']['noise_cov'] is not None:  # shouldnt be an ad-hoc noise cov
            report.add_covariance(
                info['SpatialFilter']['spatial_filter_info']['noise_cov']._as_square(),
                info=data.info,
                title='Noise Covariance Matrix',
            )
