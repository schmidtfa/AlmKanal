import mne
import numpy as np
from attrs import define
from autoreject import Ransac
from autoreject.utils import interpolate_bads
from numpy.typing import ArrayLike

from almkanal.almkanal import AlmKanalStep


def run_maxwell(
    raw: mne.io.Raw,
    coord_frame: str = 'head',
    destination: None | ArrayLike = None,
    calibration_file: None | str = None,
    cross_talk_file: None | str = None,
    st_duration: float | None = None,
) -> mne.io.Raw:
    """
    Perform Maxwell filtering on raw MEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data to preprocess.
    coord_frame : str, optional
        Coordinate frame for Maxwell filtering ('head' or 'meg'). Defaults to 'head'.
    destination : None | ArrayLike, optional
        Destination coordinate frame for alignment. Defaults to None.
    calibration_file : None | str, optional
        Path to the calibration file. Defaults to None.
    cross_talk_file : None | str, optional
        Path to the cross-talk file. Defaults to None.
    st_duration : float | None, optional
        Duration (in seconds) for tSSS (temporal Signal Space Separation). Defaults to None.

    Returns
    -------
    mne.io.Raw
        The Maxwell-filtered raw MEG data.
    """

    # find bad channels first
    noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(
        raw,
        coord_frame=coord_frame,
        calibration=calibration_file,
        cross_talk=cross_talk_file,  # noqa
    )
    raw.info['bads'] = noisy_chs + flat_chs

    raw = mne.preprocessing.maxwell_filter(
        raw,
        st_duration=st_duration,
        calibration=calibration_file,
        cross_talk=cross_talk_file,
        coord_frame=coord_frame,
        destination=destination,
    )

    return raw


@define
class Maxwell(AlmKanalStep):
    must_be_before: tuple = ('ICA', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()

    mw_coord_frame: str = 'head'
    mw_destination: None | ArrayLike = None
    mw_calibration_file: None | str = None
    mw_cross_talk_file: None | str = None
    mw_st_duration: float | None = None

    def run(
        self,
        data: mne.io.BaseRaw,
        info: dict,
    ) -> dict:
        """
        Apply Maxwell filtering to the raw MEG data.

        Parameters
        ----------
        mw_coord_frame : str, optional
            Coordinate frame for Maxwell filtering ('head' or 'meg'). Defaults to 'head'.
        mw_destination : str | None, optional
            Destination coordinate frame for alignment. Defaults to None.
        mw_calibration_file : str | None, optional
            Path to the calibration file. Defaults to None.
        mw_cross_talk_file : str | None, optional
            Path to the cross-talk file. Defaults to None.
        mw_st_duration : int | None, optional
            Duration (in seconds) for tSSS (temporal Signal Space Separation). Defaults to None.

        Returns
        -------
        None
        """

        # this should do maxwell filtering
        # should only be possible on raw data and only if no other preprocessing apart from filtering was done

        raw_max = run_maxwell(
            raw=data,
            coord_frame=self.mw_coord_frame,
            destination=self.mw_destination,
            calibration_file=self.mw_calibration_file,
            cross_talk_file=self.mw_cross_talk_file,
            st_duration=self.mw_st_duration,
        )

        return {
            'data': raw_max,
            'maxwell_info': {
                'coord_frame': self.mw_coord_frame,
                'destination': self.mw_destination,
                'calibration_file': self.mw_calibration_file,
                'cross_talk_file': self.mw_cross_talk_file,
                'st_duration': self.mw_st_duration,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_maxfiltered')


@define
class MultiBlockMaxwell(AlmKanalStep):
    must_be_before: tuple = ('ICA', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()

    mw_coord_frame: str = 'head'
    mw_destination: None | ArrayLike = None
    mw_calibration_file: None | str = None
    mw_cross_talk_file: None | str = None
    mw_st_duration: float | None = None

    def run(
        self,
        data: list[mne.io.BaseRaw],
        info: dict,
    ) -> dict:
        """
        Apply Maxwell filtering to the raw MEG data.

        Parameters
        ----------
        mw_coord_frame : str, optional
            Coordinate frame for Maxwell filtering ('head' or 'meg'). Defaults to 'head'.
        mw_destination : str | None, optional
            Destination coordinate frame for alignment. Defaults to None.
        mw_calibration_file : str | None, optional
            Path to the calibration file. Defaults to None.
        mw_cross_talk_file : str | None, optional
            Path to the cross-talk file. Defaults to None.
        mw_st_duration : int | None, optional
            Duration (in seconds) for tSSS (temporal Signal Space Separation). Defaults to None.

        Returns
        -------
        None
        """

        # this should do maxwell filtering
        # should only be possible on raw data and only if no other preprocessing apart from filtering was done
        # block_pos_l = [raw.info["dev_head_t"]['trans'][:3, 3] for raw in data]

        # blocks_pos = np.array(block_pos_l)
        # all_distances = np.sqrt(blocks_pos[:,0]**2 + blocks_pos[:,1]**2 + blocks_pos[:,2]**2)
        # mean_distance = np.median(all_distances)
        block_pos_l = [raw.info['dev_head_t']['trans'][:3, 3] for raw in data]
        trans_avg_pos = np.median(block_pos_l, axis=0)

        raw_max_list = []
        for raw in data:
            raw_max_list.append(
                run_maxwell(
                    raw=raw,
                    coord_frame=self.mw_coord_frame,
                    destination=trans_avg_pos,
                    calibration_file=self.mw_calibration_file,
                    cross_talk_file=self.mw_cross_talk_file,
                    st_duration=self.mw_st_duration,
                )
            )

        raw_max = mne.concatenate_raws(raw_max_list)

        return {
            'data': raw_max,
            'maxwell_info': {
                'coord_frame': self.mw_coord_frame,
                'destination': trans_avg_pos,
                'calibration_file': self.mw_calibration_file,
                'cross_talk_file': self.mw_cross_talk_file,
                'st_duration': self.mw_st_duration,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_maxfiltered')


@define
class RANSAC(AlmKanalStep):
    must_be_before: tuple = ('ICA', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()

    ransac_epoch_duration: int | float = 4
    n_resample: int = 50
    min_channels: float = 0.25
    min_corr: float = 0.75
    unbroken_time: float = 0.4
    n_jobs: int = 1
    verbose: bool = False

    def run(
        self,
        data: mne.io.BaseRaw,
        info: dict,
    ) -> dict:
        """
        Apply RANSAC to discover bad channels and interpolate them using autorejects methods.

        Parameters
        ----------
        ransac_epoch_duration : int | float = 4
            Number indicating the length of epochs in seconds that will be generated for the
            continuous file to run RANSAC.

        n_resample : int, optional
            Number of times the sensors are resampled.

        min_channels : float, optional
            Fraction of sensors for robust reconstruction.

        min_corr : float, optional
            Cut-off correlation for abnormal wrt neighbours.

        unbroken_time : float, optional
            Cut-off fraction of time sensor can have poor RANSAC predictability.

        n_jobs: int, optional
            Number of parallel jobs.

        verbose: bool, optional
            The verbosity of progress messages. If False, suppress all output messages.



        Returns
        -------
        None
        """

        epo4ransac = mne.make_fixed_length_epochs(data, duration=self.ransac_epoch_duration)
        # NOTE: We only do this for eeg -> think if this is overall smart
        # So why not use picks from autoreject.Ransac -> I am not sure hwo this will handle eg. ECG or EOG signals
        # Idont want them to be part of the ransac eeg spiel TODO: Test this
        epo4ransac.load_data().pick_types(eeg=True, ecg=False, eog=False)
        ransac = Ransac(
            n_resample=self.n_resample,
            min_channels=self.min_channels,
            min_corr=self.min_corr,
            unbroken_time=self.unbroken_time,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        ransac.fit(epo4ransac)
        bad_chs_eeg = ransac.bad_chs_
        print(f'RANSAC detected the following bad channels: {bad_chs_eeg}')

        data.info['bads'] = bad_chs_eeg
        raw_ransac = interpolate_bads(data, data.info['bads'])

        return {
            'data': raw_ransac,
            'ransac_info': {
                'ransac_epoch_duration': self.ransac_epoch_duration,
                'n_resample': self.n_resample,
                'min_channels': self.min_channels,
                'min_corr': self.min_corr,
                'unbroken_time': self.unbroken_time,
                'n_jobs': self.n_jobs,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_ransac')


@define
class ReReference(AlmKanalStep):
    must_be_before: tuple = ('ICA', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()

    ref_channels = 'average'
    projection = False
    ch_type = 'auto'
    forward = None
    joint = False
    verbose = False

    def run(
        self,
        data: mne.io.BaseRaw,
        info: dict,
    ) -> dict:
        """
        Does ReReferencing of your EEG, Ecog, seeg or dbs channels.
        This is essentially a wrapper around mne.io.Raw.set_eeg_reference.
        The documentation is copied from there.

        Parameters
        ----------
        ref_channels: list of str | str | dict
            Can be:
            The name(s) of the channel(s) used to construct the reference for every channel of ch_type.

            'average' to apply an average reference (default)

            'REST' to use the Reference Electrode Standardization Technique infinity reference [4].

            A dictionary mapping names of data channels to (lists of) names of reference channels.
            For example, {‘A1’: ‘A3’} would replace the data in channel ‘A1’ with the difference between ‘A1’ and ‘A3’.
            To take the average of multiple channels as reference,
            supply a list of channel names as the dictionary value, e.g. {‘A1’: [‘A2’, ‘A3’]}
            would replace channel A1 with A1 - mean(A2, A3).

            An empty list, in which case MNE will not attempt any re-referencing of the data

        projection: bool
            If ref_channels='average' this argument specifies if the average reference should be
            computed as a projection (True) or not (False; default).
            If projection=True, the average reference is added as a projection and is not applied to the data
            (it can be applied afterwards with the apply_proj method).
            If projection=False, the average reference is directly applied to the data.
            If ref_channels is not 'average',
            projection must be set to False (the default in this case).

        ch_typelist of str | str
            The name of the channel type to apply the reference to. Valid channel types are 'auto', 'eeg', 'ecog',
            'seeg', 'dbs'. If 'auto', the first channel type of eeg, ecog, seeg or dbs that is found
            (in that order) will be selected.

        forwardinstance of Forward | None
            Forward solution to use. Only used with ref_channels='REST'.

        jointbool
            How to handle list-of-str ch_type. If False (default), one projector is created per channel type.
            If True, one projector is created across all channel types. This is only used when projection=True.

        verbosebool | str | int | None
            Control verbosity of the logging output. If None, use the default verbosity level.
            See the logging documentation and mne.verbose() for details. Should only be passed as a keyword argument.


        Returns
        -------
        None
        """

        reref = data.set_eeg_reference(
            ref_channels=self.ref_channels,
            projection=self.projection,
            ch_type=self.ch_type,
            forward=self.forward,
            joint=self.joint,
            verbose=self.verbose,
        )

        return {
            'data': reref,
            'ref_info': {
                'ref_channels': self.ref_channels,
                'projection': self.projection,
                'ch_type': self.ch_type,
                'forward': self.forward,
                'joint': self.joint,
                'verbose': self.verbose,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_reref')
