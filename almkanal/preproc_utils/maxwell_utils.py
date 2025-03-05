import mne
from attrs import define
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
