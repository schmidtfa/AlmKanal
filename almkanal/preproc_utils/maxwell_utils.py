import mne
from numpy.typing import ArrayLike


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
