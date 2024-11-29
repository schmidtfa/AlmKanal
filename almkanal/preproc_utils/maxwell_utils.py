import mne


def run_maxwell(
    raw, coord_frame='head', destination=None, calibration_file=None, cross_talk_file=None, st_duration=None
):
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
