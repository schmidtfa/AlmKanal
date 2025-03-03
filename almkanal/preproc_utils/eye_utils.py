import mne
import numpy as np
import pandas as pd
from scipy.constants import pi


# TODO: Write function for MEG channel
# TODO: TRF cleaner -> fit trf on blinks & saccades -> subtract prediction from eyedata
def clean_pixx_eye_data(  # noqa PLR0915
    eye_data,
    gaze_lims={'x': 6, 'y': 6},
    filter_settings={'pupil_diameter': (None, 30), 'xy_movements': (0.1, 40)},
    annotate_bads=True,
    trigger_ch_name='STI101',
    tpixx_fs=2000,
    distance=82,
    screen_width=63,
    screen_rect=[0, 0, 1920, 1080],
    verbose=False,
):
    """Preprocess eyetracking data recorded using a TRACKPixx3.

    NOTE: Default settings are based upon the recording setup in the meg lab at the university of salzburg
    and should be adjust if used elsewhere.

    This function can be used to apply some basic preprocessing steps on the .mat file obtained
    from the trackpixx eyetracker. The function returns an mne.io.Raw instance.

    Parameters
    ----------
    eye_data : array
        A numpy array containing the information obtained from the .mat file returned by the eyetracker.
    gaze_lims : dict
        The limits for the gaze in units of ° visual angle. This is used to identify excessive eye movements.
    filter_settings : dict
        The filter settings for the pupil diameter and the eye movements on the x and y axis.
    annotate_bads : bool
        Whether or not bad segments in the data like blinks and excessive eye movements should be annotated.
    tpixx_fs : int
        The sampling rate of the eyetracker
    distance : int
        The distance to the eye tracker in cm
    screen_width : int
        The screen width in cm
    screem_rect : list
        The dimensions of the screen area
    verbose : bool
        Whether or not we want a verbose output


    Returns
    -------
    raw : mne.io.Raw
        Raw object.
    """
    nan_value = 9999.0
    va1_deg_cm = 2 * pi * distance / 360  # visual angle 1 deg [unit:cm]
    px_in_cm = screen_width / screen_rect[2]
    va1_deg_px = np.floor(va1_deg_cm / px_in_cm)
    px2deg = 1 / va1_deg_px
    gaze_xlim = (screen_rect[2] / gaze_lims['x']) * px2deg
    gaze_ylim = (screen_rect[3] / gaze_lims['y']) * px2deg

    # columns labels for the raw data we get from the trackpixx3
    columns = [
        'Time tag',
        'Left Eye x',
        'Left Eye y',
        'Left Eye Pupil Diameter',
        'Right Eye x',
        'Right Eye y',
        'Right Eye Pupil Diameter',
        'Digital Input',
        'Left Eye Blink',
        'Right Eye Blink',
        'Digital Output',
        'Left Eye Fixation',
        'Right Eye Fixation',
        'Left Eye Saccade',
        'Right Eye Saccade',
        'Message code',
        'Left Eye Raw x',
        'Left Eye Raw y',
        'Right Eye Raw x',
        'Right Eye Raw y',
    ]

    df = pd.DataFrame(eye_data, columns=columns)

    df['time'] = df['Time tag'] - df['Time tag'][0]

    if (np.greater(df['Digital Output'].to_numpy(), 256)).sum() > 1:  # just some high value
        df['trigger'] = df['Digital Output'] / 256
    else:
        df['trigger'] = df['Digital Output']

    df[df == nan_value] = np.nan  # bad values

    def movmean(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    t_exc = 0.1  # gaze
    n_blk_smpl = int(tpixx_fs * t_exc)
    blinks_l = movmean(df['Left Eye Blink'], n_blk_smpl) > 0
    blinks_r = movmean(df['Right Eye Blink'], n_blk_smpl) > 0

    t_exc = 0.3  # pupil
    n_blk_smpl = int(tpixx_fs * t_exc)
    blinks_l_pp = movmean(df['Left Eye Blink'], n_blk_smpl) > 0
    blinks_r_pp = movmean(df['Right Eye Blink'], n_blk_smpl) > 0

    # % remove blinks...
    df['Left Eye x'][blinks_l == 1] = np.nan
    df['Right Eye x'][blinks_r == 1] = np.nan

    df['Left Eye y'][blinks_l == 1] = np.nan
    df['Right Eye y'][blinks_r == 1] = np.nan

    df['Left Eye Pupil Diameter'][blinks_l_pp == 1] = np.nan
    df['Right Eye Pupil Diameter'][blinks_r_pp == 1] = np.nan

    # % we take the average across both eyes (should be fine unless you are a chameleon)
    df['x'] = np.nanmean([df['Left Eye x'], df['Right Eye x']], axis=0)
    df['y'] = np.nanmean([df['Left Eye y'], df['Right Eye y']], axis=0)
    df['diameter'] = np.nanmean([df['Left Eye Pupil Diameter'], df['Right Eye Pupil Diameter']], axis=0)

    # convert to °...
    df['x'] *= px2deg
    df['y'] *= px2deg

    df['xy_thd'] = np.logical_or(np.abs(df['x']) > gaze_xlim, np.abs(df['y']) > gaze_ylim)

    for param in ['x', 'y', 'diameter']:
        if np.isnan(df[param].iloc[0]):
            df[param].iloc[0] = np.nanmean(df[param])

        if np.isnan(df[param].iloc[-1]):
            df[param].iloc[-1] = np.nanmean(df[param])

    # % interpolate nans
    df['x'].interpolate(method='pchip', inplace=True)  # TODO: maybe linear
    df['y'].interpolate(method='pchip', inplace=True)
    df['diameter'].interpolate(method='pchip', inplace=True)
    df['blinks'] = np.logical_and(blinks_l_pp, blinks_r_pp)

    # TODO: do trf with blinks and saccades on x,y and diameter -> subtract prediction

    # %move to mne python
    # NOTE: The trigger channel
    info = mne.create_info(
        ch_names=['x', 'y', 'diameter', 'xy_thd', 'blinks'], sfreq=tpixx_fs, ch_types='misc', verbose=verbose
    )
    info_trigger = mne.create_info(ch_names=[trigger_ch_name], sfreq=tpixx_fs, ch_types='stim', verbose=verbose)
    raw = mne.io.RawArray(
        df[['x', 'y', 'diameter', 'xy_thd', 'blinks']].T, info=info, first_samp=df['Time tag'][0], verbose=verbose
    )
    raw_trigger = mne.io.RawArray([df['trigger']], info_trigger, verbose=verbose)

    raw.add_channels([raw_trigger], force_update_info=True)
    raw.set_channel_types(
        {trigger_ch_name: 'stim'},
        # on_unit_change='ignore',
        verbose=verbose,
    )  # a unit change is expected

    xy = mne.pick_channels(ch_names=raw.ch_names, include=['x', 'y'])
    dia = mne.pick_channels(ch_names=raw.ch_names, include=['diameter'])

    # % filter
    raw.filter(
        filter_settings['xy_movements'][0], filter_settings['xy_movements'][1], verbose=verbose, picks=xy
    )  # this needs a hp filter
    raw.filter(
        filter_settings['pupil_diameter'][0], filter_settings['pupil_diameter'][1], verbose=verbose, picks=dia
    )  # this doesnt need a hp filter

    # % annotate bad segments
    if annotate_bads:
        annot = mne.Annotations(
            onset=raw['blinks'][1][raw['blinks'][0][0, :] == 1], duration=1 / tpixx_fs, description='bad_blinks'
        )

        annot.append(onset=raw['xy_thd'][1][raw['xy_thd'][0][0, :] == 1], duration=1 / tpixx_fs, description='bad_view')

        raw.set_annotations(annot)

    return raw
