import mne
from numpy.typing import ArrayLike, NDArray


def get_events_from_sti(data_raw: mne.io.Raw, sti_ch: None | str = 'STI101') -> NDArray:
    """
    Extract events from the stimulus channel in MEG data.

    Parameters
    ----------
    data_raw : mne.io.Raw
        The raw MEG data containing the stimulus channel.
    sti_ch : None | str, optional
        Name of the stimulus channel. Defaults to 'STI101'.

    Returns
    -------
    NDArray
        Array of events extracted from the stimulus channel.
    """

    trigger_min_duration = 9e-3
    events = mne.find_events(
        data_raw, stim_channel=sti_ch, output='onset', min_duration=trigger_min_duration, initial_event=True
    )

    return events


def gen_epochs(
    raw: mne.io.Raw,
    event_dict: dict | None,
    epoch_settings: dict,
    events: None | ArrayLike = None,
    sti_ch: None | str = 'STI101',
) -> mne.Epochs:
    """
    Generate epoched data based on events and an event dictionary.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data to epoch.
    event_dict : dict | None
        Dictionary mapping event IDs to event labels.
    epoch_settings : dict
        Settings for epoch specification (e.g., tmin, tmax, baseline).
    events : None | ArrayLike, optional
        Predefined events array. If None, events are extracted from the stimulus channel. Defaults to None.
    sti_ch : None | str, optional
        Name of the stimulus channel for event extraction. Defaults to 'STI101'.

    Returns
    -------
    mne.Epochs
        The generated epochs object.
    """

    if events is None:
        events = get_events_from_sti(raw, sti_ch=sti_ch)

    epochs = mne.Epochs(raw, events=events, event_id=event_dict, **epoch_settings)

    return epochs


# def gen_trf_epochs(raw, stim, event_dict, events=None, sti_ch='STI101'):
#     """
#     This function takes a stimulus file, related M/EEG data and generates "TRF Epochs" i.e.
#     epochs where each is allowed to be of differing length.
#     """

#     if events is None:
#         events = get_events_from_sti(raw, sti_ch=sti_ch)

#     return trf_epochs
