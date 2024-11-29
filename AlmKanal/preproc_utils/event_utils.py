import mne


def get_events_from_sti(data_raw, sti_ch='STI101'):
    """This function gets all events from the stimulus channel"""

    trigger_min_duration = 9e-3
    events = mne.find_events(
        data_raw, stim_channel=sti_ch, output='onset', min_duration=trigger_min_duration, initial_event=True
    )

    return events


def gen_epochs(raw, event_dict, epoch_settings, events=None, sti_ch='STI101'):
    """
    This function generates epoched data based on an event dictionary.
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
