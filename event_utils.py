import mne



def get_events_from_sti(data_raw, sti_ch, event_fun, event_fun_kwargs, epochs_settings):
    
    '''This function gets all events. I use this for source modelling'''
    
    trigger_min_duration = 9e-3
    events = mne.find_events(data_raw,
                         stim_channel=sti_ch,  #'STI101'
                         output='onset',
                         min_duration=trigger_min_duration,
                         initial_event=True)
    
    
    epochs = mne.Epochs(data_raw,
                    events=events,
                    event_id=event_fun(event_fun_kwargs),
                    **epochs_settings)
        
    return epochs