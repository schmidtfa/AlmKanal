import mne
import numpy as np
import pandas as pd

from copy import deepcopy


def filter_event_id(event_id, conditions):
    """Filter mne-python style event_ids.

    Parameters
    ----------
    event_id : dict
        The event_id
    conditions : str or list or tuple
        The conditions to keep

    Returns
    -------
    The filtered event_id

    """
    conditions = [conditions] if not isinstance(conditions,
                                                (list, tuple)) else conditions
    good_event_keys = mne.epochs._hid_match(event_id, conditions)

    new_event_id = {key: value for key, value in event_id.items() if
                    key in good_event_keys}
    return new_event_id


def read_events_from_analogue(raw, tolerance=1, trigger_channels=None):
    if trigger_channels is None:
        trigger_channels = ['STI001', 'STI002', 'STI003',
                            'STI004', 'STI005', 'STI006', 'STI007', 'STI008']

    trigger_data = raw[mne.pick_channels(raw.ch_names, trigger_channels)][0] / 5  # noqa

    bit_mult = 2 ** np.arange(0, trigger_data.shape[0])

    trigger_values = np.sum((trigger_data.T * bit_mult).T, axis=0).astype(int)

    trigger_idx = np.where(np.diff(trigger_values) > 0)[0] + 1
    tmp_events = np.zeros((trigger_idx.shape[0], 3))
    tmp_events[:, 0] = trigger_idx + raw.first_samp
    tmp_events[:, 2] = trigger_values[trigger_idx]

    bad_triggers_idx = np.where(np.diff(tmp_events[:, 0]) <= tolerance)[0]

    tmp_events[bad_triggers_idx + 1, 0] = tmp_events[bad_triggers_idx, 0]
    events = np.delete(tmp_events, bad_triggers_idx, axis=0)

    return events.astype(int)


class AdvancedEvents(mne.io.fiff.Raw):
    """Integrate event loading and handling into :class:`mne.io.Raw`.

    Including this mixin in your study specific ``Raw`` class provides event
    handling features directly in that class.

    More specifically, it provides three extra properties:

    1. events
    2. event_id
    3. evt_metadata

    Which are automatically filled and kept up-to-date. They correspond to
    the respective meaning in :class:`mne.Epochs`.

    You can also create a subclass of this class and use
    :meth:`_process_events` to process the events (fill the event_id,
    modify the event codes....)
    """

    trigger_min_duration = 9e-3

    def __init__(self, *args, **kwargs):
        self._events = None
        self._event_id = None
        self._evt_metadata = None

        super(AdvancedEvents, self).__init__(*args, **kwargs)
        self._load_events()

    def _load_events(self):
        self._event_id = dict()
        self._events = mne.find_events(self,
                                       min_duration=self.trigger_min_duration)

        self._process_events()

    def _process_events(self):
        pass

    def get_filtered_event_id(self, condition_filter):
        """Return a filtered version of the event_id field.

        Refer to :meth:`obob_mne.events.filter_event_id` for further
        details.
        """
        return filter_event_id(self.event_id, condition_filter)

    def has_filtered_events(self, condition_filter):
        """Check whether the event_ids are present.

        Parameters
        ----------
        condition_filter : str
            The event_ids to check

        Returns
        -------
        has_events : bool
            True if the filtered events are present.

        """
        try:
            self.get_filtered_event_id(condition_filter)
        except KeyError:
            return False

        return len(self.get_filtered_event_id(condition_filter)) > 0

    def resample(self, *args, **kwargs):
        """Resample the data and reloads the events.

        For the rest, refer to :meth:`mne.io.Raw.resample`.

        """
        self._events = None
        self._event_id = None
        self._evt_metadata = None

        super(AdvancedEvents, self).resample(*args, **kwargs)
        self._load_events()

    @property
    def events(self):
        """:class:`numpy.ndarray`: The event matrix."""
        if not isinstance(self._events, np.ndarray):
            self._load_events()

        return self._events

    @property
    def event_id(self):
        """dict: The event_ids"""
        if not isinstance(self._events, np.ndarray):
            self._load_events()

        return self._event_id

    @property
    def evt_metadata(self):
        """:class:`pandas.DataFrame`: The metadata"""
        if not isinstance(self._events, np.ndarray):
            self._load_events()

        return self._evt_metadata


class AutomaticBinaryEvents(AdvancedEvents):
    """Mixin for binary events.

    If your triggers code events with binary triggers, this mixin can help you
    a lot.

    Let's suppose, you have an experiment with two types of blocks.
    At the beginning of each block, the type of the block is signalled by a
    a trigger code:

    1. Attend Auditory: Trigger 1
    2. Attend Visual: Trigger 2

    Then you present either:

    1. A tone: Trigger 4
    2. An image: Trigger 8

    And sometimes, one of them is an oddball which is marked by adding 1 to
    the trigger codes.

    In this case, you can use this mixin and write something like this:

    .. code-block:: python

        class Raw(mne.io.fiff.Raw, LoadFromSinuhe, AutomaticBinaryEvents):
            study_acronym = 'test_study'

            condition_triggers = {
                'attention': {
                    'auditory': 1,
                    'visual': 2
                }
            }

            stimulus_triggers = {
                'modality': {
                    'audio': 4,
                    'visual': 8
                },
                'oddball': 1
            }

    This will automatically result in mne-python aware event_ids like:

    ``'attention:visual/modality:audio/oddball:True'``

    """

    condition_triggers = None
    stimulus_triggers = None

    def __init__(self, *args, **kwargs):

        if not isinstance(self.stimulus_triggers, dict):
            raise ValueError('Please set stimulus_triggers to a dictionary')

        super(AutomaticBinaryEvents, self).__init__(*args, **kwargs)

    def _decode_bin_trigger(self, trigger_value, trigger_dict_item):
        if isinstance(trigger_dict_item, dict):
            for key, value in trigger_dict_item.items():
                if trigger_value & value:
                    return key
        else:
            if trigger_value & trigger_dict_item:
                return 'yes'
            else:
                return 'no'

    def _process_events(self):
        super(AutomaticBinaryEvents, self)._process_events()

        conditions_string_list = list()
        conditions_metadata = {}
        all_raw_metadata = []

        if self.condition_triggers:
            condition_trigger = self._events[0, 2]
            self._events = np.delete(self._events, (0), axis=0)

            for allcond_key, allcond_value in self.condition_triggers.items():
                decoded_trigger = self._decode_bin_trigger(condition_trigger,
                                                           allcond_value)
                conditions_string_list.append(
                    '%s:%s' % (allcond_key, decoded_trigger))
                conditions_metadata[allcond_key] = decoded_trigger

        all_event_codes = np.unique(self._events[:, 2])
        original_events = self._events.copy()

        for cur_code in all_event_codes:
            cur_string_list = list(conditions_string_list)

            for stim_group_name, stim_group_choices in \
                    self.stimulus_triggers.items():
                cur_string_list.append('%s:%s' % (stim_group_name,
                                                  self._decode_bin_trigger(
                                                      cur_code,
                                                      stim_group_choices)))

            new_event_key = '/'.join(cur_string_list)
            new_event_code = np.mod(np.abs(hash(new_event_key)), 4000)
            self._event_id[new_event_key] = new_event_code
            self._events[self._events[:, 2] == cur_code, 2] = new_event_code

        for cur_evt_code in original_events[:, 2]:
            cur_metadata = deepcopy(conditions_metadata)
            for stim_group_name, stim_group_choices in \
                    self.stimulus_triggers.items():
                cur_metadata[stim_group_name] = self._decode_bin_trigger(
                    cur_evt_code, stim_group_choices
                )

            all_raw_metadata.append(cur_metadata)

        self._evt_metadata = pd.DataFrame(all_raw_metadata)


class AutomaticBinaryEventsWithMetadata(AutomaticBinaryEvents):
    """Mixin for binary events with metadata.

    """
    def _process_events(self):
        super(AutomaticBinaryEvents, self)._process_events()

        conditions_metadata = {}
        all_raw_metadata = []

        if self.condition_triggers:
            condition_trigger = self._events[0, 2]
            self._events = np.delete(self._events, (0), axis=0)

            for allcond_key, allcond_value in self.condition_triggers.items():
                conditions_metadata[allcond_key] = self._decode_bin_trigger(
                    condition_trigger,
                    allcond_value
                )

        for cur_evt_code in self._events[:, 2]:
            cur_metadata = deepcopy(conditions_metadata)
            for stim_group_name, stim_group_choices in \
                    self.stimulus_triggers.items():
                cur_metadata[stim_group_name] = self._decode_bin_trigger(
                    cur_evt_code, stim_group_choices
                )

            all_raw_metadata.append(cur_metadata)

        self._evt_metadata = pd.DataFrame(all_raw_metadata)
