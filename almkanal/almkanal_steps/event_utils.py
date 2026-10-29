import mne
from attrs import define, field

from almkanal import AlmKanalStep
from almkanal.defaults import default_field, step_with_defaults
from almkanal.info import AlmKanalInfo


@define
class Events(AlmKanalStep):
    """Extract discrete events from a continuous raw recording.

    ``Events`` detects events from one or more stimulus channels using
    :func:`mne.find_events`. The detected event array is returned unchanged
    alongside the original raw object and is stored in the processing history
    for use by later steps such as ``Epochs``.

    Parameters
    ----------
    stim_channel : str | None, default=None
        Name of the stimulus channel used for event detection. If ``None``,
        MNE's default stimulus-channel selection is used.
    output : str, default='onset'
        Event representation returned by :func:`mne.find_events`, for example
        ``'onset'``, ``'offset'``, or ``'step'``.
    consecutive : bool | str, default='increasing'
        How consecutive non-zero values on the stimulus channel are treated.
        Passed directly to :func:`mne.find_events`.
    min_duration : float, default=0.0
        Minimum event duration, in seconds.
    shortest_event : int, default=2
        Minimum number of samples that an event must span.
    mask : int | None, default=None
        Optional bit mask applied to the stimulus channel during event
        detection.
    uint_cast : bool, default=False
        Whether the stimulus channel should be cast to an unsigned integer
        representation before event detection.
    mask_type : str, default='and'
        Masking operation applied when ``mask`` is provided.
    initial_event : bool, default=False
        Whether a non-zero stimulus value at the beginning of the recording
        should be emitted as an event.
    verbose : bool | str | int | None, default=None
        Verbosity setting passed to :func:`mne.find_events`.

    Notes
    -----
    Event detection does not modify the input raw data.

    The processing metadata returned by :meth:`run` is stored under
    ``'event_info'`` and contains both the detected ``events`` array and the
    effective event-detection settings.

    The resulting event array has the standard MNE shape ``(n_events, 3)`` and
    can be consumed by later processing steps such as ``Epochs``.

    ``Events`` can occur only once in a pipeline.

    See Also
    --------
    mne.find_events
        Detect events from stimulus channels.
    Epochs
        Create epochs from explicitly supplied or previously detected events.
    """

    stim_channel: str | list[str] | None = default_field('events', 'stim_channel')
    output: str = default_field('events', 'output')
    consecutive: bool | str = default_field('events', 'consecutive')
    min_duration: float = default_field('events', 'min_duration')
    shortest_event: int = default_field('events', 'shortest_event')
    mask: int | None = default_field('events', 'mask')
    uint_cast: bool = default_field('events', 'uint_cast')
    mask_type: str = default_field('events', 'mask_type')
    initial_event: bool = default_field('events', 'initial_event')
    verbose: bool | str | int | None = default_field('events', 'verbose')

    must_be_before: tuple = ('Epochs', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()
    allow_repeated: bool = field(default=False, init=False)

    @step_with_defaults
    def run(
        self,
        data: mne.io.BaseRaw,
        info: AlmKanalInfo,
    ) -> dict:
        # this should build events based on information stored in the raw file
        events = mne.find_events(
            data,
            stim_channel=self.stim_channel,
            output=self.output,
            consecutive=self.consecutive,
            min_duration=self.min_duration,
            shortest_event=self.shortest_event,
            mask=self.mask,
            uint_cast=self.uint_cast,
            mask_type=self.mask_type,
            initial_event=self.initial_event,
            verbose=self.verbose,
        )

        return {
            'data': data,
            'event_info': {
                'events': events,
                'stim_channel': self.stim_channel,
                'output': self.output,
                'consecutive': self.consecutive,
                'min_duration': self.min_duration,
                'shortest_event': self.shortest_event,
                'mask': self.mask,
                'uint_cast': self.uint_cast,
                'mask_type': self.mask_type,
                'initial_event': self.initial_event,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: AlmKanalInfo) -> None:
        events = info.get_step_info('Events', occurrence=-1, required=True)['event_info']['events']
        report.add_events(events=events, sfreq=data.info['sfreq'], title='events')
