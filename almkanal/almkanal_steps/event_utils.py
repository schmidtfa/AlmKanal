import mne
from attrs import define

from almkanal import AlmKanalStep


@define
class Events(AlmKanalStep):
    stim_channel: None | str = None
    output: str = 'onset'
    consecutive: bool | str = 'increasing'
    min_duration: float = 0.0
    shortest_event: int = 2
    mask: int | None = None
    uint_cast: bool = False
    mask_type: str = 'and'
    initial_event: bool = False
    verbose: bool | str | int | None = None

    def run(
        self,
        data: mne.io.BaseRaw,
        info: dict,
    ) -> dict:
        """
        Extract events from the raw MEG data.

        Parameters
        ----------
        stim_channel : str | None, optional
            Name of the stimulus channel. Defaults to None.
        output : str, optional
            Type of output ('onset', etc.). Defaults to 'onset'.
        consecutive : bool | str, optional
            Whether to consider consecutive events ('increasing', etc.). Defaults to 'increasing'.
        min_duration : float, optional
            Minimum duration of events. Defaults to 0.0.
        shortest_event : int, optional
            Minimum number of samples for an event. Defaults to 2.
        mask : int | None, optional
            Binary mask for event detection. Defaults to None.
        uint_cast : bool, optional
            Whether to cast to unsigned integer. Defaults to False.
        mask_type : str, optional
            Type of masking ('and', etc.). Defaults to 'and'.
        initial_event : bool, optional
            Whether to include initial events. Defaults to False.
        verbose : bool | str | int | None, optional
            Verbosity level. Defaults to None.

        Returns
        -------
        None
        """

        # this should build events based on information stored in the raw file
        events = mne.find_events(
            data,
            stim_channel=self.stim_channel,  #'STI101',
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

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        events = info['Events']['event_info']['events']
        report.add_events(events=events, sfreq=data.info['sfreq'], title='events')
