import mne
from attrs import define, field

from almkanal.almkanal import AlmKanalStep
from almkanal.eye_utils.eog_utils import clean_eog
from almkanal.info import AlmKanalInfo


@define
class EOGCleaner(AlmKanalStep):
    """Clean EOG using the settings described in clean_eog().

    Process one Raw at a time. Keep file discovery and block loops in the
    analysis pipeline. Automatic assignment/polarity are experimental and
    disabled by default. This step does not yet add report content. #TODO: add reporting
    """

    eog_channels: tuple[str, str] = ('EOG001', 'EOG002')
    find_blinks: bool = True
    interpolate_blinks: bool = True
    peak_threshold: float = 3.0
    blink_window_ms: float = 300.0
    check_and_correct_eog_polarity: bool = True
    check_and_correct_eog_channel_assignment: bool = True
    flat_threshold_v: float = 1e-8
    max_amplitude_v: float | None = None
    max_bad_fraction: float = 0.5
    l_freq: float | None = 0.1
    h_freq: float | None = 40.0

    must_be_before: tuple = ('ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()
    allow_repeated: bool = field(default=False, init=False)

    def run(self, data: mne.io.BaseRaw, info: AlmKanalInfo) -> dict:
        result = clean_eog(
            data,
            eog_channels=self.eog_channels,
            find_blinks=self.find_blinks,
            interpolate_blinks=self.interpolate_blinks,
            peak_threshold=self.peak_threshold,
            blink_window_ms=self.blink_window_ms,
            check_and_correct_eog_polarity=self.check_and_correct_eog_polarity,
            check_and_correct_eog_channel_assignment=self.check_and_correct_eog_channel_assignment,
            flat_threshold_v=self.flat_threshold_v,
            max_amplitude_v=self.max_amplitude_v,
            max_bad_fraction=self.max_bad_fraction,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
        )
        # report = mne.Report(title="EOG Cleaning Report")

        return result  # , report

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo) -> None:
        pass
