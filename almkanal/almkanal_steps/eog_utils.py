
import mne
from attrs import define, field

from almkanal.almkanal import AlmKanalStep
from almkanal.eye_utils.eog_utils import clean_eog
from almkanal.info import AlmKanalInfo

@define
class EOGCleaner(AlmKanalStep):
    """Detect, correct and interpolate blinks in continuous EOG data.

    Optionally assigns the two EOG channels to vertical and horizontal
    positions based on blink amplitudes and corrects vertical channel
    polarity. Blink detection uses a z-score threshold, with
    blink_rel_height controlling the detected peak widths.

    Original channel names are restored after processing. Optional
    filtering can be applied using l_freq and h_freq.

    File discovery, block iteration, resampling and epoching are
    handled by the surrounding pipeline.
    """

    eog_channels: tuple[str, str] = ('EOG001', 'EOG002')

    find_blinks: bool = True
    interpolate_blinks: bool = True

    peak_threshold: float = 2.0
    blink_rel_height: float = 0.8

    check_and_correct_eog_polarity: bool = True
    check_and_correct_eog_channel_assignment: bool = True

    flat_threshold_v: float = 1e-8
    max_amplitude_v: float | None = None
    max_bad_fraction: float = 0.8

    l_freq: float | None = 0.1
    h_freq: float | None = 40.0

    must_be_before: tuple = (
        'Resample',
        'EpochTRF',
        'ForwardModel',
        'SpatialFilter',
        'SourceReconstruction',
    )
    must_be_after: tuple = ()

    allow_repeated: bool = field(default=False, init=False)

    def run(
        self,
        data: mne.io.BaseRaw,
        info: AlmKanalInfo,
    ) -> dict:

        print('Starting EOGCleaner')

        result = clean_eog(
            data,
            eog_channels=self.eog_channels,
            find_blinks=self.find_blinks,
            interpolate_blinks=self.interpolate_blinks,
            peak_threshold=self.peak_threshold,
            blink_rel_height=self.blink_rel_height,
            check_and_correct_eog_polarity=self.check_and_correct_eog_polarity,
            check_and_correct_eog_channel_assignment=self.check_and_correct_eog_channel_assignment,
            flat_threshold_v=self.flat_threshold_v,
            max_amplitude_v=self.max_amplitude_v,
            max_bad_fraction=self.max_bad_fraction,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
        )

        print('Finished EOGCleaner')

        return result

    def reports(
        self,
        data: mne.io.BaseRaw,
        report: mne.Report,
        info: AlmKanalInfo,
    ) -> None:
        pass
