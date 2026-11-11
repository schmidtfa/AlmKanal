import mne
from attrs import define, field

from almkanal.almkanal import AlmKanalStep
from almkanal.info import AlmKanalInfo

from almkanal.eye_utils.eog_utils import (
    clean_eog,
    DEFAULT_EOG_CHANNELS,
    DEFAULT_FIND_BLINKS,
    DEFAULT_INTERPOLATE_BLINKS,
    DEFAULT_PEAK_THRESHOLD,
    DEFAULT_BLINK_REL_HEIGHT,
    DEFAULT_CHECK_POLARITY,
    DEFAULT_CHECK_CHANNEL_ASSIGNMENT,
    DEFAULT_FLAT_THRESHOLD_V,
    DEFAULT_MAX_BAD_FRACTION,
    DEFAULT_L_FREQ,
    DEFAULT_H_FREQ,
)



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
    eog_channels: tuple[str, str] = DEFAULT_EOG_CHANNELS

    find_blinks: bool = DEFAULT_FIND_BLINKS
    interpolate_blinks: bool = DEFAULT_INTERPOLATE_BLINKS

    peak_threshold: float = DEFAULT_PEAK_THRESHOLD
    blink_rel_height: float = DEFAULT_BLINK_REL_HEIGHT

    check_and_correct_eog_polarity: bool = DEFAULT_CHECK_POLARITY
    check_and_correct_eog_channel_assignment: bool = DEFAULT_CHECK_CHANNEL_ASSIGNMENT

    flat_threshold_v: float = DEFAULT_FLAT_THRESHOLD_V
    max_bad_fraction: float = DEFAULT_MAX_BAD_FRACTION

    l_freq: float | None = DEFAULT_L_FREQ
    h_freq: float | None = DEFAULT_H_FREQ

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
