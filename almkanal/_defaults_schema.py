"""Typed schemas for defaults profiles; scientific validation remains with each step."""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TCH003 - used by get_type_hints at runtime
from typing import Any, Literal

from attrs import field, frozen


@frozen
class FilterDefaults:
    highpass: float | None = 0.1
    lowpass: float | None = 40.0
    picks: Any = None
    filter_length: str | int = 'auto'
    l_trans_bandwidth: float | Literal['auto'] = 'auto'
    h_trans_bandwidth: float | Literal['auto'] = 'auto'
    n_jobs: int | str | None = None
    method: str = 'fir'
    iir_params: dict[str, Any] | None = None
    phase: str = 'zero'
    fir_window: str = 'hamming'
    fir_design: str = 'firwin'
    skip_by_annotation: str | tuple[str, ...] | list[str] = ('edge', 'bad_acq_skip')
    pad: str = 'reflect_limited'


@frozen
class ResampleDefaults:
    npad: str = 'auto'
    window: str | tuple[str, float] = 'auto'
    n_jobs: int | None = None
    pad: str = 'auto'
    method: Literal['fft', 'polyphase'] = 'fft'


@frozen
class ICADefaults:
    fit_only: bool = False
    n_components: None | int | float = None
    method: str = 'picard'
    random_state: None | int = 42
    fit_params: dict | None = None
    ica_hp_freq: None | float = 1.0
    ica_lp_freq: None | float = None
    resample_freq: int | None = 200
    eog: bool = True
    surrogate_eog_chs: None | dict = None
    eog_corr_thresh: float = 0.5
    ecg: bool = True
    ecg_corr_thresh: float = 0.5
    emg: bool = False
    emg_thresh: float = 0.5
    train: bool = False
    train_freq: float = 16.666
    train_thresh: float = 6.0
    img_path: None | str = None
    fname: None | str = None


@frozen
class EventsDefaults:
    stim_channel: str | list[str] | None = None
    output: str = 'onset'
    consecutive: bool | str = 'increasing'
    min_duration: float = 0.0
    shortest_event: int = 2
    mask: int | None = None
    uint_cast: bool = False
    mask_type: str = 'and'
    initial_event: bool = False
    verbose: bool | str | int | None = None


@frozen
class PhysioCleanerDefaults:
    ecg: None | str | list = None
    resp: None | str | list = None
    eog: None | str | list = None
    emg: None | str | list = None


@frozen
class MaxwellDefaults:
    mw_coord_frame: str = 'head'
    mw_destination: Any = None
    mw_calibration_file: str | bool | None = None
    mw_cross_talk_file: str | bool | None = None
    mw_st_duration: float | None = None
    mw_st_correlation: float = 0.98


@frozen
class EEGRANSACDefaults:
    ransac_epoch_duration: int | float = 4
    n_resample: int = 50
    min_channels: float = 0.25
    min_corr: float = 0.75
    unbroken_time: float = 0.4
    n_jobs: int = 1
    verbose: bool = False
    random_state: int | None = 435656


@frozen
class ReReferenceDefaults:
    ref_channels: str | list[str] | dict = 'average'
    projection: bool = False
    ch_type: str | list[str] = 'auto'
    forward: Any = None
    joint: bool = False
    verbose: bool | str | int | None = False


@frozen
class EpochsDefaults:
    tmin: float = -0.15
    tmax: float = 0.5
    events: Any = None
    event_id: None | dict = None
    baseline: None | tuple = None
    preload: bool = True
    picks: Any = None
    reject: dict | None = None
    flat: dict | None = None
    proj: bool | str = True
    decim: int = 1
    reject_tmin: float | None = None
    reject_tmax: float | None = None
    detrend: int | None = None
    on_missing: str = 'raise'
    reject_by_annotation: bool = True
    metadata: Any = None
    event_repeated: str = 'error'
    verbose: bool | str | int | None = None


@frozen
class ForwardModelDefaults:
    pick_dict: dict | None = None
    source: str = 'surface'
    redo_hdm: bool = True
    spacing: str = 'oct6'
    source_ico: int = 4
    bem_conductivity: float | Sequence[float] = (0.3,)
    volume_pos: float = 5.0
    use_template_mri: bool = True
    min_dist_src: float = 5.0
    meg: bool = True
    eeg: bool = False
    redo_bem: bool = False


@frozen
class SpatialFilterDefaults:
    fwd: Any = None
    pick_dict: dict | None = None
    data_cov: Any = None
    noise_cov: Any = None
    empty_room: Any = None
    nearest_empty_room: bool = False
    chans2keep: list[str] | None = None
    lcmv_reg: float = 0.05
    lcmv_pick_ori: str | None = 'max-power'
    lcmv_weight_norm: str | None = 'nai'
    lcmv_reduce_rank: bool = False


@frozen
class SourceReconstructionDefaults:
    filters: Any = None
    return_parc: bool = False
    label_mode: str = 'pca_flip'
    atlas: str = 'glasser'
    morph2fsaverage: bool = True


@frozen
class EpochTRFDefaults:
    feature: str = 'envelope'
    audio_cutoff_hz: float = 80.0
    hw_delay_s: float = 0.0
    epoch_len_s: float = 5.0
    audio_channels: Sequence[str] | None = None
    alignment_kwargs: Mapping[str, Any] | None = None
    preserve_annotations: bool = True
    on_alignment_error: Literal['raise', 'skip'] = 'raise'
    verbose: bool = True
    realign_without_audio: bool = False
    fallback_drift_us_per_s: float = 0.0


@frozen
class PipelineDefaults:
    pick_params: dict | None = None


@frozen
class EyeDefaults:
    gaze_lims: dict = field(factory=lambda: {'x': 6, 'y': 6})
    filter_settings: dict = field(factory=lambda: {'pupil_diameter': (None, 30), 'xy_movements': (0.1, 40)})
    annotate_bads: bool = True
    trigger_ch_name: str = 'stim'
    tpixx_fs: int | None = None
    distance: float | None = None
    screen_width: float | None = None
    screen_rect: list[int] | tuple[int, ...] | None = None
    verbose: bool = False


@frozen
class ICAEOGDefaults:
    left_eog_chs: list[str] | None = None
    right_eog_chs: list[str] | None = None
    threshold: float = 0.5
    tol: float = 0.2


@frozen
class ICATrainDefaults:
    duration: int = 8
    overlap: float = 0.5
    hmax: float = 2.0


@frozen
class TrialDefaults:
    end_triggers: int | Sequence[int] | None = None
    infer_missing_ends: bool = False
    base_audio_path: Any = None


@frozen
class AlignmentDefaults:
    sync_sfreq: float = 500.0
    audio_band: tuple[float, float] = (80.0, 400.0)
    envelope_lowpass: float = 30.0
    window_s: float = 10.0
    step_s: float = 5.0
    max_lag_s: float = 0.250
    min_corr: float = 0.30
    min_anchors: int = 5
    reject_outliers: bool = True
    warn_residual_rms_ms: float = 10.0
    verbose: bool = True


@frozen
class AudioDefaults:
    feature: str = 'envelope'
    target_fs: float = 100.0
    cutoff_hz: float = 80.0
    n_mels: int = 32
    fmin: float = 50.0
    fmax: float = 8000.0


SECTION_TYPES = {
    'filter': FilterDefaults,
    'resample': ResampleDefaults,
    'ica': ICADefaults,
    'events': EventsDefaults,
    'physio': PhysioCleanerDefaults,
    'maxwell': MaxwellDefaults,
    'ransac': EEGRANSACDefaults,
    'rereference': ReReferenceDefaults,
    'epochs': EpochsDefaults,
    'forward_model': ForwardModelDefaults,
    'spatial_filter': SpatialFilterDefaults,
    'source_reconstruction': SourceReconstructionDefaults,
    'trf': EpochTRFDefaults,
    'pipeline': PipelineDefaults,
    'eye': EyeDefaults,
    'ica_eog': ICAEOGDefaults,
    'ica_train': ICATrainDefaults,
    'trials': TrialDefaults,
    'alignment': AlignmentDefaults,
    'audio': AudioDefaults,
}
