import mne
import pandas as pd
from attrs import define, field
from numpy.typing import ArrayLike

from almkanal import AlmKanalStep
from almkanal.defaults import default_field, step_with_defaults
from almkanal.info import AlmKanalInfo


@define
class Epochs(AlmKanalStep):
    """Create epochs from continuous M/EEG data.

    ``Epochs`` segments a continuous :class:`mne.io.BaseRaw` recording into
    time-locked trials using :class:`mne.Epochs`.

    Events can either be supplied directly through ``events`` or obtained from the
    most recent preceding ``Events`` step in the processing history. Explicitly
    supplied events take precedence.

    Parameters
    ----------
    tmin : float, default=-0.15
        Start of each epoch relative to the event, in seconds.
    tmax : float, default=0.5
        End of each epoch relative to the event, in seconds.
    events : array-like | None, default=None
        Event array used to construct the epochs. Expected to follow MNE's standard
        ``(n_events, 3)`` event representation. If ``None``, events are retrieved
        from the most recent preceding ``Events`` step.
    event_id : dict | None, default=None
        Mapping from event names to integer event codes. Passed to
        :class:`mne.Epochs`.
    baseline : tuple | None, default=None
        Baseline correction interval in seconds. ``None`` disables baseline
        correction.
    preload : bool, default=True
        Whether to preload epoched data into memory.
    picks : str | array-like | None, default=None
        Channels to include in the epochs.
    reject : dict | None, default=None
        Peak-to-peak amplitude rejection thresholds by channel type.
    flat : dict | None, default=None
        Minimum acceptable peak-to-peak amplitudes by channel type.
    proj : bool | str, default=True
        Projection handling passed to :class:`mne.Epochs`.
    reject_tmin : float | None, default=None
        Start of the time interval used for rejection, in seconds relative to the
        event.
    reject_tmax : float | None, default=None
        End of the time interval used for rejection, in seconds relative to the
        event.
    detrend : int | None, default=None
        Detrending order. ``0`` removes the mean, ``1`` removes a linear trend, and
        ``None`` disables detrending.
    on_missing : str, default='raise'
        Behaviour when entries in ``event_id`` are not present in the event array.
        Passed directly to :class:`mne.Epochs`.
    reject_by_annotation : bool, default=True
        Whether epochs overlapping annotations marked as bad should be rejected.
    metadata : pandas.DataFrame | None, default=None
        Metadata associated with individual epochs.
    event_repeated : str, default='error'
        Behaviour when multiple events occur at the same sample. Passed directly to
        :class:`mne.Epochs`.
    verbose : bool | str | int | None, default=None
        Verbosity setting passed to :class:`mne.Epochs`.

    Notes
    -----
    If ``events`` is provided directly, no preceding ``Events`` step is required.
    Otherwise, the event array is obtained from the most recent ``Events`` entry in
    the processing history.

    The input raw object is not replaced by this step. :meth:`run` returns a new
    :class:`mne.Epochs` object under ``'data'``.

    The processing metadata returned by :meth:`run` is stored under
    ``'epochs_info'`` and contains the event array together with the effective epoch
    construction, baseline, channel-selection, rejection, decimation, detrending,
    and metadata settings.

    ``Epochs`` can occur only once in a pipeline and is expected to run after
    artifact-correction steps such as ``Maxwell`` and ``ICA`` and before source
    modelling and reconstruction steps.

    See Also
    --------
    mne.Epochs
        MNE-Python class used to construct the epoched data.
    Events
        Detect events from a continuous recording for use by this step.
    """

    must_be_before: tuple = ('ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = (
        'Maxwell',
        'ICA',
    )
    allow_repeated: bool = field(default=False, init=False)

    tmin: float = default_field('epochs', 'tmin')
    tmax: float = default_field('epochs', 'tmax')
    events: None | ArrayLike = default_field('epochs', 'events')
    event_id: None | dict = default_field('epochs', 'event_id')
    baseline: None | tuple = default_field('epochs', 'baseline')
    preload: bool = default_field('epochs', 'preload')
    picks: str | ArrayLike | None = default_field('epochs', 'picks')
    reject: dict | None = default_field('epochs', 'reject')
    flat: dict | None = default_field('epochs', 'flat')
    proj: bool | str = default_field('epochs', 'proj')
    reject_tmin: float | None = default_field('epochs', 'reject_tmin')
    reject_tmax: float | None = default_field('epochs', 'reject_tmax')
    detrend: int | None = default_field('epochs', 'detrend')
    on_missing: str = default_field('epochs', 'on_missing')
    reject_by_annotation: bool = default_field('epochs', 'reject_by_annotation')
    metadata: None | pd.DataFrame = default_field('epochs', 'metadata')
    event_repeated: str = default_field('epochs', 'event_repeated')
    verbose: bool | str | int | None = default_field('epochs', 'verbose')

    @step_with_defaults
    def run(
        self,
        data: mne.io.BaseRaw,
        info: AlmKanalInfo,
    ) -> dict:
        events = self.events

        if events is None:
            event_info = info.get_step_info('Events', occurrence=-1)

            if event_info is None:
                raise ValueError(
                    'You need to either supply `events` to epochs or select them in a previous '
                    'step in the pipeline to create epochs'
                )

            events = event_info['event_info']['events']

        epochs = mne.Epochs(
            data,
            events=events,
            event_id=self.event_id,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            picks=self.picks,
            preload=self.preload,
            reject=self.reject,
            flat=self.flat,
            proj=self.proj,
            reject_tmin=self.reject_tmin,
            reject_tmax=self.reject_tmax,
            detrend=self.detrend,
            on_missing=self.on_missing,
            reject_by_annotation=self.reject_by_annotation,
            metadata=self.metadata,
            event_repeated=self.event_repeated,
            verbose=self.verbose,
        )

        return {
            'data': epochs,
            'epochs_info': {
                'events': events,
                'event_id': self.event_id,
                'baseline': self.baseline,
                'tmin': self.tmin,
                'tmax': self.tmax,
                'picks': self.picks,
                'preload': self.preload,
                'reject': self.reject,
                'flat': self.flat,
                'proj': self.proj,
                'reject_tmin': self.reject_tmin,
                'reject_tmax': self.reject_tmax,
                'detrend': self.detrend,
                'on_missing': self.on_missing,
                'reject_by_annotation': self.reject_by_annotation,
                'metadata': self.metadata,
                'event_repeated': self.event_repeated,
            },
        }

    def reports(self, data: mne.BaseEpochs, report: mne.Report, info: AlmKanalInfo) -> None:
        base_corr = data.copy()

        evokeds = base_corr.average(by_event_type=True)
        report.add_evokeds(evokeds, n_time_points=5)
