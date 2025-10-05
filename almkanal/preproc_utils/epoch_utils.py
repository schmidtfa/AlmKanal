import mne
import pandas as pd
from attrs import define
from numpy.typing import ArrayLike

from almkanal import AlmKanalStep


@define
class Epochs(AlmKanalStep):
    must_be_before: tuple = ('ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = (
        'Maxwell',
        'ICA',
    )

    tmin: float = -0.15
    tmax: float = 0.5
    events: None | ArrayLike = None
    event_id: None | dict = None
    baseline: None | tuple = None
    preload: bool = True
    picks: str | ArrayLike | None = None
    reject: dict | None = None
    flat: dict | None = None
    proj: bool | str = True
    decim: int = 1
    reject_tmin: float | None = None
    reject_tmax: float | None = None
    detrend: int | None = None
    on_missing: str = 'raise'
    reject_by_annotation: bool = True
    metadata: None | pd.DataFrame = None
    event_repeated: str = 'error'
    verbose: bool | str | int | None = None

    """
    Create epochs from raw MEG data and events.

    Parameters
    ----------
    tmin : float
        Start time before the event (in seconds).
    tmax : float
        End time after the event (in seconds).
    event_id : dict | None, optional
        Dictionary mapping event IDs to event labels. Defaults to None.
    baseline : tuple | None, optional
        Baseline correction period. Defaults to None.
    preload : bool, optional
        Whether to preload the data into memory. Defaults to True.
    picks : str | ArrayLike | None, optional
        Channels to include in the epochs. Defaults to None.
    reject : dict | None, optional
        Rejection criteria. Defaults to None.
    flat : dict | None, optional
        Flatness criteria. Defaults to None.
    proj : bool | str, optional
        Whether to apply projection. Defaults to True.
    decim : int, optional
        Decimation factor for downsampling. Defaults to 1.
    reject_tmin : float | None, optional
        Start of rejection window. Defaults to None.
    reject_tmax : float | None, optional
        End of rejection window. Defaults to None.
    detrend : int | None, optional
        Detrending mode. Defaults to None.
    on_missing : str, optional
        Behavior for missing events ('raise', etc.). Defaults to 'raise'.
    reject_by_annotation : bool, optional
        Whether to reject epochs based on annotations. Defaults to True.
    metadata : pd.DataFrame | None, optional
        Additional metadata for epochs. Defaults to None.
    event_repeated : str, optional
        Behavior for repeated events ('error', etc.). Defaults to 'error'.
    verbose : bool | str | int | None, optional
        Verbosity level. Defaults to None.

    Returns
    -------
    None
    """

    def run(
        self,
        data: mne.io.BaseRaw,
        info: dict,
    ) -> dict:
        if self.events is None:
            self.events = info['Events']['event_info']['events']
        else:
            raise ValueError(
                'You need to either supply `events` to epochs or select them in a previous '
                'step in the pipeline to create epochs'
            )

        epochs = mne.Epochs(
            data,
            events=self.events,
            event_id=self.event_id,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            picks=self.picks,
            preload=self.preload,
            reject=self.reject,
            flat=self.flat,
            proj=self.proj,
            decim=self.decim,
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
                'events': self.events,
                'event_id': self.event_id,
                'baseline': self.baseline,
                'tmin': self.tmin,
                'tmax': self.tmax,
                'picks': self.picks,
                'preload': self.preload,
                'reject': self.reject,
                'flat': self.flat,
                'proj': self.proj,
                'decim': self.decim,
                'reject_tmin': self.reject_tmin,
                'reject_tmax': self.reject_tmax,
                'detrend': self.detrend,
                'on_missing': self.on_missing,
                'reject_by_annotation': self.reject_by_annotation,
                'metadata': self.metadata,
                'event_repeated': self.event_repeated,
            },
        }

    def reports(self, data: mne.BaseEpochs, report: mne.Report, info: dict) -> None:
        base_corr = data.copy()
        base_corr.apply_baseline(baseline=(None, 0))
        # report.add_epochs(base_corr, psd=False, title='epochs')

        evokeds = base_corr.average(by_event_type=True)
        report.add_evokeds(evokeds, n_time_points=5)
