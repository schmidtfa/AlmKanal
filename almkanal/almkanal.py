from pathlib import Path

import attrs
import mne
import numpy as np
import pandas as pd
from attrs import define, field
from numpy.typing import ArrayLike, NDArray

from almkanal.data_utils.check_data import check_raw_epoch
from almkanal.data_utils.data_classes import ICAInfoDict, InfoClass, PickDictClass
from almkanal.preproc_utils.bio_utils import run_bio_preproc
from almkanal.preproc_utils.ica_utils import run_ica

# all them utility functions
from almkanal.preproc_utils.maxwell_utils import run_maxwell
from almkanal.src_utils.headmodel_utils import compute_headmodel, make_fwd
from almkanal.src_utils.src_utils import comp_spatial_filters, src2parc


@define
class AlmKanal:
    raw: None | mne.io.Raw = None
    epoched: None | mne.Epochs = None
    pick_dict: PickDictClass = PickDictClass(meg=True, eog=True, ecg=True, eeg=False, stim=True)
    events: None | np.ndarray = None
    fwd: None | mne.Forward = None
    ica: None | mne.preprocessing.ICA = None
    ica_ids: None | list = None
    filters: None | mne.beamformer.Beamformer = None
    stim: None | np.ndarray = None  # This is for TRF stuff
    info: InfoClass = field(default=attrs.Factory(InfoClass))
    _initialized: bool = field(default=False, init=False)  # Internal flag for initialization
    _data_check_disabled: bool = field(default=False, init=False)  # Disable data check temporarily

    """
    AlmKanal: A high-level interface for MEG data processing and analysis.

    This class can be used as a pipeline builder for preprocessing MEG data leveraging MNE-Python's functionality.

    Attributes
    ----------
    raw : mne.io.Raw | None
        The raw MEG data.
    epoched : mne.Epochs | None
        The epoched MEG data.
    pick_dict : PickDictClass
        Dictionary specifying channel selection criteria.
    events : np.ndarray | None
        Event markers extracted from the raw data.
    fwd : mne.Forward | None
        The forward model for source reconstruction.
    ica : mne.preprocessing.ICA | None
        ICA object(s) for artifact removal.
    ica_ids : list | None
        Indices of ICA components identified as artifacts.
    filters : mne.beamformer.Beamformer | None
        Spatial filters for source reconstruction.
    stim : np.ndarray | None
        TRF-related stimulus information.
    info : InfoClass
        Configuration and preprocessing metadata.
    """

    # TODO: Use attrs validators -> checks whenver a field is set
    # Check that we only have either raw or epoched data when we initalize the method
    def __attrs_post_init__(self) -> None:
        check_raw_epoch(self)
        self._initialized = True

    # Check that we only have either raw or epoched data whenever we add a raw or epoched attribute
    def __setattr__(self, name: str, value: bool) -> None:
        # Call the special method before returning any other method
        super().__setattr__(name, value)

        if getattr(self, '_initialized', False) and name in {'raw', 'epoched'}:
            check_raw_epoch(self)

    def do_maxwell(
        self,
        mw_coord_frame: str = 'head',
        mw_destination: str | None = None,
        mw_calibration_file: str | None = None,
        mw_cross_talk_file: str | None = None,
        mw_st_duration: int | None = None,
    ) -> None:
        """
        Apply Maxwell filtering to the raw MEG data.

        Parameters
        ----------
        mw_coord_frame : str, optional
            Coordinate frame for Maxwell filtering ('head' or 'meg'). Defaults to 'head'.
        mw_destination : str | None, optional
            Destination coordinate frame for alignment. Defaults to None.
        mw_calibration_file : str | None, optional
            Path to the calibration file. Defaults to None.
        mw_cross_talk_file : str | None, optional
            Path to the cross-talk file. Defaults to None.
        mw_st_duration : int | None, optional
            Duration (in seconds) for tSSS (temporal Signal Space Separation). Defaults to None.

        Returns
        -------
        None
        """

        # this should do maxwell filtering
        # should only be possible on raw data and only if no other preprocessing apart from filtering was done
        if self.info.maxwell is None:
            self.info.maxwell = {
                'coord_frame': mw_coord_frame,
                'destination': mw_destination,
                'calibration_file': mw_calibration_file,
                'cross_talk_file': mw_cross_talk_file,
                'st_duration': mw_st_duration,
            }

            self.raw = run_maxwell(raw=self.raw, **self.info.maxwell)
        else:
            print('You already maxfiltered your data')

    def do_ica(
        self,
        n_components: None | int | float = None,
        method: str = 'picard',
        random_state: None | int = 42,
        fit_params: dict | None = None,
        ica_hp_freq: None | float = 1.0,
        ica_lp_freq: None | float = None,
        resample_freq: int = 200,  # downsample to 200hz per default
        eog: bool = True,
        surrogate_eog_chs: None | dict = None,
        eog_corr_thresh: float = 0.5,
        ecg: bool = True,
        ecg_corr_thresh: float = 0.5,
        emg: bool = False,
        emg_thresh: float = 0.5,
        train: bool = True,
        train_freq: int = 16,
        train_thresh: float = 2.0,
        img_path: None | str = None,
        fname: None | str = None,
    ) -> None:
        """
        Perform ICA to identify and remove peripheral physiological signals like
        EOG and ECG as well as an artifact caused by our local train in Salzburg.

        Parameters
        ----------
        n_components : int | float | None, optional
            Number of ICA components to compute. Defaults to None.
        method : str, optional
            ICA method to use ('picard', etc.). Defaults to 'picard'.
        random_state : int | None, optional
            Random seed for reproducibility. Defaults to 42.
        fit_params : dict | None, optional
            Additional fitting parameters for ICA. Defaults to None.
        ica_hp_freq : float | None, optional
            High-pass filter frequency for ICA preprocessing. Defaults to 1.0 Hz.
        ica_lp_freq : float | None, optional
            Low-pass filter frequency for ICA preprocessing. Defaults to None.
        resample_freq : int, optional
            Downsampling frequency before ICA. Defaults to 200 Hz.
        eog : bool, optional
            Whether to detect and remove EOG artifacts. Defaults to True.
        eog_corr_thresh : float, optional
            Correlation threshold for EOG artifact detection. Defaults to 0.5.
        ecg : bool, optional
            Whether to detect and remove ECG artifacts. Defaults to True.
        ecg_corr_thresh : float, optional
            Correlation threshold for ECG artifact detection. Defaults to 0.5.
        emg : bool,
            Whether to detect and remove EMG artifacts. Defaults to False.
        emg_thresh:
            Value above which a component should be marked as muscle-related, relative to a typical muscle component.
        train : bool, optional
            Whether to detect and remove train-related artifacts. Defaults to True.
        train_freq : int, optional
            Frequency for train artifact detection. Defaults to 16 Hz.
        img_path : str | None, optional
            Path to save ICA plots. Defaults to None.
        fname : str | None, optional
            Filename for ICA plots. Defaults to None.

        Returns
        -------
        None
        """

        # this should do an ica
        ica_info = ICAInfoDict(
            n_components=n_components,
            method=method,
            random_state=random_state,
            fit_params=fit_params,
            ica_hp_freq=ica_hp_freq,
            ica_lp_freq=ica_lp_freq,
            resample_freq=resample_freq,
            eog=eog,
            eog_corr_thresh=eog_corr_thresh,
            surrogate_eog_chs=surrogate_eog_chs,
            ecg=ecg,
            ecg_corr_thresh=ecg_corr_thresh,
            emg=emg,
            emg_thresh=emg_thresh,
            train=train,
            train_freq=train_freq,
            train_thresh=train_thresh,
            img_path=img_path,
            fname=fname,
        )

        if self.info.ica is None:
            self.info.ica = [ica_info]

            self.raw, ica, ica_ids = run_ica(self.raw, **ica_info)
            self.ica = [ica]
            self.ica_ids = [ica_ids]
        elif self.info.ica is not None:
            assert isinstance(self.info.ica, list)
            # Take care of case where you ran multiple icas.
            # TODO: When we end up applying them to the noise cov dont forget
            # to also do it successively.
            self.info.ica.append(ica_info)

            self.raw, ica, ica_ids = run_ica(self.raw, **ica_info)
            assert isinstance(self.ica, list)
            self.ica.append(ica)
            assert isinstance(self.ica_ids, list)
            self.ica_ids.append(ica_ids)

    def do_bio_process(
        self,
        ecg: None | str | list = None,
        resp: None | str | list = None,
        eog: None | str | list = None,
        emg: None | str | list = None,
    ) -> mne.io.Raw:
        """
        Preprocess physiological signals (ECG, EOG, RESP, EMG) in an MNE raw object.

        This method extracts specified physiological channels, preprocesses them using `neurokit2`,
        and returns a new raw object containing the cleaned physiological data along with stimulus channels.

        Parameters
        ----------
        ecg : str | list | None, optional
            Name(s) of the ECG channel(s) to preprocess. Defaults to None.
        resp : str | list | None, optional
            Name(s) of the respiratory channel(s) to preprocess. Defaults to None.
        eog : str | list | None, optional
            Name(s) of the EOG channel(s) to preprocess. Defaults to None.
        emg : str | list | None, optional
            Name(s) of the EMG channel(s) to preprocess. Defaults to None.

        Returns
        -------
        mne.io.Raw
            A new raw object containing the preprocessed physiological signals and stimulus channels.
        """

        if self.raw is None:
            raise ValueError("""This method requires raw data.
                              It will return mne.io.Raw object containing your
                              trigger channels so you can epoch the data later""")

        return run_bio_preproc(
            raw=self.raw,
            ecg=ecg,
            resp=resp,
            eog=eog,
            emg=emg,
        )

    def do_events(
        self,
        stim_channel: None | str = None,
        output: str = 'onset',
        consecutive: bool | str = 'increasing',
        min_duration: float = 0.0,
        shortest_event: int = 2,
        mask: int | None = None,
        uint_cast: bool = False,
        mask_type: str = 'and',
        initial_event: bool = False,
        verbose: bool | str | int | None = None,
    ) -> None:
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
            self.raw,
            stim_channel=stim_channel,  #'STI101',
            output=output,
            consecutive=consecutive,
            min_duration=min_duration,
            shortest_event=shortest_event,
            mask=mask,
            uint_cast=uint_cast,
            mask_type=mask_type,
            initial_event=initial_event,
            verbose=verbose,
        )
        self.events = events

    def do_epochs(
        self,
        tmin: float,
        tmax: float,
        event_id: None | dict = None,
        baseline: None | tuple = None,
        preload: bool = True,
        picks: str | ArrayLike | None = None,
        reject: dict | None = None,
        flat: dict | None = None,
        proj: bool | str = True,
        decim: int = 1,
        reject_tmin: float | None = None,
        reject_tmax: float | None = None,
        detrend: int | None = None,
        on_missing: str = 'raise',
        reject_by_annotation: bool = True,
        metadata: None | pd.DataFrame = None,
        event_repeated: str = 'error',
        verbose: bool | str | int | None = None,
    ) -> None:
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

        # this should take raw and events and epoch the data
        if np.logical_and(self.raw is not None, self.events is not None):
            epoched = mne.Epochs(
                self.raw,
                events=self.events,
                event_id=event_id,
                baseline=baseline,
                tmin=tmin,
                tmax=tmax,
                picks=picks,
                preload=preload,
                reject=reject,
                flat=flat,
                proj=proj,
                decim=decim,
                reject_tmin=reject_tmin,
                reject_tmax=reject_tmax,
                detrend=detrend,
                on_missing=on_missing,
                reject_by_annotation=reject_by_annotation,
                metadata=metadata,
                event_repeated=event_repeated,
                verbose=verbose,
            )

            self._data_check_disabled = True
            self.raw = None
            self.epoched = epoched
            self._data_check_disabled = False
        else:
            print('You need both `raw` data and `events` in the pipeline to create epochs')

    def do_fwd_model(
        self,
        subject_id: str,
        subjects_dir: str,
        source: str = 'surface',
        template_mri: bool = True,
        redo_hdm: bool = True,
    ) -> None:
        """
        Generate a forward model for source reconstruction.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        subjects_dir : str
            Path to the FreeSurfer subjects directory.
        source : str, optional
            Type of source space ('surface' or 'volume'). Defaults to 'surface'.
        template_mri : bool, optional
            Whether to use a template MRI. Defaults to True.
        redo_hdm : bool, optional
            Whether to recompute the head model. Defaults to True.

        Returns
        -------
        None
        """

        # This should generate a fwd model
        if self.raw is not None:
            cur_info = self.raw.info
        elif self.epoched is not None:
            cur_info = self.epoched.info

        # fetch fsaverage if subjects_dir is not yet there
        freesurfer_dir = Path(subjects_dir) / 'freesurfer'
        if freesurfer_dir.is_dir() is False:
            print('Download missing freesurfer fsaverage data for source modelling.')
            mne.datasets.fetch_fsaverage(freesurfer_dir)
            # also build a downsampled version of the ico-5 to save some processing power
            src = mne.setup_source_space(
                subject='fsaverage',  # Subject name
                spacing='ico4',  # Use ico-4 source spacing
                add_dist=False,  # Avoid computing inter-source distances (optional)
                subjects_dir=freesurfer_dir,  # FreeSurfer's subjects directory
            )

            # Save the source space to a file
            mne.write_source_spaces(f'{freesurfer_dir}/fsaverage/bem/fsaverage-ico-4-src.fif', src)

        if redo_hdm:
            # recompute or take the saved one
            trans = compute_headmodel(
                info=cur_info,
                subject_id=subject_id,
                subjects_dir=subjects_dir,
                pick_dict=self.pick_dict,
                template_mri=template_mri,
            )
        else:
            trans = Path(subjects_dir) / 'headmodels' / subject_id / (subject_id + '-trans.fif')

        fwd = make_fwd(
            cur_info,
            source=source,
            fname_trans=trans,
            subjects_dir=subjects_dir,
            subject_id=subject_id,
            template_mri=template_mri,
        )

        self.fwd = fwd

    def do_spatial_filters(
        self,
        fwd: None | mne.Forward = None,
        data_cov: None | NDArray = None,
        noise_cov: None | NDArray = None,
        empty_room: None | str | mne.io.Raw = None,
        get_nearest_empty_room: bool = False,
    ) -> None:
        """
        Compute spatial filters for source projection using LCMV beamformers.

        Parameters
        ----------
        fwd : mne.Forward | None, optional
            The forward model. Defaults to None.
        data_cov : NDArray | None, optional
            Data covariance matrix. Defaults to None.
        noise_cov : NDArray | None, optional
            Noise covariance matrix. Defaults to None.
        empty_room : str | mne.io.Raw | None, optional
            Path to or preloaded empty room recording. Defaults to None.
        get_nearest_empty_room : bool, optional
            Whether to find the nearest empty room recording. Defaults to False.

        Returns
        -------
        None
        """

        # here we want to embed the logic that, if your object has been epoched we do epoched2src else raw2src
        if np.logical_and(self.fwd is not None, fwd is not None):
            raise ValueError('You cannot set fwd as you already have a fwd model in the Almkanal.')

        if fwd is not None:
            self.fwd = fwd

        if self.fwd is not None:
            data = self.raw if self.raw is not None else self.epoched

            self.filters = comp_spatial_filters(
                data=data,
                fwd=self.fwd,
                pick_dict=self.pick_dict,
                icas=self.ica,
                ica_ids=self.ica_ids,
                data_cov=data_cov,
                noise_cov=noise_cov,
                preproc_info=self.info,
                empty_room=empty_room,
                get_nearest_empty_room=get_nearest_empty_room,
            )
        else:
            raise ValueError('The pipeline needs a forward model to be able to compute spatial filters.')

    def do_src(
        self,
        return_parc: bool = False,
        label_mode: str = 'mean_flip',
        subject_id: None | str = None,
        subjects_dir: None | str = None,
        atlas: str = 'glasser',
        source: str = 'surface',
    ) -> None:
        """
        Perform source reconstruction and optional parcellation.

        Parameters
        ----------
        return_parc : bool, optional
            Whether to return parcellated source data. Defaults to False.
        label_mode : str, optional
            Mode for extracting label time courses ('mean_flip', etc.). Defaults to 'mean_flip'.
        subject_id : str | None, optional
            Subject identifier for parcellation. Required if `return_parc` is True.
        subjects_dir : str | None, optional
            Path to FreeSurfer subjects directory. Required if `return_parc` is True.
        atlas : str, optional
            Atlas for parcellation ('glasser', 'dk', etc.). Defaults to 'glasser'.
        source : str, optional
            Source space type ('surface' or 'volume'). Defaults to 'surface'.

        Returns
        -------
        mne.SourceEstimate | dict
            Source time courses or parcellated data.
        """

        if self.filters is None:
            raise ValueError('You need to compute spatial filters before you are able to go to source.')

        data = self.raw if self.raw is not None else self.epoched

        if isinstance(data, mne.io.fiff.raw.Raw):
            stc = mne.beamformer.apply_lcmv_raw(data, self.filters)

        elif isinstance(data, mne.epochs.Epochs):
            stc = mne.beamformer.apply_lcmv_epochs(data, self.filters)

        if np.logical_and(return_parc, np.logical_and(subject_id is not None, subjects_dir is not None)):
            assert isinstance(
                subject_id, str
            ), 'You need to set the correct name for the `subject_id` and `subjects_dir` if you want to parcels.'
            assert isinstance(
                subjects_dir, str
            ), 'You need to set the correct name for the `subject_id` and `subjects_dir` if you want to parcels.'
            stc = src2parc(
                stc,
                subject_id=subject_id,
                subjects_dir=subjects_dir,
                atlas=atlas,
                source=source,
                label_mode=label_mode,
            )

        return stc

    def do_trf_epochs(self) -> None:
        # mne only allows epochs of equal length.
        # This should become a shorthand to split the raw file in smaller raw files based on events
        pass
