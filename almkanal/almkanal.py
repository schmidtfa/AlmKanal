from pathlib import Path

import attrs
import mne
import numpy as np
import pandas as pd
from attrs import define, field
from numpy.typing import ArrayLike, NDArray

from almkanal.data_utils.check_data import check_raw_epoch
from almkanal.data_utils.data_classes import ICAInfoDict, InfoClass, PickDictClass
from almkanal.preproc_utils.ica_utils import run_ica

# all them utility functions
from almkanal.preproc_utils.maxwell_utils import run_maxwell
from almkanal.src_utils.headmodel_utils import compute_headmodel, make_fwd
from almkanal.src_utils.src_utils import data2source, src2parc


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
        resample_freq: int = 200,  # downsample to 200hz per default
        eog: bool = True,
        ecg: bool = True,
        muscle: bool = False,
        train: bool = True,
        train_freq: int = 16,
        threshold: float = 0.4,
        img_path: None | str = None,
        fname: None | str = None,
    ) -> None:
        # this should do an ica
        ica_info = ICAInfoDict(
            n_components=n_components,
            method=method,
            resample_freq=resample_freq,
            eog=eog,
            ecg=ecg,
            muscle=muscle,
            train=train,
            train_freq=train_freq,
            ica_corr_thresh=threshold,
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

    def do_src(
        self,
        data_cov: None | NDArray = None,
        noise_cov: None | NDArray = None,
        empty_room: None | str | mne.io.Raw = None,
        get_nearest_empty_room: bool = False,
        return_parc: bool = False,
        subject_id: None | str = None,
        subjects_dir: None | str = None,
        fwd: None | mne.Forward = None,
        atlas: str = 'glasser',
        source: str = 'surface',
    ) -> None:
        # here we want to embed the logic that, if your object has been epoched we do epoched2src else raw2src
        if np.logical_and(self.fwd is not None, fwd is not None):
            raise ValueError('You cannot set fwd as you already have a fwd model in the Almkanal.')

        if fwd is not None:
            self.fwd = fwd

        if self.fwd is not None:
            data = self.raw if self.raw is not None else self.epoched

            stc, self.filters = data2source(
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

            if np.logical_and(return_parc, np.logical_and(subject_id is not None, subjects_dir is not None)):
                assert isinstance(
                    subject_id, str
                ), 'You need to set the correct name for the `subject_id` and `subjects_dir` if you want to parcels.'
                assert isinstance(
                    subjects_dir, str
                ), 'You need to set the correct name for the `subject_id` and `subjects_dir` if you want to parcels.'
                stc = src2parc(stc, subject_id=subject_id, subjects_dir=subjects_dir, atlas=atlas, source=source)

        else:
            raise ValueError('The pipeline needs a forward model to be able to go to source.')

        return stc

    def do_trf_epochs(self) -> None:
        # mne only allows epochs of equal length.
        # This should become a shorthand to split the raw file in smaller raw files based on events
        pass

    def convert2eelbrain(self) -> None:
        # This should take the thht mixin to convert raw, epoched or stc objects into eelbrain
        pass
