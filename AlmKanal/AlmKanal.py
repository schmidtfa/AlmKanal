from pathlib import Path

import attrs
import mne
import numpy as np
from attrs import define, field

from AlmKanal.data_utils.check_data import check_raw_epoch
from AlmKanal.data_utils.data_classes import InfoClass, PickDictClass
from AlmKanal.preproc_utils.ica_utils import run_ica

# all them utility functions
from AlmKanal.preproc_utils.maxwell_utils import run_maxwell
from AlmKanal.src_utils.headmodel_utils import compute_headmodel, make_fwd
from AlmKanal.src_utils.src_utils import data2source, src2parc


@define
class AlmKanal:
    raw: None | mne.io.Raw = None
    epoched: None | mne.Epochs = None
    pick_dict: dict = PickDictClass(meg=True, eog=True, ecg=True, eeg=False, stim=True)
    events: None | np.ndarray = None
    fwd: None | mne.Forward = None
    ica: None | mne.preprocessing.ICA = None
    ica_ids: None | list = None
    filters: None | mne.beamformer.Beamformer = None
    stim: None | np.ndarray = None  # This is for TRF stuff
    info: dict = field(default=attrs.Factory(InfoClass))
    _initialized: bool = field(default=False, init=False)  # Internal flag for initialization
    _data_check_disabled: bool = field(default=False, init=False)  # Disable data check temporarily

    # TODO: Use attrs validators -> checks whenver a field is set
    # Check that we only have either raw or epoched data when we initalize the method
    def __attrs_post_init__(self):
        check_raw_epoch(self)
        self._initialized = True

    # Check that we only have either raw or epoched data whenever we add a raw or epoched attribute
    def __setattr__(self, name, value):
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
    ):
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
        n_components=None,
        method='picard',
        resample_freq=200,  # downsample to 200hz per default
        eog=True,
        ecg=True,
        muscle=False,
        train=True,
        train_freq=16.6,
        threshold=0.4,
    ):
        # this should do an ica
        ica_info = {
            'n_components': n_components,
            'method': method,
            'resample_freq': resample_freq,
            'eog': eog,
            'ecg': ecg,
            'muscle': muscle,
            'train': train,
            'train_freq': train_freq,
            'ica_corr_thresh': threshold,
        }

        if self.info.ica is None:
            self.info.ica = ica_info

            self.raw, ica, ica_ids = run_ica(self.raw, **self.info.ica)
            self.ica = [ica]
            self.ica_ids = [ica_ids]
        else:
            # Take care of case where you ran multiple icas.
            # TODO: When we end up applying them to the noise cov dont forget
            # to also do it successively.
            self.info.ica.update(ica_info)

            self.raw, ica, ica_ids = run_ica(self.raw, **self.info['ica'])
            self.ica.append(ica)
            self.ica_ids.append(ica_ids)

    def do_events(
        self,
        stim_channel=None,
        output='onset',
        consecutive='increasing',
        min_duration=0,
        shortest_event=2,
        mask=None,
        uint_cast=False,
        mask_type='and',
        initial_event=False,
        verbose=None,
    ):
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
        tmin,
        tmax,
        event_id=None,
        baseline=None,
        preload=True,
        picks=None,
        reject=None,
        flat=None,
        proj=True,
        decim=1,
        reject_tmin=None,
        reject_tmax=None,
        detrend=None,
        on_missing='raise',
        reject_by_annotation=True,
        metadata=None,
        event_repeated='error',
        verbose=None,
    ):
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

    def do_fwd_model(self, subject_id, subjects_dir, source='surface', template_mri=True, redo_hdm=True):
        # This should generate a fwd model
        if self.raw is not None:
            cur_info = self.raw.info
        elif self.epoched is not None:
            cur_info = self.epoched.info

        # fetch fsaverage if subjects_dir is not yet there
        freesurfer_dir = Path(subjects_dir) / 'freesurfer'
        if Path.isdir(freesurfer_dir) is False:
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
            trans = Path(subjects_dir) / 'headmodels' / subject_id / subject_id + '-trans.fif'

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
        data_cov=None,
        noise_cov=None,
        empty_room_path=None,
        return_parc=False,
        subject_id=None,
        subjects_dir=None,
        fwd=None,
        atlas='glasser',
        source='surface',
    ):
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
                empty_room_path=empty_room_path,
            )

            if return_parc:
                if np.logical_and(subject_id is not None, subjects_dir is not None):
                    stc = src2parc(stc, subject_id=subject_id, subjects_dir=subjects_dir, atlas=atlas, source=source)
                else:
                    raise ValueError(
                        """You need to set the correct name for the `subject_id` and `subjects_dir`
                        if you want to parcels."""
                    )

        else:
            raise ValueError('The pipeline needs a forward model to be able to go to source.')

        return stc

    def do_trf_epochs(self):
        # mne only allows epochs of equal length.
        # This should become a shorthand to split the raw file in smaller raw files based on events
        pass

    def convert2eelbrain(self):
        # This should take the thht mixin to convert raw, epoched or stc objects into eelbrain
        pass
