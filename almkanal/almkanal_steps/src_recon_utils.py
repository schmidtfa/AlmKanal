from pathlib import Path

import mne
import numpy as np
from attrs import define

from almkanal import AlmKanalStep


def src2parc(
    stc: mne.SourceEstimate,
    fs: int,
    subject_id: str | None,
    subjects_dir: Path | str,
    source: str | None = 'surface',
    atlas: str = 'glasser',
    label_mode: str = 'mean_flip',
) -> dict:
    """
    Parcellate source data into predefined brain regions using an atlas.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate object containing the source data to parcellate.
    subject_id : str
        Subject identifier for the source data.
    subjects_dir : str
        Path to the directory containing FreeSurfer subject data.
    atlas : str, optional
        Atlas to use for parcellation ('glasser', 'dk', or 'destrieux'). Defaults to 'glasser'.
    source : str, optional
        Source space type ('surface' or 'volume'). Defaults to 'surface'.
    label_mode : str, optional
        Mode for extracting label time courses ('mean', 'mean_flip', etc.). Defaults to 'mean_flip'.

    Returns
    -------
    dict
        Dictionary containing parcellation information, including labels, hemisphere assignments,
        and extracted time courses for each region.
    """

    if atlas == 'dk':
        vol_atlas = 'aparc+aseg'
        surf_atlas = 'aparc'
    elif atlas == 'destrieux':
        vol_atlas = 'aparc.a2009s+aseg'
        surf_atlas = 'aparc.a2009s'
    elif atlas == 'glasser':
        if source == 'volume':
            raise ValueError('No volumetric model for the glasser atlas available')
        surf_atlas = 'HCPMMP1'

    fs_dir = Path(subjects_dir) / 'freesurfer'
    # mean flip time series costs significantly less memory than averaging the irasa'd spectra
    if source == 'surface':
        src_file = f'{fs_dir}/{subject_id}_from_template/bem/{subject_id}_from_template-ico-4-src.fif'
        src = mne.read_source_spaces(src_file)
        labels_mne = mne.read_labels_from_annot(f'{subject_id}_from_template', parc=surf_atlas, subjects_dir=fs_dir)
        names_order_mne = np.array([label.name[:-3] for label in labels_mne])

        rh = [label.hemi == 'rh' for label in labels_mne]
        lh = [label.hemi == 'lh' for label in labels_mne]

        parc = {'lh': lh, 'rh': rh, 'parc': surf_atlas, 'names_order_mne': names_order_mne, 'fs': fs}
        parc.update({'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode=label_mode)})
    elif source == 'volume':
        src_file = f'{fs_dir}/{subject_id}_from_template/bem/{subject_id}_from_template-vol-5-src.fif'
        src = mne.read_source_spaces(src_file)
        labels_mne = (
            fs_dir / f'{subject_id}_from_template' / 'mri' / (vol_atlas + '.mgz')
        )  # os.path.join(fs_dir, f'{subject_id}_from_template', 'mri/' + vol_atlas + '.mgz')

        label_names = mne.get_volume_labels_from_aseg(labels_mne)

        ctx_logical = ['ctx' in label for label in label_names]
        sctx_logical = [not f for f in ctx_logical]

        ctx_labels = np.array([label[4:] for label in label_names if 'ctx' in label])
        sctx_labels = list(np.array(label_names)[sctx_logical])
        rh = [label[:2] == 'rh' for label in ctx_labels]
        lh = [label[:2] == 'lh' for label in ctx_labels]

        parc = {
            'lh': lh,
            'rh': rh,
            'parc': vol_atlas + '.mgz',
            'labels_mne': label_names,
            'ctx_labels': ctx_labels,
            'ctx_logical': ctx_logical,
            'sctx_logical': sctx_logical,
            'sctx_labels': sctx_labels,
            'fs': fs,
        }
        parc.update(
            {'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode='auto')}
        )  # NOTE: This needs to be auto

    else:
        raise ValueError('the only valid options for source are `surface` and `volume`.')

    return parc


@define
class SourceReconstruction(AlmKanalStep):
    filters: None | mne.beamformer.Beamformer = None
    return_parc: bool = False
    label_mode: str = 'pca_flip'
    subject_id: str | None = None
    subjects_dir: Path | str | None = None
    atlas: str = 'glasser'
    source: str | None = None

    must_be_before: tuple = ()
    must_be_after: tuple = (
        'Maxwell',
        'ICA',
        'ForwardModel',
        'SpatialFilter',
    )

    def run(  # noqa C901
        self,
        data: mne.io.BaseRaw | mne.BaseEpochs,
        info: dict,
    ) -> dict | mne.SourceEstimate | mne.VolSourceEstimate:
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
        dict | mne.SourceEstimate | dict | mne.VolSourceEstimate
            Source time courses or parcellated data.
        """

        if self.filters is None:
            self.filters = info['SpatialFilter']['spatial_filter_info']['filters']

        if self.source is None:
            self.source = info['ForwardModel']['fwd_info']['source_type']

        if isinstance(data, mne.io.BaseRaw):
            stc = mne.beamformer.apply_lcmv_raw(data, self.filters)

        elif isinstance(data, mne.BaseEpochs):
            stc = mne.beamformer.apply_lcmv_epochs(data, self.filters)

        if self.return_parc:
            if np.logical_and(self.subject_id is None, 'ForwardModel' in info):
                self.subject_id: str = info['ForwardModel']['fwd_info']['subject_id_freesurfer']

            elif np.logical_and(self.subject_id is None, 'ForwardModel' not in info):
                assert isinstance(
                    self.subject_id, str
                ), 'You need to set the correct name for the `subject_id` if you want to get parcels.'

            if np.logical_and(self.subjects_dir is None, 'ForwardModel' in info):
                self.subjects_dir: str = info['ForwardModel']['fwd_info']['subjects_dir']

            elif np.logical_and(self.subjects_dir is None, 'ForwardModel' not in info):
                assert isinstance(
                    self.subject_id, str
                ), 'You need to set the correct name for the `subjects_dir` if you want to get parcels.'

            # handle case if subjects_dir is still None
            if self.subjects_dir is None:
                raise ValueError('You need to set the correct name for the `subjects_dir` if you want to get parcels.')

            stc = src2parc(
                stc,
                fs=data.info['sfreq'],
                subject_id=self.subject_id,
                subjects_dir=self.subjects_dir,
                atlas=self.atlas,
                source=self.source,
                label_mode=self.label_mode,
            )

        if type(stc) is dict:
            stc['extra_data'] = info['SpatialFilter']['spatial_filter_info']['extra_data']
        else:
            stc = {'stc': stc, 'extra_data': info['SpatialFilter']['spatial_filter_info']['extra_data']}

        # add metadata for events to src file
        if isinstance(data, mne.BaseEpochs):
            stc['metadata'] = data.metadata

        return {
            'data': stc,
            'stc_info': {
                'orig_data_type': 'raw' if isinstance(data, mne.io.BaseRaw) else 'epochs',
                'subject_id': self.subject_id,
                'subjects_dir': self.subjects_dir,
                'label_mode': self.label_mode,
                'atlas': self.atlas,
                'source': self.source,
            },
        }

    def reports(self, data: dict | mne.SourceEstimate | mne.VolSourceEstimate, report: mne.Report, info: dict) -> None:
        import matplotlib.pyplot as plt
        import scipy.signal as dsp

        if isinstance(data, dict) and info['SourceReconstruction']['stc_info']['orig_data_type'] == 'raw':
            freq, psd = dsp.welch(data['label_tc'], fs=data['fs'], nperseg=data['fs'] * 4, noverlap=data['fs'] * 2)

            f, ax = plt.subplots(ncols=2, figsize=(15, 5))

            for cax, title in zip(ax, ['SemiLog', 'LogLog']):
                cax.set_title(title)
                cax.set_xlabel('Frequency (Hz)')
                cax.set_ylabel('Power (Log)')

            ax[0].semilogy(freq, psd.T, alpha=0.25)
            ax[1].loglog(freq, psd.T, alpha=0.25)
            report.add_figure(
                fig=f,
                title='ParcellationPowerSpectra',
                image_format='PNG',
                caption='',
            )
