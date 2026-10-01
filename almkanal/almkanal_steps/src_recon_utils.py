from pathlib import Path

import mne
import numpy as np
from attrs import define, field

from almkanal import AlmKanalStep
from almkanal.info import AlmKanalInfo


def src2parc(  # noqa: C901, PLR0912
    stc: mne.SourceEstimate | mne.VolSourceEstimate | list,
    src: mne.SourceSpaces,
    fs: float,
    subject_id: str,
    subjects_dir: Path,
    atlas: str = 'glasser',
    label_mode: str = 'pca_flip',
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

    atlases = {
        'dk': ('aparc', 'aparc+aseg'),
        'destrieux': ('aparc.a2009s', 'aparc.a2009s+aseg'),
        'glasser': ('HCPMMP1', None),
    }
    if atlas not in atlases:
        raise ValueError("atlas must be 'dk', 'destrieux', or 'glasser'.")
    surf_atlas, vol_atlas = atlases[atlas]

    if src.kind == 'volume' and vol_atlas is None:
        raise ValueError('No volumetric model for the glasser atlas is available.')

    if src.kind == 'surface':
        for hemi in ('lh', 'rh'):
            annot_file = subjects_dir / subject_id / 'label' / f'{hemi}.{surf_atlas}.annot'
            if not annot_file.is_file():
                raise FileNotFoundError(
                    f'Missing annotation: {annot_file}. Prepare this atlas for '
                    'the individual FreeSurfer subject before parcellation.'
                )
        labels_mne = mne.read_labels_from_annot(subject_id, parc=surf_atlas, subjects_dir=subjects_dir)
        return {
            'lh': [label.hemi == 'lh' for label in labels_mne],
            'rh': [label.hemi == 'rh' for label in labels_mne],
            'parc': surf_atlas,
            'names_order_mne': np.array([label.name[:-3] for label in labels_mne]),
            'fs': fs,
            'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode=label_mode),
        }

    labels_mne = subjects_dir / subject_id / 'mri' / f'{vol_atlas}.mgz'
    if not labels_mne.is_file():
        raise FileNotFoundError(f'Missing volumetric atlas: {labels_mne}')
    label_names = mne.get_volume_labels_from_aseg(labels_mne)
    ctx_logical = ['ctx' in label for label in label_names]
    sctx_logical = [not is_ctx for is_ctx in ctx_logical]
    ctx_labels = np.array([label[4:] for label in label_names if 'ctx' in label])
    return {
        'lh': [label[:2] == 'lh' for label in ctx_labels],
        'rh': [label[:2] == 'rh' for label in ctx_labels],
        'parc': f'{vol_atlas}.mgz',
        'labels_mne': label_names,
        'ctx_labels': ctx_labels,
        'ctx_logical': ctx_logical,
        'sctx_logical': sctx_logical,
        'sctx_labels': list(np.array(label_names)[sctx_logical]),
        'fs': fs,
        'label_tc': mne.extract_label_time_course(stc, str(labels_mne), src, mode='auto'),
    }


@define
class SourceReconstruction(AlmKanalStep):
    """
    Perform source reconstruction using a preceding ForwardModel.

    The source space, FreeSurfer subject, and FreeSurfer subjects directory
    are taken from the preceding ForwardModel to ensure that source
    reconstruction uses the same anatomical model and source geometry.

    Parameters
    ----------
    filters : mne.beamformer.Beamformer | None, optional
        Spatial filter to apply. If None, use the filters produced by the
        preceding SpatialFilter step.

    return_parc : bool, optional
        Whether to parcellate the reconstructed source estimates.
        Default is False.

    label_mode : str, optional
        Extraction mode used for surface parcellation. Volume
        parcellation uses ``mode='auto'``. Default is ``'pca_flip'``.

    atlas : {'glasser', 'dk', 'destrieux'}, optional
        Atlas used when ``return_parc=True``. Default is ``'glasser'``.

    morph2fsaverage : bool, optional
        Whether to morph source estimates to the common ``fsaverage``
        source space prepared by ForwardModel before returning or
        parcellating them. Default is True.

        Returns
    -------
    dict | mne.SourceEstimate | dict | mne.VolSourceEstimate
        Source time courses or parcellated data.
    """

    filters: None | mne.beamformer.Beamformer = None
    return_parc: bool = False
    label_mode: str = 'pca_flip'
    atlas: str = 'glasser'
    morph2fsaverage: bool = True

    must_be_before: tuple = ()
    must_be_after: tuple = ('Maxwell', 'ICA', 'ForwardModel')
    allow_repeated: bool = field(default=False, init=False)

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: AlmKanalInfo) -> dict:  # noqa: C901, PLR0912
        fwd_info = info.get_step('ForwardModel')
        if fwd_info is None:
            raise ValueError(
                'SourceReconstruction requires a completed ForwardModel step. '
                'Run ForwardModel before SourceReconstruction.'
            )
        fwd_info = info.get_step_info('ForwardModel', required=True)['fwd_info']

        spatial_info = {}
        filters = self.filters

        if filters is None:
            spatial_info = info.get_step_info('SpatialFilter', required=True)
            if spatial_info is None:
                raise ValueError('Provide filters or run SpatialFilter before SourceReconstruction.')

            spatial_info = spatial_info['spatial_filter_info']
            filters = spatial_info.get('filters')

        fwd = fwd_info['fwd']
        src = fwd['src']
        subject_id = fwd_info['subject_id_freesurfer']
        subjects_dir = Path(fwd_info['subjects_dir'])

        if isinstance(data, mne.io.BaseRaw):
            stc = mne.beamformer.apply_lcmv_raw(data, filters)
        elif isinstance(data, mne.BaseEpochs):
            stc = mne.beamformer.apply_lcmv_epochs(data, filters)
        else:
            raise TypeError('data must be an MNE Raw or Epochs object.')

        if self.morph2fsaverage:
            src_to = mne.read_source_spaces(fwd_info['template_src'])

            morph = mne.compute_source_morph(
                src,
                subject_from=subject_id,
                subject_to='fsaverage',
                src_to=src_to,
                subjects_dir=subjects_dir,
            )

            if isinstance(data, mne.BaseEpochs):
                stc = [morph.apply(epoch_stc) for epoch_stc in stc]
            else:
                stc = morph.apply(stc)

            src = src_to
            subject_id = 'fsaverage'

        if self.return_parc:
            result = src2parc(
                stc,
                fs=data.info['sfreq'],
                subject_id=subject_id,
                subjects_dir=subjects_dir,
                atlas=self.atlas,
                label_mode=self.label_mode,
                src=src,
            )
        else:
            result = {'label_tc': stc, 'fs': data.info['sfreq']}

        result['extra_data'] = spatial_info.get('extra_data')
        if isinstance(data, mne.BaseEpochs):
            result['metadata'] = data.metadata
        return {
            'data': result,
            'stc_info': {
                'orig_data_type': 'raw' if isinstance(data, mne.io.BaseRaw) else 'epochs',
                'subject_id': subject_id,
                'subjects_dir': subjects_dir,
                'label_mode': self.label_mode,
                'effective_label_mode': (
                    ('auto' if src.kind == 'volume' else self.label_mode) if self.return_parc else None
                ),
                'atlas': self.atlas,
                'source': src.kind,
            },
        }

    def reports(
        self, data: dict | mne.SourceEstimate | mne.VolSourceEstimate, report: mne.Report, info: AlmKanalInfo
    ) -> None:
        import matplotlib.pyplot as plt
        import scipy.signal as dsp

        if (
            self.return_parc
            and info.get_step_info('SourceReconstruction', required=True)['stc_info']['orig_data_type'] == 'raw'
        ):
            freq, psd = dsp.welch(
                data['label_tc'], fs=data['fs'], nperseg=round(data['fs'] * 4), noverlap=round(data['fs'] * 2)
            )
            f, ax = plt.subplots(ncols=2, figsize=(15, 5))
            for cax, title in zip(ax, ['SemiLog', 'LogLog']):
                cax.set_title(title)
                cax.set_xlabel('Frequency (Hz)')
                cax.set_ylabel('Power (Log)')
            ax[0].semilogy(freq, psd.T, alpha=0.25)
            ax[1].loglog(freq, psd.T, alpha=0.25)
            report.add_figure(fig=f, title='ParcellationPowerSpectra', image_format='PNG', caption='')
