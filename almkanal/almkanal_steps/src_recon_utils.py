from pathlib import Path

import mne
import numpy as np
from attrs import define

from almkanal import AlmKanalStep


def src2parc(  # noqa: C901, PLR0912
    stc: mne.SourceEstimate | mne.VolSourceEstimate | list,
    fs: float,
    subject_id: str | None,
    subjects_dir: Path | str,
    source: str | None = 'surface',
    atlas: str = 'glasser',
    label_mode: str = 'mean_flip',
    *,
    src: mne.SourceSpaces | None = None,
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
    if not subject_id:
        raise ValueError('subject_id must be the actual FreeSurfer subject name.')
    if source not in ('surface', 'volume'):
        raise ValueError("source must be 'surface' or 'volume'.")
    atlases = {
        'dk': ('aparc', 'aparc+aseg'),
        'destrieux': ('aparc.a2009s', 'aparc.a2009s+aseg'),
        'glasser': ('HCPMMP1', None),
    }
    if atlas not in atlases:
        raise ValueError("atlas must be 'dk', 'destrieux', or 'glasser'.")
    surf_atlas, vol_atlas = atlases[atlas]
    if source == 'volume' and vol_atlas is None:
        raise ValueError('No volumetric model for the glasser atlas is available.')

    fs_dir = Path(subjects_dir).expanduser().resolve()
    if not (fs_dir / subject_id).is_dir() and (fs_dir / 'freesurfer' / subject_id).is_dir():
        fs_dir = fs_dir / 'freesurfer'
    subject_path = fs_dir / subject_id
    if not subject_path.is_dir():
        raise FileNotFoundError(f'FreeSurfer subject not found: {subject_path}')

    if src is None:
        # Compatibility for callers using existing template source-space files.
        # ForwardModel -> SourceReconstruction always supplies fwd['src'].
        suffix = 'ico-4' if source == 'surface' else 'vol-5'
        src_file = subject_path / 'bem' / f'{subject_id}-{suffix}-src.fif'
        if not src_file.is_file():
            raise ValueError("Pass src=fwd['src']; no compatible saved source space was found.")
        src = mne.read_source_spaces(src_file)

    if src.kind != source:
        raise ValueError(f'source={source!r} does not match the supplied source space ({src.kind!r}).')
    src_subjects = {space.get('subject_his_id') for space in src} - {None}
    if src_subjects and src_subjects != {subject_id}:
        raise ValueError(f'Source space subjects {src_subjects} do not match {subject_id!r}.')

    if source == 'surface':
        for hemi in ('lh', 'rh'):
            annot_file = subject_path / 'label' / f'{hemi}.{surf_atlas}.annot'
            if not annot_file.is_file():
                raise FileNotFoundError(
                    f'Missing annotation: {annot_file}. Prepare this atlas for '
                    'the individual FreeSurfer subject before parcellation.'
                )
        labels_mne = mne.read_labels_from_annot(subject_id, parc=surf_atlas, subjects_dir=fs_dir)
        return {
            'lh': [label.hemi == 'lh' for label in labels_mne],
            'rh': [label.hemi == 'rh' for label in labels_mne],
            'parc': surf_atlas,
            'names_order_mne': np.array([label.name[:-3] for label in labels_mne]),
            'fs': fs,
            'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode=label_mode),
        }

    labels_mne = subject_path / 'mri' / f'{vol_atlas}.mgz'
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

    filters: None | mne.beamformer.Beamformer = None
    return_parc: bool = False
    label_mode: str = 'pca_flip'
    subject_id: str | None = None
    subjects_dir: Path | str | None = None
    atlas: str = 'glasser'
    source: str | None = None
    must_be_before: tuple = ()
    must_be_after: tuple = ('Maxwell', 'ICA', 'ForwardModel', 'SpatialFilter')
    src: mne.SourceSpaces | None = None

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict) -> dict:
        fwd_info = info.get('ForwardModel', {}).get('fwd_info', {})
        spatial_info = info.get('SpatialFilter', {}).get('spatial_filter_info', {})
        filters = self.filters if self.filters is not None else spatial_info.get('filters')
        if filters is None:
            raise ValueError('Provide filters or run SpatialFilter before SourceReconstruction.')

        source = self.source if self.source is not None else fwd_info.get('source_type')
        subject_id = self.subject_id if self.subject_id is not None else fwd_info.get('subject_id_freesurfer')
        subjects_dir = self.subjects_dir
        if subjects_dir is None:
            subjects_dir = fwd_info.get('subjects_dir', fwd_info.get('subject_dir'))
        src = self.src
        if src is None and 'fwd' in fwd_info:
            src = fwd_info['fwd']['src']
        if source is None and src is not None:
            source = src.kind

        if isinstance(data, mne.io.BaseRaw):
            stc = mne.beamformer.apply_lcmv_raw(data, filters)
        elif isinstance(data, mne.BaseEpochs):
            stc = mne.beamformer.apply_lcmv_epochs(data, filters)
        else:
            raise TypeError('data must be an MNE Raw or Epochs object.')

        if self.return_parc:
            if subject_id is None or subjects_dir is None or source is None:
                raise ValueError(
                    'Parcellation needs subject_id, subjects_dir, and source, '
                    'either explicitly or from ForwardModel.'
                )
            result = src2parc(
                stc,
                fs=data.info['sfreq'],
                subject_id=subject_id,
                subjects_dir=subjects_dir,
                source=source,
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
                'atlas': self.atlas,
                'source': source,
            },
        }

    def reports(self, data: dict | mne.SourceEstimate | mne.VolSourceEstimate, report: mne.Report, info: dict) -> None:
        import matplotlib.pyplot as plt
        import scipy.signal as dsp

        if self.return_parc and info['SourceReconstruction']['stc_info']['orig_data_type'] == 'raw':
            freq, psd = dsp.welch(data['label_tc'], fs=data['fs'], nperseg=data['fs'] * 4, noverlap=data['fs'] * 2)
            f, ax = plt.subplots(ncols=2, figsize=(15, 5))
            for cax, title in zip(ax, ['SemiLog', 'LogLog']):
                cax.set_title(title)
                cax.set_xlabel('Frequency (Hz)')
                cax.set_ylabel('Power (Log)')
            ax[0].semilogy(freq, psd.T, alpha=0.25)
            ax[1].loglog(freq, psd.T, alpha=0.25)
            report.add_figure(fig=f, title='ParcellationPowerSpectra', image_format='PNG', caption='')
