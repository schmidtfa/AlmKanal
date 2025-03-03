import mne
import numpy as np
from pathlib import Path


def src2parc(
    stc: mne.SourceEstimate,
    subject_id: str,
    subjects_dir: str,
    atlas: str = 'glasser',
    source: str = 'surface',
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

        parc = {'lh': lh, 'rh': rh, 'parc': surf_atlas, 'names_order_mne': names_order_mne}
        parc.update({'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode=label_mode)})
    elif source == 'volume':
        src_file = f'{fs_dir}/{subject_id}_from_template/bem/{subject_id}_from_template-vol-5-src.fif'
        src = mne.read_source_spaces(src_file)
        labels_mne = (
            fs_dir / f'{subject_id}_from_template' / 'mri' / (vol_atlas + '.mgz')
        )  # os.path.join(fs_dir, f'{subject_id}_from_template', 'mri/' + vol_atlas + '.mgz')
        label_names = mne.get_volume_labels_from_aseg(labels_mne)

        ctx_logical = 'ctx' in label_names  # [True if 'ctx' in label else False for label in label_names]
        sctx_logical = not ctx_logical  # [True if not f else False for f in ctx_logical]

        ctx_labels = np.array([label[4:] for label in label_names if 'ctx' in label])
        sctx_labels = list(np.array(label_names)[sctx_logical])
        rh = ctx_labels[:2] == 'rh'  # [True if label[:2] == 'rh' else False for label in ctx_labels]
        lh = ctx_labels[:2] == 'lh'  # [True if label[:2] == 'lh' else False for label in ctx_labels]

        parc = {
            'lh': lh,
            'rh': rh,
            'parc': vol_atlas + '.mgz',
            'ctx_labels': ctx_labels,
            'ctx_logical': ctx_logical,
            'sctx_logical': sctx_logical,
            'sctx_labels': sctx_labels,
        }
        parc.update(
            {'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode='auto')}
        )  # NOTE: This needs to be auto

    else:
        raise ValueError('the only valid options for source are `surface` and `volume`.')

    return parc
