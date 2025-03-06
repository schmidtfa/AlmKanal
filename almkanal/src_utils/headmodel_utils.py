import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from attrs import define
from mne.coreg import Coregistration

from almkanal.almkanal import AlmKanalStep


def compute_headmodel(
    info: mne.Info,
    subject_id: str,
    subjects_dir: str,
    pick_dict: dict | None,
    template_mri: bool = True,
) -> mne.transforms.Transform:
    """
    Compute a head model for MEG data by coregistering it to an MRI.

    Parameters
    ----------
    info : mne.Info
        The MEG data information structure.
    subject_id : str
        Subject identifier for the head model.
    subjects_dir : str
        Path to the directory containing subject-specific data (e.g., FreeSurfer and head models).
    pick_dict : dict
        Dictionary specifying channels to include in the head model.
    template_mri : bool, optional
        Whether to use a template MRI ('fsaverage'). Defaults to True.

    Returns
    -------
    mne.transforms.Transform
        The transformation matrix for aligning MEG and MRI coordinate systems.
    """

    mri_path = Path(subjects_dir) / 'freesurfer'  # os.path.join(subjects_dir, 'freesurfer')
    out_folder = Path(subjects_dir) / 'headmodels' / subject_id  # os.path.join(subjects_dir, 'headmodels', subject_id)
    trans = 'fsaverage'

    if pick_dict is not None:
        info = mne.pick_info(info, mne.pick_types(info, **pick_dict))

    # %% do the coregistration
    coreg = Coregistration(info, trans, mri_path)
    coreg.set_scale_mode('3-axis')
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2, verbose=True)
    coreg.omit_head_shape_points(distance=5 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10, verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f'Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm '
        f'/ {np.min(dists):.2f} mm / '
        f'{np.max(dists):.2f} mm'
    )

    # %% create and save a scaled copy of mri subject fs average

    # Dont forget this sdcales bem, atlas and source space automatically too
    mne.coreg.scale_mri('fsaverage', subject_id, scale=coreg.scale, subjects_dir=mri_path, annot=True, overwrite=True)

    if out_folder.is_dir() is False:
        out_folder.mkdir(parents=True)
    file = Path.open(out_folder / (subject_id + 'info.pickle'), 'wb')
    # open(os.path.join(out_folder, subject_id) + 'info.pickle', 'wb')
    pickle.dump(info, file)
    file.close()

    mne.write_trans(Path(out_folder) / (subject_id + '-trans.fif'), coreg.trans, overwrite=True)
    print('Coregistration done!')

    fig = plot_head_model(coreg.trans, info, subject_id, mri_path)

    return coreg.trans, fig


def plot_head_model(  # noqa PLR0912, PLR0915
    coreg: mne.transforms.Transform,
    info: mne.Info,
    subject_id: str,
    subjects_dir: str | Path,
) -> matplotlib.figure.Figure:
    """
    Plot the head model coregistration, including sensor and digitization points.

    Parameters
    ----------
    coreg : mne.transforms.Transform
        The coregistration transform matrix aligning MEG and MRI coordinate systems.
    info : mne.Info
        The MEG data information structure.
    subject_id : str
        Subject identifier for labeling the plots.
    out_folder : Path
        Path to the directory where the coregistration plot will be saved.

    Returns
    -------
    None
    """

    head_mri_t = mne.transforms._get_trans(coreg, 'head', 'mri')[0]
    coord_frame = 'head'
    to_cf_t = mne.transforms._get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)

    sensor_locs = np.array([ch['loc'][:3] for ch in info['chs'] if ch['ch_name'].startswith('MEG')])
    sensor_locs = mne.transforms.apply_trans(to_cf_t['meg'], sensor_locs)

    # Extract Digitization Points (excluding fiducials)
    fids = 4
    head_shape_points = np.array([point['r'] for point in info['dig'] if point['kind'] == fids])
    head_shape_points = mne.transforms.apply_trans(to_cf_t['head'], head_shape_points)

    # also adjust the bem
    bem_path = Path(subjects_dir) / subject_id / 'bem' / f'{subject_id}-head.fif'  # inner_skull-bem
    bem_surfaces = mne.read_bem_surfaces(bem_path)

    for bem in bem_surfaces:
        bem['rr'] = mne.transforms.apply_trans(to_cf_t['mri'], bem['rr'])

    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f'MEG - DIG Coregistration of {subject_id}', fontsize=16)

    # Axial view
    ax1 = fig.add_subplot(221)
    ax1.scatter(sensor_locs[:, 0], sensor_locs[:, 1], s=20, c='r', label='Sensors')
    ax1.scatter(head_shape_points[:, 0], head_shape_points[:, 1], s=10, c='b', label='Head Shape')
    first = True
    for bem in bem_surfaces:
        if first:
            ax1.scatter(bem['rr'][:, 0], bem['rr'][:, 1], s=1, c='gray', alpha=0.5, label='BEM')
            first = False
        else:
            ax1.scatter(bem['rr'][:, 0], bem['rr'][:, 1], s=1, c='gray', alpha=0.5)
    ax1.set_title('Axial View')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Distance (m)')
    ax1.legend()

    # Coronal view
    ax2 = fig.add_subplot(222)
    ax2.scatter(sensor_locs[:, 0], sensor_locs[:, 2], s=20, c='r')
    ax2.scatter(head_shape_points[:, 0], head_shape_points[:, 2], s=10, c='b')
    first = True
    for bem in bem_surfaces:
        if first:
            ax2.scatter(bem['rr'][:, 0], bem['rr'][:, 2], s=1, c='gray', alpha=0.5, label='BEM')
            first = False
        else:
            ax2.scatter(bem['rr'][:, 0], bem['rr'][:, 2], s=1, c='gray', alpha=0.5)
    ax2.set_title('Coronal View')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Distance (m)')

    # Sagittal view
    ax3 = fig.add_subplot(223)
    ax3.scatter(sensor_locs[:, 1], sensor_locs[:, 2], s=20, c='r')
    ax3.scatter(head_shape_points[:, 1], head_shape_points[:, 2], s=10, c='b')
    first = True
    for bem in bem_surfaces:
        if first:
            ax3.scatter(bem['rr'][:, 1], bem['rr'][:, 2], s=1, c='gray', alpha=0.5, label='BEM')
            first = False
        else:
            ax3.scatter(bem['rr'][:, 1], bem['rr'][:, 2], s=1, c='gray', alpha=0.5)
    ax3.set_title('Sagittal View')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Distance (m)')

    # 3D plot
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(
        sensor_locs[:, 0],
        sensor_locs[:, 1],
        sensor_locs[:, 2],
        s=20,
        c='r',
        label='Sensors',
    )  # type: ignore
    ax4.plot(
        sensor_locs[:, 0], sensor_locs[:, 1], sensor_locs[:, 2], color='k', linewidth=0.5
    )  # Connect sensors with lines
    ax4.scatter(
        head_shape_points[:, 0],
        head_shape_points[:, 1],
        head_shape_points[:, 2],
        s=10,
        c='b',
        label='Head Shape',
    )  # type: ignore
    first = True
    for bem in bem_surfaces:
        if first:
            ax4.scatter(bem['rr'][:, 0], bem['rr'][:, 1], bem['rr'][:, 2], s=1, c='gray', alpha=0.5, label='BEM')  # type: ignore
            first = False
        else:
            ax4.scatter(bem['rr'][:, 0], bem['rr'][:, 1], bem['rr'][:, 2], s=1, c='gray', alpha=0.5)  # type: ignore
    ax4.set_title('3D View')
    ax4.grid(False)  # Remove grid
    ax4.axis('off')  # Remove axis

    plt.tight_layout()
    return fig


def make_fwd(
    info: mne.Info, source: str, fname_trans: str, subjects_dir: str, subject_id: str, template_mri: bool = False
) -> mne.Forward:
    """
    Generate a forward model for MEG data.

    Parameters
    ----------
    info : mne.Info
        The MEG data information structure.
    source : str
        Type of source space ('volume' or 'surface').
    fname_trans : str
        Path to the transformation file aligning MEG and MRI coordinate systems.
    subjects_dir : str
        Path to the directory containing subject-specific data (e.g., FreeSurfer).
    subject_id : str
        Subject identifier for the forward model.
    template_mri : bool, optional
        Whether to use a template MRI ('fsaverage'). Defaults to False.

    Returns
    -------
    mne.Forward
        The computed forward model.
    """

    ###### MAKE FORWARD SOLUTION AND INVERSE OPERATOR
    # fpath_add_on = '_from_template' if template_mri else ''

    # fs_path = os.path.join(subjects_dir, 'freesurfer', f'{subject_id}{fpath_add_on}')
    fs_path = Path(subjects_dir) / 'freesurfer' / f'{subject_id}'  # {fpath_add_on}'
    bem_file = f'{fs_path}/bem/{subject_id}-5120-5120-5120-bem.fif'  # {fpath_add_on}

    if source == 'volume':
        src_file = f'{fs_path}/bem/{subject_id}-vol-5-src.fif'  # {fpath_add_on}

    elif source == 'surface':
        src_file = f'{fs_path}/bem/{subject_id}-ico-4-src.fif'  # {fpath_add_on}

    # if isinstance(fname_trans, str):
    #     fname_trans = os.path.join(fname_trans, subject_id, subject_id + '-trans.fif')

    bem_sol = mne.make_bem_solution(bem_file, solver='mne', verbose=True)
    fwd = mne.make_forward_solution(info=info, trans=fname_trans, src=src_file, bem=bem_sol)

    return fwd


@define
class ForwardModel(AlmKanalStep):
    subject_id: str
    subjects_dir: str
    pick_dict: dict | None = None

    must_be_before: tuple = (
        'SpatialFilter',
        'SourceReconstruction',
    )
    must_be_after: tuple = (
        'Maxwell',
        'ICA',
    )

    source: str = 'surface'
    template_mri: bool = True
    redo_hdm: bool = True

    def run(
        self,
        data: mne.io.BaseRaw | mne.BaseEpochs,
        info: dict,
    ) -> dict:
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
        mne.Forward
        """
        if self.pick_dict is None and info['Picks'] is not None:
            self.pick_dict = info['Picks']

        new_source_identifier = self.subject_id + '_from_template' if self.template_mri else self.subject_id

        # fetch fsaverage if subjects_dir and fsaverage is not yet there
        freesurfer_dir = Path(self.subjects_dir) / 'freesurfer'
        if (freesurfer_dir / 'fsaverage').is_dir() is False:
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

        if self.redo_hdm:
            # recompute or take the saved one
            trans, fig = compute_headmodel(
                info=data.info,
                subject_id=new_source_identifier,
                subjects_dir=self.subjects_dir,
                pick_dict=self.pick_dict,
                template_mri=self.template_mri,
            )

        else:
            trans = Path(self.subjects_dir) / 'headmodels' / self.subject_id / (self.subject_id + '-trans.fif')
            fig = plot_head_model(trans, data.info, subject_id=new_source_identifier, subjects_dir=freesurfer_dir)

        fwd = make_fwd(
            data.info,
            source=self.source,
            fname_trans=trans,
            subjects_dir=self.subjects_dir,
            subject_id=new_source_identifier,
            template_mri=self.template_mri,
        )

        return {
            'data': data,
            'fwd_info': {
                'coreg_fig': fig,
                'fwd': fwd,
                'subject_id_freesurfer': new_source_identifier,
                'subject_dir': self.subjects_dir,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: dict) -> None:
        report.add_figure(
            fig=info['ForwardModel']['fwd_info']['coreg_fig'],
            title='Coregistration',
            image_format='PNG',
            caption='',
        )
        report.add_forward(info['ForwardModel']['fwd_info']['fwd'], title='ForwardModel')
