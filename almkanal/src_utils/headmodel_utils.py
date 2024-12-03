import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.coreg import Coregistration

from almkanal.data_utils.data_classes import PickDictClass


def compute_headmodel(
    info: mne.Info,
    subject_id: str,
    subjects_dir: str,
    pick_dict: PickDictClass,
    template_mri: bool = True,
    savefig: bool = True,
) -> mne.transforms.Transform:
    """
    base_data_path: The path to the data directory where freesurfer and the headmodels are going to be stored.
                    The folder in which the data is saved is dependent on this.

    """

    mri_path = Path(subjects_dir) / 'freesurfer'  # os.path.join(subjects_dir, 'freesurfer')
    out_folder = Path(subjects_dir) / 'headmodels' / subject_id  # os.path.join(subjects_dir, 'headmodels', subject_id)
    trans = 'fsaverage'

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
    new_source_identifier = subject_id + '_from_template' if template_mri else subject_id

    # Dont forget this sdcales bem, atlas and source space automatically too
    mne.coreg.scale_mri(
        'fsaverage', new_source_identifier, scale=coreg.scale, subjects_dir=mri_path, annot=True, overwrite=True
    )

    if out_folder.is_dir() is False:
        out_folder.mkdir(parents=True)
    file = Path.open(out_folder / (subject_id + 'info.pickle'), 'wb')
    # open(os.path.join(out_folder, subject_id) + 'info.pickle', 'wb')
    pickle.dump(info, file)
    file.close()

    mne.write_trans(Path(out_folder) / (subject_id + '-trans.fif'), coreg.trans, overwrite=True)
    print('Coregistration done!')

    if savefig:  # REDO IT LATER
        # PLACEHOLDER: plot 3d stuff when xvfb is available
        # Assuming 'info' is your data structure containing sensor and digitization information
        plot_head_model(coreg, info, subject_id, out_folder)
        # plt.savefig(os.path.join(out_folder, subject_id) + '_coreg.png', dpi=300)
        # not sohw ... plt.show()

    return coreg.trans


def plot_head_model(coreg: mne.transforms.Transform, info: mne.Info, subject_id: str, out_folder: Path) -> None:
    head_mri_t = mne.transforms._get_trans(coreg.trans, 'head', 'mri')[0]
    coord_frame = 'head'
    to_cf_t = mne.transforms._get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)

    sensor_locs = np.array([ch['loc'][:3] for ch in info['chs'] if ch['ch_name'].startswith('MEG')])
    sensor_locs = mne.transforms.apply_trans(to_cf_t['meg'], sensor_locs)

    # Extract Digitization Points (excluding fiducials)
    fids = 4
    head_shape_points = np.array([point['r'] for point in info['dig'] if point['kind'] == fids])
    head_shape_points = mne.transforms.apply_trans(to_cf_t['head'], head_shape_points)

    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f'MEG - DIG Coregistration of {subject_id}', fontsize=16)

    # Axial view
    ax1 = fig.add_subplot(221)
    ax1.scatter(sensor_locs[:, 0], sensor_locs[:, 1], s=20, c='r', label='Sensors')
    ax1.scatter(head_shape_points[:, 0], head_shape_points[:, 1], s=10, c='b', label='Head Shape')
    ax1.set_title('Axial View')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Distance (m)')
    ax1.legend()

    # Coronal view
    ax2 = fig.add_subplot(222)
    ax2.scatter(sensor_locs[:, 0], sensor_locs[:, 2], s=20, c='r')
    ax2.scatter(head_shape_points[:, 0], head_shape_points[:, 2], s=10, c='b')
    ax2.set_title('Coronal View')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Distance (m)')

    # Sagittal view
    ax3 = fig.add_subplot(223)
    ax3.scatter(sensor_locs[:, 1], sensor_locs[:, 2], s=20, c='r')
    ax3.scatter(head_shape_points[:, 1], head_shape_points[:, 2], s=10, c='b')
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
    ax4.set_title('3D View')
    ax4.grid(False)  # Remove grid
    ax4.axis('off')  # Remove axis

    plt.tight_layout()
    # save

    plt.savefig(Path(out_folder) / (subject_id + '_coreg.png'), dpi=300)


def make_fwd(
    info: mne.Info, source: str, fname_trans: str, subjects_dir: str, subject_id: str, template_mri: bool = False
) -> mne.Forward:
    ###### MAKE FORWARD SOLUTION AND INVERSE OPERATOR
    fpath_add_on = '_from_template' if template_mri else ''

    # fs_path = os.path.join(subjects_dir, 'freesurfer', f'{subject_id}{fpath_add_on}')
    fs_path = Path(subjects_dir) / 'freesurfer' / f'{subject_id}{fpath_add_on}'
    bem_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-5120-5120-5120-bem.fif'

    if source == 'volume':
        src_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-vol-5-src.fif'

    elif source == 'surface':
        src_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-ico-4-src.fif'

    # if isinstance(fname_trans, str):
    #     fname_trans = os.path.join(fname_trans, subject_id, subject_id + '-trans.fif')

    bem_sol = mne.make_bem_solution(bem_file, solver='mne', verbose=True)
    fwd = mne.make_forward_solution(info=info, trans=fname_trans, src=src_file, bem=bem_sol)

    return fwd
