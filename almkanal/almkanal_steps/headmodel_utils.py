import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from attrs import define
from mne.coreg import Coregistration

from almkanal.almkanal import AlmKanalStep


def _resolve_mri_paths(
    subject_id: str,
    subjects_dir: str | Path,
    mri_path: str | Path | None = None,
) -> tuple[Path, str]:
    """Return the FreeSurfer subjects directory and actual MRI subject name."""
    if mri_path is None:
        return Path(subjects_dir).expanduser().resolve() / 'freesurfer', subject_id

    subject_path = Path(mri_path).expanduser().resolve()
    if not subject_path.is_dir():
        raise ValueError(
            f'mri_path must be an existing FreeSurfer subject directory, ' f'not a T1 image file: {subject_path}'
        )
    if not all((subject_path / folder).is_dir() for folder in ('mri', 'surf')):
        raise ValueError(f'{subject_path} is not a FreeSurfer subject directory containing mri/ and surf/.')
    return subject_path.parent, subject_path.name


def compute_headmodel(
    info: mne.Info,
    subject_id: str,
    subjects_dir: str | Path,
    pick_dict: dict | None,
    template_mri: bool = True,
    *,
    mri_path: str | Path | None = None,
) -> tuple[mne.transforms.Transform, matplotlib.figure.Figure]:
    """Estimate head-to-MRI coregistration and return its transform and figure.

    For template anatomy, fit and scale fsaverage to the recording.
    For individual anatomy, fit a rigid transform without scaling the MRI.

    subject_id is the output/cache identifier supplied by ForwardModel.
    For template processing, ForwardModel already includes the
    '_from_template' suffix in this identifier.

    When supplied, mri_path selects the FreeSurfer subject independently
    of the output/cache identifier.
    """
    use_template = template_mri and mri_path is None
    fs_dir, mri_subject = _resolve_mri_paths(subject_id, subjects_dir, mri_path)
    out_folder = Path(subjects_dir).expanduser().resolve() / 'headmodels' / subject_id

    if pick_dict is not None:
        info = mne.pick_info(info, mne.pick_types(info, **pick_dict))

    coreg = Coregistration(
        info,
        subject='fsaverage' if use_template else mri_subject,
        subjects_dir=fs_dir,
        fiducials='auto',
    )
    coreg.set_scale_mode('3-axis' if use_template else None)
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2, verbose=True)
    coreg.omit_head_shape_points(distance=5 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10, verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3
    if dists.size:
        print(
            f'Distance between HSP and MRI (mean/min/max):\n'
            f'{np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm'
        )

    if use_template:
        mne.coreg.scale_mri(
            'fsaverage',
            mri_subject,
            scale=coreg.scale,
            subjects_dir=fs_dir,
            annot=True,
            overwrite=True,
        )

    out_folder.mkdir(parents=True, exist_ok=True)
    with (out_folder / f'{subject_id}info.pickle').open('wb') as file:
        pickle.dump(info, file)
    mne.write_trans(out_folder / f'{subject_id}-trans.fif', coreg.trans, overwrite=True)

    fig = plot_head_model(coreg.trans, info, mri_subject, fs_dir)
    return coreg.trans, fig


def plot_head_model(  # noqa: PLR0912, PLR0915, C901
    coreg: mne.transforms.Transform | str | Path,
    info: mne.Info,
    subject_id: str,
    subjects_dir: str | Path,
) -> matplotlib.figure.Figure:
    """Plot sensor/digitization alignment with a FreeSurfer scalp surface."""
    head_mri_t = mne.transforms._get_trans(coreg, 'head', 'mri')[0]
    coord_frame = 'head'
    to_cf_t = mne.transforms._get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)

    sensor_locs = np.array([ch['loc'][:3] for ch in info['chs'] if ch['ch_name'].startswith('MEG')]).reshape(-1, 3)
    sensor_locs = mne.transforms.apply_trans(to_cf_t['meg'], sensor_locs)
    fids = 4
    head_shape_points = np.array([point['r'] for point in (info['dig'] or []) if point['kind'] == fids]).reshape(-1, 3)
    head_shape_points = mne.transforms.apply_trans(to_cf_t['head'], head_shape_points)

    subject_path = Path(subjects_dir) / subject_id
    bem_path = subject_path / 'bem' / f'{subject_id}-head.fif'
    if bem_path.is_file():
        bem_surfaces = mne.read_bem_surfaces(bem_path)
    else:
        # Coregistration also accepts these FreeSurfer-format scalp surfaces.
        scalp_paths = (
            subject_path / 'bem' / 'outer_skin.surf',
            subject_path / 'surf' / 'lh.seghead',
            subject_path / 'surf' / 'lh.smseghead',
        )
        scalp_path = next((path for path in scalp_paths if path.is_file()), None)
        if scalp_path is not None:
            rr, _ = mne.read_surface(scalp_path)
            bem_surfaces = [{'rr': rr / 1000.0}]  # FreeSurfer mm -> MNE m.
        else:
            bem_surfaces = [
                mne.get_head_surf(
                    subject_id,
                    source=('head-sparse', 'head-dense', 'bem'),
                    subjects_dir=subjects_dir,
                )
            ]

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
    info: mne.Info,
    source: str,
    fname_trans: str | Path | mne.transforms.Transform,
    subjects_dir: str | Path,
    subject_id: str,
    template_mri: bool = False,
    spacing: str = 'oct6',
    *,
    mri_path: str | Path | None = None,
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
    if source not in ('surface', 'volume'):
        raise ValueError("source must be 'surface' or 'volume'.")

    use_template = template_mri and mri_path is None
    fs_dir, mri_subject = _resolve_mri_paths(subject_id, subjects_dir, mri_path)
    subject_path = fs_dir / mri_subject

    if use_template:
        bem_file = subject_path / 'bem' / f'{mri_subject}-5120-5120-5120-bem.fif'
        suffix = 'ico-4' if source == 'surface' else 'vol-5'
        src_file = subject_path / 'bem' / f'{mri_subject}-{suffix}-src.fif'
        bem = mne.make_bem_solution(bem_file, solver='mne', verbose=True)
        return mne.make_forward_solution(info=info, trans=fname_trans, src=src_file, bem=bem)

    inner_skull = subject_path / 'bem' / 'inner_skull.surf'
    if not inner_skull.is_file():
        raise FileNotFoundError(
            f'Missing BEM surface: {inner_skull}. Prepare BEM surfaces first, '
            'for example with mne.bem.make_watershed_bem().'
        )
    model = mne.make_bem_model(mri_subject, ico=4, conductivity=(0.3,), subjects_dir=fs_dir)
    bem = mne.make_bem_solution(model)

    if source == 'surface':
        src = mne.setup_source_space(
            mri_subject,
            spacing=spacing,
            surface='white',
            subjects_dir=fs_dir,
            add_dist=True,
        )
    else:
        src = mne.setup_volume_source_space(
            mri_subject,
            pos=5.0,
            mri=subject_path / 'mri' / 'T1.mgz',
            bem=bem,
            subjects_dir=fs_dir,
            add_interpolator=True,
        )

    return mne.make_forward_solution(
        info=info,
        trans=fname_trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
    )


@define
class ForwardModel(AlmKanalStep):
    """
    Build a forward model for source reconstruction.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    subjects_dir : str | Path
        AlmKanal working directory. Template anatomy is stored under
        ``subjects_dir / 'freesurfer'`` and headmodel outputs under
        ``subjects_dir / 'headmodels'``.
    mri_path : str | Path | None, optional
        Path to one prepared FreeSurfer subject directory, not a raw
        MRI image file and not the parent directory containing subjects.
        When provided, individual anatomy is used regardless of
        ``template_mri``. Defaults to None.
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

    subject_id: str
    subjects_dir: str | Path
    pick_dict: dict | None = None
    must_be_before: tuple = ('SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ('Maxwell', 'ICA')
    source: str = 'surface'
    template_mri: bool = True
    redo_hdm: bool = True
    mri_path: str | Path | None = None

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict) -> dict:
        if self.source not in ('surface', 'volume'):
            raise ValueError("source must be 'surface' or 'volume'.")

        pick_dict = self.pick_dict if self.pick_dict is not None else info.get('Picks')
        use_template = self.template_mri and self.mri_path is None
        cache_id = f'{self.subject_id}_from_template' if use_template else self.subject_id
        fs_dir, mri_subject = _resolve_mri_paths(cache_id, self.subjects_dir, self.mri_path)

        if use_template:
            fs_dir.mkdir(parents=True, exist_ok=True)
            mne.datasets.fetch_fsaverage(subjects_dir=fs_dir)

            # This must also run when fsaverage was downloaded previously.
            template_src = fs_dir / 'fsaverage' / 'bem' / 'fsaverage-ico-4-src.fif'
            if not template_src.is_file():
                src = mne.setup_source_space(
                    subject='fsaverage',
                    spacing='ico4',
                    add_dist=False,
                    subjects_dir=fs_dir,
                )
                mne.write_source_spaces(template_src, src, overwrite=True)

        trans_file = Path(self.subjects_dir).expanduser().resolve() / 'headmodels' / cache_id / f'{cache_id}-trans.fif'
        if self.redo_hdm:
            trans, fig = compute_headmodel(
                info=data.info,
                subject_id=cache_id,
                subjects_dir=self.subjects_dir,
                pick_dict=pick_dict,
                template_mri=use_template,
                mri_path=self.mri_path,
            )
        else:
            if not trans_file.is_file():
                raise FileNotFoundError(f'No saved transform at {trans_file}. Run with redo_hdm=True first.')
            trans = mne.read_trans(trans_file)
            fig = plot_head_model(trans, data.info, subject_id=mri_subject, subjects_dir=fs_dir)

        fwd = make_fwd(
            data.info,
            source=self.source,
            fname_trans=trans,
            subjects_dir=self.subjects_dir,
            subject_id=cache_id,
            template_mri=use_template,
            mri_path=self.mri_path,
        )
        return {
            'data': data,
            'fwd_info': {
                'coreg_fig': fig,
                'fwd': fwd,
                'source_type': self.source,
                'subject_id_freesurfer': mri_subject,
                'subjects_dir': str(fs_dir),
                'subject_dir': self.subjects_dir,  # Legacy workspace metadata.
                'template_mri': use_template,
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
