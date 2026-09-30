import pickle
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from attrs import define, field
from mne.coreg import Coregistration

from almkanal.almkanal import AlmKanalStep
from almkanal.info import AlmKanalInfo


def compute_headmodel(
    info: mne.Info,
    subject_id: str,
    subjects_dir: str | Path,
    pick_dict: dict | None,
    use_template_mri: bool = True,
    plot_coreg: bool = True,
) -> tuple[mne.transforms.Transform, matplotlib.figure.Figure | None]:
    """Estimate head-to-MRI coregistration and return its transform and figure.

    For template anatomy, fit and scale fsaverage to the recording.
    For individual anatomy, fit a rigid transform without scaling the MRI.

    subject_id is the output/cache identifier supplied by ForwardModel.
    For template processing, ForwardModel already includes the
    '_from_template' suffix in this identifier.

    When supplied, mri_path selects the FreeSurfer subject independently
    of the output/cache identifier.
    """

    mri_path = Path(subjects_dir) / 'freesurfer'
    out_folder = Path(subjects_dir) / 'headmodels' / subject_id

    if pick_dict is not None:
        info = mne.pick_info(info, mne.pick_types(info, **pick_dict))

    coreg = Coregistration(
        info,
        subject='fsaverage' if use_template_mri else subject_id,
        subjects_dir=mri_path,
        fiducials='auto',
    )
    coreg.set_scale_mode('3-axis' if use_template_mri else None)
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2, verbose=True)
    coreg.omit_head_shape_points(distance=5 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10, verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3
    print(
        f'Distance between HSP and MRI (mean/min/max):\n'
        f'{np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm'
    )

    if use_template_mri:
        mne.coreg.scale_mri(
            'fsaverage',
            subject_id,
            scale=coreg.scale,
            subjects_dir=mri_path,
            annot=True,
            overwrite=True,
        )

    out_folder.mkdir(parents=True, exist_ok=True)
    with (out_folder / f'{subject_id}info.pickle').open('wb') as file:
        pickle.dump(info, file)
    mne.write_trans(out_folder / f'{subject_id}-trans.fif', coreg.trans, overwrite=True)

    fig = plot_head_model(coreg.trans, info, subject_id, mri_path) if plot_coreg else None

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
    mri_path: Path,
    subject_id: str,
    spacing: str = 'oct6',
    source_ico: int = 4,
    bem_conductivity: float | Sequence[float] = (0.3,),  # fine for MEG should be changed for EEG
    volume_pos: float = 5.0,
    min_dist_src: float = 5,  # in mm
    use_template_mri: bool = True,
    meg: bool = True,
    eeg: bool = False,
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

    model = mne.make_bem_model(
        subject_id, ico=4, conductivity=bem_conductivity, subjects_dir=mri_path
    )  # this ico is different from source ico
    bem = mne.make_bem_solution(model, solver='mne', verbose=True)

    if use_template_mri:
        suffix = f'ico-{int(source_ico)}' if source == 'surface' else f'vol-{volume_pos:g}'
        src_file = mri_path / subject_id / 'bem' / f'{subject_id}-{suffix}-src.fif'
        if not src_file.is_file():
            mne.scale_source_space(
                subject_to=subject_id,
                src_name=f'{{subject}}-{suffix}-src.fif',
                subjects_dir=mri_path,
            )
        return mne.make_forward_solution(
            info=info, trans=fname_trans, src=src_file, bem=bem, meg=meg, eeg=eeg, mindist=min_dist_src
        )

    if source == 'surface':
        src = mne.setup_source_space(
            subject_id,
            spacing=spacing,
            surface='white',
            subjects_dir=mri_path,
            add_dist=True,
        )
    else:
        src = mne.setup_volume_source_space(
            subject_id,
            pos=volume_pos,
            bem=bem,
            subjects_dir=mri_path,
            add_interpolator=True,
        )

    return mne.make_forward_solution(
        info=info,
        trans=fname_trans,
        src=src,
        bem=bem,
        meg=meg,
        eeg=eeg,
        mindist=min_dist_src,
    )


@define
class ForwardModel(AlmKanalStep):
    """
    Build an MEG forward model for source reconstruction.

    The forward model combines the sensor geometry, MEG-to-MRI
    coregistration, source space, and boundary-element model (BEM).

    Individual FreeSurfer reconstructions are expected under
    ``subjects_dir / 'freesurfer' / subject_id``. When template anatomy is
    used, ``fsaverage`` is scaled to the participant and stored as
    ``<subject_id>_from_template``.

    Parameters
    ----------
    subject_id : str
        Subject identifier.

    subjects_dir : str | Path
        AlmKanal working directory. FreeSurfer subjects are stored under
        ``subjects_dir / 'freesurfer'`` and coregistration outputs under
        ``subjects_dir / 'headmodels'``.

    pick_dict : dict | None, optional
        Channel selection passed to :func:`mne.pick_types` before
        coregistration. If None, channel picks are taken from the pipeline
        information when available.

    source : {'surface', 'volume'}, optional
        Type of source space used for the forward model. Default is
        ``'surface'``.

    redo_hdm : bool, optional
        If True, recompute the MEG-to-MRI coregistration and, when using
        template anatomy, rescale ``fsaverage``. If False, use the
        previously saved transform. Default is True.

    spacing : str, optional
        Source-space spacing used for surface source spaces based on an
        individual MRI, for example ``'oct6'`` or ``'ico5'``. Default is
        ``'oct6'``.

    source_ico : int, optional
        Icosahedral subdivision level of the ``fsaverage`` surface source
        space used for template-based forward models and as the common
        surface source space for morphing. This controls source-space
        density and is independent of the BEM surface resolution.
        Default is 4.

    bem_conductivity : float | Sequence[float], optional
        Conductivity value or values passed to :func:`mne.make_bem_model`.
        A one-layer BEM is typically used for MEG, whereas EEG requires a
        three-layer BEM. Default is ``(0.3,)``.

    volume_pos : float, optional
        Grid spacing in millimetres for volume source spaces. The same
        spacing is used when preparing the corresponding ``fsaverage``
        volume source space for morphing. Default is 5.0.

    min_dist_src : float, optional
        Minimum distance in millimetres between sources and the inner skull
        used when computing the forward solution. Default is 5.0.

    use_template_mri : bool, optional
        If True, use a participant-specific scaling of ``fsaverage`` as the
        anatomical model. If False, use the participant's existing
        FreeSurfer reconstruction. Default is True.

    meg : bool, optional
        Include MEG channels in the forward solution. Default is True.

    eeg : bool, optional
        Include EEG channels in the forward solution. EEG forward models
        require a three-layer BEM conductivity specification. Default is
        False.

    Notes
    -----
    The BEM surface resolution is fixed internally and is independent of
    ``source_ico``. ``source_ico`` controls the number of cortical source
    locations, not the resolution of the BEM geometry.

    An ``fsaverage`` source space corresponding to the requested surface or
    volume resolution is prepared so that later source estimates can be
    morphed to a common source space.

    Returns
    -------
    mne.Forward
    """

    subject_id: str
    subjects_dir: str | Path
    pick_dict: dict | None = None
    source: str = 'surface'
    redo_hdm: bool = True
    spacing: str = 'oct6'
    source_ico: int = 4
    bem_conductivity: float | Sequence[float] = (0.3,)  # fine for MEG should be changed for EEG
    volume_pos: float = 5.0
    use_template_mri: bool = True
    min_dist_src: float = 5.0
    meg: bool = True
    eeg: bool = False

    must_be_before: tuple = ('SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ('Maxwell', 'ICA')
    allow_repeated: bool = field(default=False, init=False)

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: AlmKanalInfo) -> dict:
        if self.source not in ('surface', 'volume'):
            raise ValueError("source must be 'surface' or 'volume'.")

        pick_dict = self.pick_dict if self.pick_dict is not None else info.pick_params
        cache_id = f'{self.subject_id}_from_template' if self.use_template_mri else self.subject_id

        if self.eeg and np.size(self.bem_conductivity) == 1:
            raise ValueError(
                'EEG forward models require a three-layer BEM. ' 'Set bem_conductivity to three conductivity values.'
            )

        fs_dir = Path(self.subjects_dir) / 'freesurfer'
        fs_dir.mkdir(parents=True, exist_ok=True)
        mne.datasets.fetch_fsaverage(subjects_dir=fs_dir)

        # This is run to ensure that appropriate template files exist that we can use for morphing
        if self.source == 'surface':
            template_src = fs_dir / 'fsaverage' / 'bem' / f'fsaverage-ico-{self.source_ico}-src.fif'
            if not template_src.is_file():
                src = mne.setup_source_space(
                    subject='fsaverage',
                    spacing=f'ico{self.source_ico}',
                    add_dist=False,
                    subjects_dir=fs_dir,
                )
                mne.write_source_spaces(template_src, src, overwrite=True)
        else:
            template_src = fs_dir / 'fsaverage' / 'bem' / f'fsaverage-vol-{self.volume_pos:g}-src.fif'
            if not template_src.is_file():
                fsaverage_model = mne.make_bem_model(
                    'fsaverage',
                    ico=4,
                    conductivity=(0.3,),
                    subjects_dir=fs_dir,
                )  # Use the fsaverage inner-skull BEM as the volume source-space boundary.
                fsaverage_bem = mne.make_bem_solution(fsaverage_model)
                src = mne.setup_volume_source_space(
                    subject='fsaverage',
                    pos=self.volume_pos,
                    bem=fsaverage_bem,
                    mri=fs_dir / 'fsaverage' / 'mri' / 'T1.mgz',
                    subjects_dir=fs_dir,
                    add_interpolator=True,
                )
                mne.write_source_spaces(template_src, src, overwrite=True)

        trans_file = Path(self.subjects_dir) / 'headmodels' / cache_id / f'{cache_id}-trans.fif'
        if self.redo_hdm:
            trans, fig = compute_headmodel(
                info=data.info,
                subject_id=cache_id,
                subjects_dir=self.subjects_dir,
                pick_dict=pick_dict,
                use_template_mri=self.use_template_mri,
                plot_coreg=self.meg,
            )
        else:
            if not trans_file.is_file():
                raise FileNotFoundError(f'No saved transform at {trans_file}. Run with redo_hdm=True first.')
            trans = mne.read_trans(trans_file)
            fig = plot_head_model(trans, data.info, subject_id=cache_id, subjects_dir=fs_dir) if self.meg else None

        fwd = make_fwd(
            data.info,
            source=self.source,
            fname_trans=trans,
            mri_path=fs_dir,
            subject_id=cache_id,
            spacing=self.spacing,
            min_dist_src=self.min_dist_src,
            use_template_mri=self.use_template_mri,
            bem_conductivity=self.bem_conductivity,
            source_ico=self.source_ico,
            volume_pos=self.volume_pos,
            meg=self.meg,
            eeg=self.eeg,
        )
        return {
            'data': data,
            'fwd_info': {
                'coreg_fig': fig,
                'fwd': fwd,
                'source_type': self.source,
                'subject_id_freesurfer': cache_id,
                'subjects_dir': str(fs_dir),
                'template_src': str(template_src),
                'subject_dir': self.subjects_dir,  # Legacy workspace metadata.
                'template_mri': self.use_template_mri,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: AlmKanalInfo) -> None:
        fig = info.get_step_info('ForwardModel', required=True)['fwd_info']['coreg_fig']
        if fig is not None:
            report.add_figure(
                fig=fig,
                title='Coregistration',
                image_format='PNG',
                caption='',
            )
        report.add_forward(info.get_step_info('ForwardModel', required=True)['fwd_info']['fwd'], title='ForwardModel')
