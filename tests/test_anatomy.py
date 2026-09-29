"""Anatomy and source-reconstruction tests for the current API.

No dataset downloads or expensive inverse computations are performed.
"""

from pathlib import Path
from unittest.mock import Mock, call

import mne
import numpy as np
import pytest

from almkanal.almkanal_steps import headmodel_utils as hm
from almkanal.almkanal_steps import src_recon_utils as sr


@pytest.fixture
def raw_small():
    info = mne.create_info(['MEG 0111'], 100.0, ch_types='mag')
    return mne.io.RawArray(np.zeros((1, 500)), info, verbose=False)


@pytest.fixture
def epochs_small(raw_small):
    return mne.EpochsArray(
        np.zeros((2, 1, 20)),
        raw_small.info,
        verbose=False,
    )


def make_subject(path: Path) -> Path:
    for folder in ('mri', 'surf', 'bem', 'label'):
        (path / folder).mkdir(parents=True, exist_ok=True)
    return path


def surface_example(subject: str):
    vertices = [np.array([0, 1]), np.array([0, 1])]
    src = mne.SourceSpaces(
        [
            {
                'type': 'surf',
                'subject_his_id': subject,
                'vertno': vertices[hemi],
                'nuse': 2,
                'nn': np.array(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
                ),
            }
            for hemi in range(2)
        ]
    )
    stc = mne.SourceEstimate(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [10.0, 11.0, 12.0],
                [14.0, 15.0, 16.0],
            ]
        ),
        vertices=vertices,
        tmin=0,
        tstep=0.01,
        subject=subject,
    )
    labels = [
        mne.Label(
            vertices[0],
            hemi='lh',
            name='left-lh',
            subject=subject,
        ),
        mne.Label(
            vertices[1],
            hemi='rh',
            name='right-rh',
            subject=subject,
        ),
    ]
    return src, stc, labels


@pytest.mark.parametrize('use_template_mri', [False, True])
def test_compute_headmodel_keeps_real_and_template_subjects_separate(
    raw_small,
    tmp_path,
    monkeypatch,
    use_template_mri,
):
    subject = (
        'recording_from_template'
        if use_template_mri
        else 'recording'
    )
    trans = mne.transforms.Transform('head', 'mri', np.eye(4))
    coreg = Mock(trans=trans, scale=np.ones(3))
    coreg.compute_dig_mri_distances.return_value = np.array([0.001])
    factory = Mock(return_value=coreg)
    scale = Mock()
    plot = Mock(return_value='figure')

    monkeypatch.setattr(hm, 'Coregistration', factory)
    monkeypatch.setattr(mne.coreg, 'scale_mri', scale)
    monkeypatch.setattr(hm, 'plot_head_model', plot)

    result_trans, fig = hm.compute_headmodel(
        raw_small.info,
        subject,
        tmp_path,
        pick_dict=None,
        use_template_mri=use_template_mri,
        plot_coreg=False,
    )

    assert result_trans is trans
    assert fig is None
    assert factory.call_args.kwargs['subject'] == (
        'fsaverage' if use_template_mri else subject
    )
    assert factory.call_args.kwargs['subjects_dir'] == (
        tmp_path / 'freesurfer'
    )
    coreg.set_scale_mode.assert_called_once_with(
        '3-axis' if use_template_mri else None
    )
    plot.assert_not_called()

    if use_template_mri:
        assert scale.call_args.args == ('fsaverage', subject)
    else:
        scale.assert_not_called()


@pytest.mark.parametrize(
    ('source', 'source_ico', 'volume_pos', 'expected_name'),
    [
        ('surface', 5, 5.0, 'fsaverage-ico-5-src.fif'),
        ('volume', 4, 7.5, 'fsaverage-vol-7.5-src.fif'),
    ],
)
def test_forward_model_prepares_fsaverage_morph_target(
    raw_small,
    tmp_path,
    monkeypatch,
    source,
    source_ico,
    volume_pos,
    expected_name,
):
    fs_dir = tmp_path / 'freesurfer'
    fsaverage = make_subject(fs_dir / 'fsaverage')

    fetch = Mock()
    monkeypatch.setattr(mne.datasets, 'fetch_fsaverage', fetch)
    monkeypatch.setattr(
        mne.datasets,
        'fetch_hcp_mmp_parcellation',
        Mock(),
    )

    setup_surface = Mock(return_value='surface-target')
    setup_volume = Mock(return_value='volume-target')
    monkeypatch.setattr(mne, 'setup_source_space', setup_surface)
    monkeypatch.setattr(
        mne,
        'setup_volume_source_space',
        setup_volume,
    )
    write_src = Mock()
    monkeypatch.setattr(mne, 'write_source_spaces', write_src)

    bem_model = Mock(return_value='model')
    bem_solution = Mock(return_value='bem')
    monkeypatch.setattr(mne, 'make_bem_model', bem_model)
    monkeypatch.setattr(mne, 'make_bem_solution', bem_solution)

    compute = Mock(return_value=('trans', None))
    forward = Mock(return_value={'src': 'subject-src'})
    monkeypatch.setattr(hm, 'compute_headmodel', compute)
    monkeypatch.setattr(hm, 'make_fwd', forward)

    step = hm.ForwardModel(
        'recording',
        tmp_path,
        source=source,
        source_ico=source_ico,
        volume_pos=volume_pos,
        use_template_mri=False,
    )
    result = step.run(raw_small, info={})

    fetch.assert_called_once_with(subjects_dir=fs_dir)
    expected_target = fsaverage / 'bem' / expected_name
    assert Path(result['fwd_info']['template_src']) == expected_target

    if source == 'surface':
        setup_surface.assert_called_once_with(
            subject='fsaverage',
            spacing='ico5',
            add_dist=False,
            subjects_dir=fs_dir,
        )
        setup_volume.assert_not_called()
    else:
        setup_surface.assert_not_called()
        bem_model.assert_called_once_with(
            'fsaverage',
            ico=4,
            conductivity=(0.3,),
            subjects_dir=fs_dir,
        )
        bem_solution.assert_called_once_with('model')
        setup_volume.assert_called_once_with(
            subject='fsaverage',
            pos=7.5,
            bem='bem',
            mri=fsaverage / 'mri' / 'T1.mgz',
            subjects_dir=fs_dir,
            add_interpolator=True,
        )

    assert write_src.call_args.args[0] == expected_target


def test_forward_model_passes_current_source_parameters(
    raw_small,
    tmp_path,
    monkeypatch,
):
    fs_dir = tmp_path / 'freesurfer'
    fsaverage = make_subject(fs_dir / 'fsaverage')
    (fsaverage / 'bem' / 'fsaverage-ico-5-src.fif').touch()

    monkeypatch.setattr(mne.datasets, 'fetch_fsaverage', Mock())
    monkeypatch.setattr(
        mne.datasets,
        'fetch_hcp_mmp_parcellation',
        Mock(),
    )
    compute = Mock(return_value=('trans', None))
    forward = Mock(return_value={'src': 'src'})
    monkeypatch.setattr(hm, 'compute_headmodel', compute)
    monkeypatch.setattr(hm, 'make_fwd', forward)

    step = hm.ForwardModel(
        'recording',
        tmp_path,
        use_template_mri=True,
        spacing='oct5',
        source_ico=5,
        bem_conductivity=(0.3,),
        volume_pos=6.0,
        min_dist_src=3.0,
        meg=True,
        eeg=False,
    )
    result = step.run(raw_small, info={'Picks': {'meg': True}})

    assert compute.call_args.kwargs['subject_id'] == (
        'recording_from_template'
    )
    assert compute.call_args.kwargs['use_template_mri'] is True

    assert forward.call_args.kwargs['subject_id'] == (
        'recording_from_template'
    )
    assert forward.call_args.kwargs['mri_path'] == fs_dir
    assert forward.call_args.kwargs['spacing'] == 'oct5'
    assert forward.call_args.kwargs['source_ico'] == 5
    assert forward.call_args.kwargs['bem_conductivity'] == (0.3,)
    assert forward.call_args.kwargs['volume_pos'] == 6.0
    assert forward.call_args.kwargs['min_dist_src'] == 3.0
    assert forward.call_args.kwargs['meg'] is True
    assert forward.call_args.kwargs['eeg'] is False

    assert result['fwd_info']['subject_id_freesurfer'] == (
        'recording_from_template'
    )
    assert Path(result['fwd_info']['subjects_dir']) == fs_dir


@pytest.mark.parametrize(
    ('source', 'expected_suffix'),
    [
        ('surface', 'ico-5'),
        ('volume', 'vol-7.5'),
    ],
)
def test_template_make_fwd_scales_missing_source_space(
    raw_small,
    tmp_path,
    monkeypatch,
    source,
    expected_suffix,
):
    subject = 'recording_from_template'
    fs_dir = tmp_path / 'freesurfer'
    make_subject(fs_dir / subject)

    model = Mock(return_value='model')
    solution = Mock(return_value='bem')
    scale_src = Mock()
    forward = Mock(return_value='forward')
    monkeypatch.setattr(mne, 'make_bem_model', model)
    monkeypatch.setattr(mne, 'make_bem_solution', solution)
    monkeypatch.setattr(mne, 'scale_source_space', scale_src)
    monkeypatch.setattr(mne, 'make_forward_solution', forward)

    result = hm.make_fwd(
        raw_small.info,
        source=source,
        fname_trans='trans.fif',
        mri_path=fs_dir,
        subject_id=subject,
        source_ico=5,
        volume_pos=7.5,
        bem_conductivity=(0.3,),
        min_dist_src=3.0,
        use_template_mri=True,
        meg=True,
        eeg=False,
    )

    assert result == 'forward'
    model.assert_called_once_with(
        subject,
        ico=4,
        conductivity=(0.3,),
        subjects_dir=fs_dir,
    )
    solution.assert_called_once_with(
        'model',
        solver='mne',
        verbose=True,
    )
    scale_src.assert_called_once_with(
        subject_to=subject,
        src_name=f'{{subject}}-{expected_suffix}-src.fif',
        subjects_dir=fs_dir,
    )

    expected_src = (
        fs_dir
        / subject
        / 'bem'
        / f'{subject}-{expected_suffix}-src.fif'
    )
    assert forward.call_args.kwargs['src'] == expected_src
    assert forward.call_args.kwargs['mindist'] == 3.0


@pytest.mark.parametrize('source', ['surface', 'volume'])
def test_individual_make_fwd_uses_requested_source_resolution(
    raw_small,
    tmp_path,
    monkeypatch,
    source,
):
    fs_dir = tmp_path / 'freesurfer'
    subject = 'recording'
    make_subject(fs_dir / subject)

    model = Mock(return_value='model')
    solution = Mock(return_value='bem')
    surface = Mock(return_value='surface-src')
    volume = Mock(return_value='volume-src')
    forward = Mock(return_value='forward')
    monkeypatch.setattr(mne, 'make_bem_model', model)
    monkeypatch.setattr(mne, 'make_bem_solution', solution)
    monkeypatch.setattr(mne, 'setup_source_space', surface)
    monkeypatch.setattr(mne, 'setup_volume_source_space', volume)
    monkeypatch.setattr(mne, 'make_forward_solution', forward)

    hm.make_fwd(
        raw_small.info,
        source=source,
        fname_trans='trans.fif',
        mri_path=fs_dir,
        subject_id=subject,
        spacing='oct5',
        volume_pos=7.5,
        use_template_mri=False,
    )

    if source == 'surface':
        surface.assert_called_once_with(
            subject,
            spacing='oct5',
            surface='white',
            subjects_dir=fs_dir,
            add_dist=True,
        )
        volume.assert_not_called()
    else:
        surface.assert_not_called()
        volume.assert_called_once_with(
            subject,
            pos=7.5,
            bem='bem',
            subjects_dir=fs_dir,
            add_interpolator=True,
        )


def test_source_reconstruction_requires_forward_model(
    raw_small,
    monkeypatch,
):
    apply = Mock()
    monkeypatch.setattr(mne.beamformer, 'apply_lcmv_raw', apply)

    step = sr.SourceReconstruction(
        filters=object(),
        morph2fsaverage=False,
    )

    with pytest.raises(
        ValueError,
        match='requires a completed ForwardModel step',
    ):
        step.run(raw_small, info={})

    apply.assert_not_called()


def test_external_filters_do_not_require_spatial_filter(
    raw_small,
    tmp_path,
    monkeypatch,
):
    src, _, _ = surface_example('recording')
    filters = object()
    estimate = object()
    apply = Mock(return_value=estimate)
    monkeypatch.setattr(mne.beamformer, 'apply_lcmv_raw', apply)

    info = {
        'ForwardModel': {
            'fwd_info': {
                'fwd': {'src': src},
                'subject_id_freesurfer': 'recording',
                'subjects_dir': str(tmp_path),
                'template_src': str(
                    tmp_path / 'fsaverage-ico-4-src.fif'
                ),
            }
        }
    }

    step = sr.SourceReconstruction(
        filters=filters,
        morph2fsaverage=False,
    )
    result = step.run(raw_small, info)

    assert apply.call_args.args[1] is filters
    assert result['data']['label_tc'] is estimate
    assert result['stc_info']['subject_id'] == 'recording'
    assert result['stc_info']['source'] == 'surface'
    assert Path(result['stc_info']['subjects_dir']) == tmp_path
    assert 'SpatialFilter' not in step.must_be_after


@pytest.mark.parametrize('as_epochs', [False, True])
def test_source_reconstruction_morphs_to_fsaverage(
    raw_small,
    epochs_small,
    tmp_path,
    monkeypatch,
    as_epochs,
):
    src_from, _, _ = surface_example('recording')
    src_to, _, _ = surface_example('fsaverage')
    filters = object()

    if as_epochs:
        data = epochs_small
        apply = Mock(return_value=['stc-1', 'stc-2'])
        monkeypatch.setattr(
            mne.beamformer,
            'apply_lcmv_epochs',
            apply,
        )
    else:
        data = raw_small
        apply = Mock(return_value='stc')
        monkeypatch.setattr(
            mne.beamformer,
            'apply_lcmv_raw',
            apply,
        )

    target = tmp_path / 'fsaverage-ico-4-src.fif'
    monkeypatch.setattr(
        mne,
        'read_source_spaces',
        Mock(return_value=src_to),
    )

    morph = Mock()
    if as_epochs:
        morph.apply.side_effect = ['morphed-1', 'morphed-2']
    else:
        morph.apply.return_value = 'morphed'
    compute_morph = Mock(return_value=morph)
    monkeypatch.setattr(mne, 'compute_source_morph', compute_morph)

    info = {
        'ForwardModel': {
            'fwd_info': {
                'fwd': {'src': src_from},
                'subject_id_freesurfer': 'recording',
                'subjects_dir': str(tmp_path),
                'template_src': str(target),
            }
        }
    }

    result = sr.SourceReconstruction(filters=filters).run(
        data,
        info,
    )

    compute_morph.assert_called_once_with(
        src_from,
        subject_from='recording',
        subject_to='fsaverage',
        src_to=src_to,
        subjects_dir=Path(tmp_path),
    )

    if as_epochs:
        assert morph.apply.call_args_list == [
            call('stc-1'),
            call('stc-2'),
        ]
        assert result['data']['label_tc'] == [
            'morphed-1',
            'morphed-2',
        ]
    else:
        morph.apply.assert_called_once_with('stc')
        assert result['data']['label_tc'] == 'morphed'

    assert result['stc_info']['subject_id'] == 'fsaverage'
    assert result['stc_info']['source'] == 'surface'


@pytest.mark.parametrize('as_list', [False, True])
def test_surface_parcellation_uses_supplied_src(
    tmp_path,
    monkeypatch,
    as_list,
):
    subject = 'fsaverage'
    subject_path = make_subject(tmp_path / subject)
    for hemi in ('lh', 'rh'):
        (
            subject_path / 'label' / f'{hemi}.aparc.annot'
        ).touch()

    src, stc, labels = surface_example(subject)
    monkeypatch.setattr(
        mne,
        'read_labels_from_annot',
        Mock(return_value=labels),
    )

    estimates = [stc, stc * 2] if as_list else stc
    result = sr.src2parc(
        estimates,
        src=src,
        fs=100.0,
        subject_id=subject,
        subjects_dir=tmp_path,
        atlas='dk',
        label_mode='mean_flip',
    )

    expected = np.array(
        [[2.0, 3.0, 4.0], [12.0, 13.0, 14.0]]
    )
    if as_list:
        np.testing.assert_allclose(
            result['label_tc'][0],
            expected,
        )
        np.testing.assert_allclose(
            result['label_tc'][1],
            2 * expected,
        )
    else:
        np.testing.assert_allclose(
            result['label_tc'],
            expected,
        )

    assert result['lh'] == [True, False]
    assert result['rh'] == [False, True]


def test_volume_parcellation_uses_subject_atlas_and_auto_mode(
    tmp_path,
    monkeypatch,
):
    subject = 'fsaverage'
    subject_path = make_subject(tmp_path / subject)
    atlas_path = subject_path / 'mri' / 'aparc+aseg.mgz'
    atlas_path.touch()

    src = mne.SourceSpaces(
        [{'type': 'vol', 'subject_his_id': subject}]
    )
    names = [
        'Left-Thalamus-Proper',
        'ctx-lh-test',
        'ctx-rh-test',
    ]
    monkeypatch.setattr(
        mne,
        'get_volume_labels_from_aseg',
        Mock(return_value=names),
    )
    extract = Mock(return_value=np.zeros((3, 10)))
    monkeypatch.setattr(
        mne,
        'extract_label_time_course',
        extract,
    )

    result = sr.src2parc(
        'stc',
        src=src,
        fs=100.0,
        subject_id=subject,
        subjects_dir=tmp_path,
        atlas='dk',
    )

    assert extract.call_args.args[1] == str(atlas_path)
    assert extract.call_args.args[2] is src
    assert extract.call_args.kwargs['mode'] == 'auto'
    assert result['ctx_logical'] == [False, True, True]
    assert result['sctx_labels'] == ['Left-Thalamus-Proper']
