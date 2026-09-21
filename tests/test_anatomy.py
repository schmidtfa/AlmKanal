from pathlib import Path
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from almkanal.almkanal_steps import headmodel_utils as hm
from almkanal.almkanal_steps import src_recon_utils as sr


@pytest.fixture
def raw_small():
    info = mne.create_info(
        ['MEG 0111'],
        100.0,
        ch_types='mag',
    )
    return mne.io.RawArray(
        np.zeros((1, 500)),
        info,
        verbose=False,
    )


def make_subject(path):
    for folder in ('mri', 'surf', 'bem', 'label'):
        (path / folder).mkdir(
            parents=True,
            exist_ok=True,
        )
    return path


def surface_example(subject):
    vertices = [
        np.array([0, 1]),
        np.array([0, 1]),
    ]

    src = mne.SourceSpaces([
        {
            'type': 'surf',
            'subject_his_id': subject,
            'vertno': vertices[hemi],
            'nuse': 2,
            'nn': np.array([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]),
        }
        for hemi in range(2)
    ])

    stc = mne.SourceEstimate(
        np.array([
            [1., 2., 3.],
            [3., 4., 5.],
            [10., 11., 12.],
            [14., 15., 16.],
        ]),
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


def test_existing_fsaverage_is_still_checked(
    raw_small,
    tmp_path,
    monkeypatch,
):
    """An existing directory does not mean a complete download."""

    fs_dir = tmp_path / 'freesurfer'
    subject = make_subject(fs_dir / 'fsaverage')

    # This local ico4 file does not establish that the other
    # fsaverage files exist.
    (
        subject
        / 'bem'
        / 'fsaverage-ico-4-src.fif'
    ).touch()

    fetch = Mock()

    monkeypatch.setattr(
        mne.datasets,
        'fetch_fsaverage',
        fetch,
    )
    monkeypatch.setattr(
        hm,
        'compute_headmodel',
        Mock(return_value=('trans', None)),
    )
    monkeypatch.setattr(
        hm,
        'make_fwd',
        Mock(return_value={'src': 'src'}),
    )

    hm.ForwardModel(
        'recording',
        tmp_path,
    ).run(raw_small, info={})

    fetch.assert_called_once_with(
        subjects_dir=fs_dir,
    )


@pytest.mark.parametrize(
    'mode',
    ['template', 'individual', 'local'],
)
def test_anatomy_routing_and_transform_reuse(
    raw_small,
    tmp_path,
    monkeypatch,
    mode,
):
    """Save with redo_hdm=True; reload the same transform with False."""

    workspace = tmp_path / 'work'
    explicit_mri = make_subject(
        tmp_path / 'anatomy' / 'fs-recording'
    )

    mri_path = (
        explicit_mri
        if mode == 'individual'
        else None
    )
    use_template = mode == 'template'

    cache_id = (
        'recording_from_template'
        if use_template
        else 'recording'
    )

    fs_dir = (
        explicit_mri.parent
        if mri_path
        else workspace / 'freesurfer'
    )
    fs_subject = (
        explicit_mri.name
        if mri_path
        else cache_id
    )

    if use_template:
        average = make_subject(fs_dir / 'fsaverage')
        (
            average
            / 'bem'
            / 'fsaverage-ico-4-src.fif'
        ).touch()
    else:
        make_subject(fs_dir / fs_subject)

    matrix = np.eye(4)
    matrix[0, 3] = 0.01

    trans = mne.transforms.Transform(
        'head',
        'mri',
        matrix,
    )

    coreg = Mock(
        trans=trans,
        scale=np.ones(3),
    )
    coreg.compute_dig_mri_distances.return_value = (
        np.array([0.001])
    )

    factory = Mock(return_value=coreg)
    scale = Mock()
    fetch = Mock()
    forward = Mock(
        return_value={'src': 'source-space'}
    )

    monkeypatch.setattr(
        hm,
        'Coregistration',
        factory,
    )
    monkeypatch.setattr(
        mne.coreg,
        'scale_mri',
        scale,
    )
    monkeypatch.setattr(
        mne.datasets,
        'fetch_fsaverage',
        fetch,
    )
    monkeypatch.setattr(
        hm,
        'plot_head_model',
        Mock(return_value=None),
    )
    monkeypatch.setattr(
        hm,
        'make_fwd',
        forward,
    )

    # Individual mode deliberately leaves template_mri=True:
    # mri_path must override it.
    step = hm.ForwardModel(
        'recording',
        workspace,
        template_mri=(mode != 'local'),
        mri_path=mri_path,
    )

    result = step.run(
        raw_small,
        info={'Picks': {'meg': True}},
    )

    assert factory.call_args.kwargs['subject'] == (
        'fsaverage' if use_template else fs_subject
    )
    assert (
        factory.call_args.kwargs['subjects_dir']
        == fs_dir
    )

    coreg.set_scale_mode.assert_called_once_with(
        '3-axis' if use_template else None
    )

    if use_template:
        assert scale.call_args.args == (
            'fsaverage',
            fs_subject,
        )
    else:
        scale.assert_not_called()
        fetch.assert_not_called()

    assert step.pick_dict is None

    fwd_info = result['fwd_info']

    assert (
        fwd_info['subject_id_freesurfer']
        == fs_subject
    )
    assert Path(fwd_info['subjects_dir']) == fs_dir
    assert fwd_info['subject_dir'] == workspace
    assert fwd_info['template_mri'] == use_template

    trans_file = (
        workspace
        / 'headmodels'
        / cache_id
        / f'{cache_id}-trans.fif'
    )

    np.testing.assert_allclose(
        mne.read_trans(trans_file)['trans'],
        matrix,
    )

    assert (
        forward.call_args.kwargs['subject_id']
        == cache_id
    )

    # No optimization should be rerun, and the cached matrix
    # must reach make_fwd.
    factory.reset_mock()
    step.redo_hdm = False

    step.run(raw_small, info={})

    factory.assert_not_called()

    np.testing.assert_allclose(
        forward.call_args.kwargs['fname_trans']['trans'],
        matrix,
    )


@pytest.mark.parametrize(
    'source',
    ['surface', 'volume'],
)
def test_individual_forward_uses_individual_anatomy(
    raw_small,
    tmp_path,
    monkeypatch,
    source,
):
    subject_path = make_subject(
        tmp_path / 'anatomy' / 'fs-recording'
    )
    (
        subject_path
        / 'bem'
        / 'inner_skull.surf'
    ).touch()

    model = Mock(return_value='model')
    solution = Mock(return_value='bem')
    surface = Mock(return_value='surface-src')
    volume = Mock(return_value='volume-src')
    forward = Mock(return_value='forward')

    monkeypatch.setattr(mne, 'make_bem_model', model)
    monkeypatch.setattr(mne, 'make_bem_solution', solution)
    monkeypatch.setattr(mne, 'setup_source_space', surface)
    monkeypatch.setattr(
        mne,
        'setup_volume_source_space',
        volume,
    )
    monkeypatch.setattr(
        mne,
        'make_forward_solution',
        forward,
    )

    result = hm.make_fwd(
        raw_small.info,
        source,
        'recording-trans.fif',
        tmp_path / 'work',
        'recording',
        template_mri=True,
        mri_path=subject_path,
    )

    assert result == 'forward'

    model.assert_called_once_with(
        'fs-recording',
        ico=4,
        conductivity=(0.3,),
        subjects_dir=subject_path.parent,
    )

    setup = (
        surface if source == 'surface' else volume
    )

    assert setup.call_args.args == ('fs-recording',)
    assert (
        setup.call_args.kwargs['subjects_dir']
        == subject_path.parent
    )

    if source == 'volume':
        surface.assert_not_called()

        assert setup.call_args.kwargs['mri'] == (
            subject_path / 'mri' / 'T1.mgz'
        )
        assert (
            setup.call_args.kwargs['add_interpolator']
            is True
        )
    else:
        volume.assert_not_called()
        assert (
            setup.call_args.kwargs['spacing']
            == 'oct6'
        )

    assert (
        forward.call_args.kwargs['src']
        == f'{source}-src'
    )

    # Preserve the existing individual-MRI MEG-only scope.
    assert forward.call_args.kwargs['eeg'] is False


@pytest.mark.parametrize(
    'source,suffix',
    [
        ('surface', 'ico-4'),
        ('volume', 'vol-5'),
    ],
)
def test_template_forward_does_not_append_suffix_twice(
    raw_small,
    tmp_path,
    monkeypatch,
    source,
    suffix,
):
    subject = 'recording_from_template'
    bem_dir = (
        tmp_path
        / 'freesurfer'
        / subject
        / 'bem'
    )

    solution = Mock(return_value='bem')
    forward = Mock(return_value='forward')

    monkeypatch.setattr(
        mne,
        'make_bem_solution',
        solution,
    )
    monkeypatch.setattr(
        mne,
        'make_forward_solution',
        forward,
    )

    hm.make_fwd(
        raw_small.info,
        source,
        'trans.fif',
        tmp_path,
        subject,
        template_mri=True,
    )

    assert solution.call_args.args[0] == (
        bem_dir
        / f'{subject}-5120-5120-5120-bem.fif'
    )

    assert forward.call_args.kwargs['src'] == (
        bem_dir / f'{subject}-{suffix}-src.fif'
    )


@pytest.mark.parametrize('as_epochs', [False, True])
def test_reconstruction_uses_current_forward_without_retaining_state(
    raw_small,
    tmp_path,
    monkeypatch,
    as_epochs,
):
    data = raw_small

    if as_epochs:
        data = mne.EpochsArray(
            np.zeros((2, 1, 20)),
            raw_small.info,
            verbose=False,
        )

    estimates = (
        ['estimate-1', 'estimate-2']
        if as_epochs
        else 'estimate'
    )

    apply = Mock(return_value=estimates)
    method = (
        'apply_lcmv_epochs'
        if as_epochs
        else 'apply_lcmv_raw'
    )

    monkeypatch.setattr(
        mne.beamformer,
        method,
        apply,
    )

    parcellate = Mock(
        side_effect=lambda stc, **kwargs: {
            'label_tc': stc,
            'fs': kwargs['fs'],
        }
    )
    monkeypatch.setattr(
        sr,
        'src2parc',
        parcellate,
    )

    step = sr.SourceReconstruction(
        return_parc=True,
        atlas='dk',
    )

    for subject, kind in [
        ('fs-a', 'surface'),
        ('fs-b', 'volume'),
    ]:
        if kind == 'surface':
            src = surface_example(subject)[0]
        else:
            src = mne.SourceSpaces([
                {
                    'type': 'vol',
                    'subject_his_id': subject,
                },
            ])

        filters = object()

        info = {
            'SpatialFilter': {
                'spatial_filter_info': {
                    'filters': filters,
                },
            },
            'ForwardModel': {
                'fwd_info': {
                    'source_type': kind,
                    'subject_id_freesurfer': subject,
                    'subjects_dir': str(tmp_path),
                    'fwd': {'src': src},
                },
            },
        }

        result = step.run(data, info)

        assert apply.call_args.args[1] is filters
        assert parcellate.call_args.args[0] is estimates
        assert parcellate.call_args.kwargs['src'] is src
        assert (
            parcellate.call_args.kwargs['subject_id']
            == subject
        )
        assert (
            parcellate.call_args.kwargs['source']
            == kind
        )
        assert (
            result['stc_info']['subject_id']
            == subject
        )

        if as_epochs:
            assert (
                result['data']['metadata']
                is data.metadata
            )

    assert step.filters is None
    assert step.subject_id is None
    assert step.subjects_dir is None
    assert step.source is None
    assert step.src is None


@pytest.mark.parametrize('as_list', [False, True])
@pytest.mark.parametrize('use_workspace', [False, True])
def test_surface_parcellation_uses_supplied_src(
    tmp_path,
    monkeypatch,
    as_list,
    use_workspace,
):
    subject = 'fs-a'
    fs_dir = tmp_path / 'freesurfer'
    subject_path = make_subject(fs_dir / subject)

    for hemi in ('lh', 'rh'):
        (
            subject_path
            / 'label'
            / f'{hemi}.aparc.annot'
        ).touch()

    src, stc, labels = surface_example(subject)

    read_labels = Mock(return_value=labels)
    read_src = Mock(
        side_effect=AssertionError(
            'Must use the supplied source space.'
        )
    )

    monkeypatch.setattr(
        mne,
        'read_labels_from_annot',
        read_labels,
    )
    monkeypatch.setattr(
        mne,
        'read_source_spaces',
        read_src,
    )

    estimates = (
        [stc, stc * 2] if as_list else stc
    )

    result = sr.src2parc(
        estimates,
        100.0,
        subject,
        tmp_path if use_workspace else fs_dir,
        source='surface',
        atlas='dk',
        label_mode='mean_flip',
        src=src,
    )

    read_src.assert_not_called()
    read_labels.assert_called_once_with(
        subject,
        parc='aparc',
        subjects_dir=fs_dir,
    )

    # Actual MNE mean_flip extraction with aligned normals.
    expected = np.array([
        [2., 3., 4.],
        [12., 13., 14.],
    ])

    if as_list:
        assert len(result['label_tc']) == 2

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


@pytest.mark.parametrize(
    'mismatch',
    ['subject', 'kind'],
)
def test_src2parc_rejects_inconsistent_anatomy(
    tmp_path,
    mismatch,
):
    make_subject(tmp_path / 'fs-a')

    src, stc, _ = surface_example(
        'fs-b' if mismatch == 'subject' else 'fs-a'
    )

    source = (
        'volume' if mismatch == 'kind' else 'surface'
    )
    message = (
        'Source space subjects'
        if mismatch == 'subject'
        else 'does not match'
    )

    with pytest.raises(ValueError, match=message):
        sr.src2parc(
            stc,
            100.0,
            'fs-a',
            tmp_path,
            source=source,
            atlas='dk',
            src=src,
        )


def test_volume_parcellation_keeps_subject_atlas_and_auto_mode(
    tmp_path,
    monkeypatch,
):
    subject_path = make_subject(tmp_path / 'fs-a')
    atlas_path = (
        subject_path / 'mri' / 'aparc+aseg.mgz'
    )
    atlas_path.touch()

    src = mne.SourceSpaces([
        {
            'type': 'vol',
            'subject_his_id': 'fs-a',
        },
    ])

    names = [
        'Left-Thalamus-Proper',
        'ctx-lh-test',
        'ctx-rh-test',
    ]

    read_labels = Mock(return_value=names)
    extract = Mock(
        return_value=np.zeros((3, 10))
    )

    monkeypatch.setattr(
        mne,
        'get_volume_labels_from_aseg',
        read_labels,
    )
    monkeypatch.setattr(
        mne,
        'extract_label_time_course',
        extract,
    )

    result = sr.src2parc(
        'stc',
        100.0,
        'fs-a',
        tmp_path,
        source='volume',
        atlas='dk',
        src=src,
    )

    read_labels.assert_called_once_with(atlas_path)

    assert extract.call_args.args[1] == str(atlas_path)
    assert extract.call_args.args[2] is src
    assert extract.call_args.kwargs['mode'] == 'auto'

    assert result['ctx_logical'] == [
        False,
        True,
        True,
    ]
    assert result['sctx_labels'] == [
        'Left-Thalamus-Proper',
    ]


def test_external_filters_without_parcellation_need_no_anatomy(
    raw_small,
    monkeypatch,
):
    filters = object()
    estimate = object()

    apply = Mock(return_value=estimate)
    monkeypatch.setattr(
        mne.beamformer,
        'apply_lcmv_raw',
        apply,
    )

    result = sr.SourceReconstruction(
        filters=filters,
    ).run(raw_small, info={})

    assert apply.call_args.args[1] is filters
    assert result['data']['label_tc'] is estimate
    assert result['stc_info']['subject_id'] is None