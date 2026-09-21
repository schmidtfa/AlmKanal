from types import SimpleNamespace

import pytest

from almkanal.report.rendering import (
    make_env,
    render_methods,
    render_methods_to_file,
)


def _ctx():
    return SimpleNamespace(
        n_subjects=2,
        files=['sub-01.json', 'sub-02.json'],
        ordered_steps=['Filter'],
        steps={},
    )


def test_kv_filter():
    env = make_env()
    kv = env.filters['kv']

    assert kv(
        {
            'a': 1,
            'b': None,
            'c': 3,
        }
    ) == 'a=1; c=3'

    assert kv(
        {
            'a': 1,
            'b': 2,
        },
        omit={'b'},
    ) == 'a=1'

    assert kv(
        {
            'a': None,
        }
    ) == '—'


def test_custom_template_overrides_packaged_template(
    tmp_path,
):
    template = (
        tmp_path
        / 'methods_master.j2'
    )

    template.write_text(
        'Subjects={{ n_subjects }}',
        encoding='utf-8',
    )

    text = render_methods(
        _ctx(),
        template_dir=tmp_path,
    )

    assert text == 'Subjects=2'


def test_missing_template_raises(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match='does_not_exist.j2',
    ):
        render_methods(
            _ctx(),
            template_dir=tmp_path,
            master_template='does_not_exist.j2',
        )


def test_render_methods_to_file(
    tmp_path,
):
    template = (
        tmp_path
        / 'methods_master.j2'
    )

    template.write_text(
        'N={{ n_subjects }}',
        encoding='utf-8',
    )

    out = (
        tmp_path
        / 'nested'
        / 'methods.md'
    )

    result = render_methods_to_file(
        _ctx(),
        out,
        template_dir=tmp_path,
    )

    assert result == out
    assert out.read_text(
        encoding='utf-8'
    ) == 'N=2'