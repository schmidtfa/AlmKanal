from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from almkanal.report import exporting


@pytest.fixture
def mocked_rendering(
    monkeypatch,
):
    ctx = object()

    build = Mock(
        return_value=ctx
    )

    render = Mock(
        return_value='# Methods\n'
    )

    monkeypatch.setattr(
        exporting,
        'build_context_from_files',
        build,
    )

    monkeypatch.setattr(
        exporting,
        'render_methods',
        render,
    )

    return build, render


def test_preprocessing_report_missing_pandoc(
    tmp_path,
    monkeypatch,
    mocked_rendering,
):
    monkeypatch.setattr(
        exporting.shutil,
        'which',
        lambda command: None,
    )

    with pytest.raises(
        FileNotFoundError,
        match='pandoc not found',
    ):
        exporting.preprocessing_report(
            ['sub-01.json'],
            tmp_path / 'methods.docx',
            to='docx',
        )


def test_preprocessing_report_pandoc_success(
    tmp_path,
    monkeypatch,
    mocked_rendering,
):
    out = (
        tmp_path
        / 'results'
        / 'methods.docx'
    )

    calls = {}

    def fake_run(
        args,
        input,
        check,
    ):
        calls['args'] = args
        calls['input'] = input
        calls['check'] = check

        # Pretend Pandoc generated the file.
        output_file = args[
            args.index('-o') + 1
        ]

        from pathlib import Path

        Path(output_file).write_bytes(
            b'fake docx'
        )

        return SimpleNamespace(
            returncode=0
        )

    monkeypatch.setattr(
        exporting.subprocess,
        'run',
        fake_run,
    )

    result = exporting.preprocessing_report(
        ['sub-01.json'],
        out,
        to='docx',
        pandoc_path='fake-pandoc',
        metadata={
            'title': 'My Methods',
            'author': 'Fabian',
        },
        extra_args=[
            '--standalone',
        ],
    )

    assert result == out
    assert out.exists()

    args = calls['args']

    assert args[:7] == [
        'fake-pandoc',
        '-f',
        'gfm',
        '-t',
        'docx',
        '-o',
        str(out),
    ]

    assert [
        '-M',
        'title=My Methods',
    ] in [
        args[ix:ix + 2]
        for ix in range(len(args) - 1)
    ]

    assert '--standalone' in args

    assert calls['input'] == (
        b'# Methods\n'
    )

    assert calls['check'] is False


def test_preprocessing_report_pandoc_failure(
    tmp_path,
    monkeypatch,
    mocked_rendering,
):
    monkeypatch.setattr(
        exporting.subprocess,
        'run',
        lambda *args, **kwargs:
            SimpleNamespace(
                returncode=2
            ),
    )

    with pytest.raises(
        RuntimeError,
        match='exit code 2',
    ):
        exporting.preprocessing_report(
            ['sub-01.json'],
            tmp_path / 'methods.pdf',
            to='pdf',
            pandoc_path='fake-pandoc',
        )