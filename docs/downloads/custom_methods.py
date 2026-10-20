"""Run a custom step and render its matching Jinja Methods partial.

Keep this file beside quickstart.py, notch_step.py, and templates/.
"""

from pathlib import Path

from notch_step import NotchAtFrequency
from quickstart import make_raw

from almkanal import AlmKanal, Filter, Resample, preprocessing_report
from almkanal.report.stepspecs import StepSpec, keys_selector, register


def main() -> None:
    out = Path('almkanal-custom-demo')
    out.mkdir(exist_ok=True)
    pipeline = AlmKanal(
        [
            NotchAtFrequency(frequency=50.0),
            Filter(highpass=1.0, lowpass=40.0),
            Resample(sfreq=200),
        ]
    )
    _, report = pipeline.run(make_raw())
    report.save(out / 'quality-control.html', overwrite=True, open_browser=False)
    history = out / 'preprocessing.json'
    pipeline.generate_json(str(history))
    register(
        'NotchAtFrequency',
        StepSpec(settings_fn=keys_selector('frequency_hz')),
    )
    methods = preprocessing_report(
        files=[history],
        out_path=out / 'methods.md',
        template_dir=Path(__file__).parent / 'templates',
    )
    text = methods.read_text(encoding='utf-8')
    if 'notch filter' not in text.lower():
        raise AssertionError('Custom Methods partial was not included.')
    print(text)
    print(f'Outputs saved in {out.resolve()}')


if __name__ == '__main__':
    main()
