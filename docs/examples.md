## Synthetic-data quick start

[Download quickstart.py](downloads/quickstart.py). It creates a reproducible synthetic EEG recording, runs filtering and resampling, saves the processed data and QC report, and renders Markdown Methods text.

```bash
python quickstart.py
```

No study-data download is needed. The script creates `almkanal-demo/` in the current directory.

## Constructor validation

[Download validation_demo.py](downloads/validation_demo.py). It intentionally builds an invalid step sequence and checks that construction raises an error. No M/EEG recording is needed.

```bash
python validation_demo.py
```

## A custom processing and reporting extension

Download the [complete example bundle](downloads/examples.zip) for the custom-step workflow. It contains `notch_step.py`, `custom_methods.py`, the quick-start helper, and a matching template directory. Extract the files together, then run:

```bash
python custom_methods.py
```

The script runs the teaching notch-filter step with built-in steps, registers its reporting specification, and writes an overridden Methods description. Expand the metadata and template before using a custom scientific procedure in a publication.

## Your own recordings

The eye-tracking and audio guides use filenames, trigger codes, and calibration values that must be replaced with study-specific inputs. They are not bundled as plug-and-play tests because representative VPixx and recorded-audio files are not part of this documentation package.

## Validation status

The documentation build checks Python syntax for complete example scripts. Their APIs were cross-checked against the public source, but these AlmKanal examples were **not executed end to end in the site-authoring environment**. Run them against your installed package revision before adopting them or publishing example outputs. The bundle includes a separate optional example-check command for that purpose.
