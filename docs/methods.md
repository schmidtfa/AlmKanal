## From execution to scientific prose

AlmKanal records processing information as steps run. Export that history to JSON and pass it to the Methods renderer. Jinja templates supply the prose; the saved settings supply the values. This is template-based reporting, not an AI service invoked at runtime.

```python
from almkanal import preprocessing_report

# Run your configured pipeline first.
processed, qc_report = pipeline.run(raw.copy())
pipeline.generate_json("sub-01_preprocessing.json")

preprocessing_report(
    files=["sub-01_preprocessing.json"],
    out_path="methods.md",
)
```

The [quick start](quickstart.md) provides a complete example, including the input data.

## Three outputs, three purposes

| Output | Purpose |
| --- | --- |
| MNE HTML report returned by `run()` | Visual and tabular quality control supplied by the steps |
| JSON from `generate_json()` | Recorded processing information for a completed run |
| Text from `preprocessing_report()` | A template-rendered draft of the Methods description |

An HTML Methods export is not the same thing as the MNE HTML QC report. Retain both when they serve your workflow.

## Combine compatible histories

```python
from pathlib import Path
from almkanal import preprocessing_report

files = sorted(Path("histories").glob("*_preprocessing.json"))
if not files:
    raise FileNotFoundError("No preprocessing histories found.")

preprocessing_report(files=files, out_path="study_methods.md")
```

The context builder checks that input files share an ordered sequence of steps and matching settings selected by each step's specification. Result summaries are handled separately. Settings mismatches raise errors rather than silently merging distinct protocols.

The context's `n_subjects` value counts input files. Do not assume this equals participants when several runs or sessions belong to each person. Group and describe your data accordingly.

## Choose an output format

The output suffix normally determines the target format; `to=` can override it.

```python
# Markdown: no Pandoc executable required.
preprocessing_report(files=files, out_path="methods.md")

# These conversions require the Pandoc executable.
preprocessing_report(files=files, out_path="methods.html", to="html")
preprocessing_report(files=files, out_path="methods.docx", to="docx")
```

PDF additionally requires a PDF engine configured for Pandoc. `metadata`, `extra_args`, and `pandoc_path` allow conversion-specific settings. See the [reporting API](api-reporting.md) for the signature.

## Customize the wording

Pass `template_dir` to override templates while retaining packaged defaults for other steps. Custom scientific operations should have a matching metadata specification and Methods template. The [custom-reporting guide](custom-reporting.md) walks through this using the custom notch-filter step.

## Before using the text

Review the generated description against your code and outputs. Include preprocessing that happened outside AlmKanal, justify study-specific decisions, check citations, and describe any excluded participants or trials. Verify that custom steps produced meaningful prose rather than only a generic fallback.

The generator helps keep code and documentation close together. It does not certify a Methods section as complete or publication-ready.
