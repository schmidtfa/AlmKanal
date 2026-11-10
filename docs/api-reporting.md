# Methods reporting

AlmKanal can automatically generate draft Methods text from the processing history saved after running a pipeline. The public interface for this is `preprocessing_report`, which uses Jinja templates to describe the preprocessing steps.

## Example

After running a pipeline and saving its processing history:

```python
from almkanal import preprocessing_report

pipeline.generate_json("history.json")

methods_path = preprocessing_report(
    files=["history.json"],
    out_path="methods.md",
)
```

The example writes the generated text to `methods.md`. Other output formats, such as Word documents or PDFs, require Pandoc (and, for PDF, a suitable PDF engine).

## Custom Jinja templates

You can also use your own Jinja templates to customize the generated descriptions:

```python
methods_path = preprocessing_report(
    files=["history.json"],
    out_path="methods.md",
    template_dir="my_templates",
)
```

See [Custom Methods templates](custom-reporting.md) for instructions on creating and using your own templates.

!!! note
    Automatically generated Methods text should always be reviewed before use in a publication. Processing performed before the data entered AlmKanal needs to be documented separately.

## API reference

::: almkanal.report.exporting.preprocessing_report
