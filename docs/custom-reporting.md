## Match three names

A custom reporting extension connects the **step class name**, the **registered `StepSpec`**, and the **Jinja partial filename**. This example continues the [NotchAtFrequency tutorial](custom-steps.md).

```text
NotchAtFrequency
    -> StepSpec registered as "NotchAtFrequency"
    -> templates/steps/NotchAtFrequency.j2
```

The example's metadata contains one nested `notch_info` dictionary, so the context builder extracts it before applying the specification.

## Register a specification

Run this registration in the same Python process as your Methods export:

```python
from almkanal.report.stepspecs import StepSpec, keys_selector, register

register(
    "NotchAtFrequency",
    StepSpec(settings_fn=keys_selector("frequency_hz")),
)
```

`settings_fn` selects settings that must match across histories. A `summarize_fn` can separately summarize per-recording results. Do not discard meaningful protocol differences just to pass a consistency check.

For a reusable extension, put registration code in an importable module or package. The reporting API's `step_packages` argument can load it alongside the built-in specifications.

## Add the prose template

Create `my_templates/steps/NotchAtFrequency.j2`:

```jinja
A notch filter was applied to the continuous recordings at
{{ s.settings.frequency_hz }} Hz using MNE-Python.
```

The packaged master template passes each step context as `s`. Its settings and summarized results are accessible as `s.settings` and `s.results`. The filename is case-sensitive.

This is intentionally minimal wording for a minimal teaching step. Before publication, extend the step metadata and prose to include the filter characteristics, relevant citations, and other details needed to reproduce the operation.

## Render with your override

```python
from almkanal import preprocessing_report

preprocessing_report(
    files=["preprocessing.json"],
    out_path="methods.md",
    template_dir="my_templates",
)
```

Your directory is checked first. Packaged templates remain available for other steps. Steps without a named partial use the packaged fallback template; that fallback is not a substitute for a carefully written scientific description.

## Catch missing information

The renderer uses Jinja's `StrictUndefined`. Referring to metadata that does not exist should therefore fail rather than quietly producing a blank value. Test both a valid history and one with a missing required setting.

A custom master template can be selected with `master_template`. Inspect the packaged master before replacing it, especially when retaining step ordering and references.

The [examples page](examples.md) includes a registration script and template directory that can be copied together.
