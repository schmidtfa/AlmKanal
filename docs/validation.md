## Declare the relationship between steps

Each step can define `must_be_before`, `must_be_after`, and `allow_repeated`. Ordering constraints refer to **class names as strings**, not object labels.

A constraint is checked when the referenced step occurs in the same pipeline. It does not require that step to be present. This allows partial workflows to start from previously processed data.

| Declaration | Meaning |
| --- | --- |
| `must_be_before = ("SourceReconstruction",)` | When source reconstruction is present, a matching step must occur later |
| `must_be_after = ("Epochs",)` | When epoch creation is present, a matching step must occur earlier |
| `allow_repeated = False` | Reject more than one instance of the same step type |

The current ordering check requires a matching occurrence on the requested side. It is not an all-pairs validator for pipelines containing repeated step types.

## Inspect a concrete rejection

`Resample` declares that an `Epochs` step, when present, should precede it. This example demonstrates constructor validation and does not need recording data:

```python
from almkanal import AlmKanal, Epochs, Resample

try:
    AlmKanal([
        Resample(sfreq=200),
        Epochs(),
    ])
except ValueError as error:
    print(error)
```

The proposed order is rejected while building the pipeline. Reordering alone does not supply events or solve all input requirements: the corrected workflow still needs appropriate data and configuration.

Similarly, `Maxwell` declares that it must precede `ICA`, `ForwardModel`, `SpatialFilter`, and `SourceReconstruction` when those steps are present.

## Separate ordering checks from data checks

Constructing `AlmKanal(...)` checks declared sequence and repetition rules. During execution, individual steps may check data-specific requirements. For example, `Resample` compares the input's recorded low-pass frequency with half the requested sampling rate and rejects an incompatible downsampling request.

## Add rules to your own step

```python
from attrs import define, field
from almkanal import AlmKanalStep

@define
class MySensorStep(AlmKanalStep):
    must_be_before: tuple[str, ...] = field(
        default=("SourceReconstruction",), init=False
    )
    allow_repeated: bool = field(default=False, init=False)

    # Implement run() and reports() before using this step.
```

This is a declaration example, not a complete processing step. See the [custom-step tutorial](custom-steps.md) for a complete implementation.

> **Scope of the checks.** Valid ordering does not guarantee methodological appropriateness, sufficient metadata, compatible hardware, or a successful run. Keep acquisition-specific and scientific checks in your workflow.
