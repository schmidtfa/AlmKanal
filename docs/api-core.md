## AlmKanal

```python
from almkanal import AlmKanal

pipeline = AlmKanal(steps, pick_params=None)
```

`steps` is an ordered list of `AlmKanalStep` objects. Construction checks their declared ordering and repetition rules. `pick_params`, when provided, controls channel-type selection before processing.

### run

```python
processed, report = pipeline.run(data)
```

Runs steps sequentially and returns the final step's data output and an `mne.Report`. Inputs may be `Raw`, `Epochs`, or homogeneous lists, subject to step compatibility.

The processing history is reset at the start of each run. The callable shorthand `pipeline(data)` also delegates to `run()` and returns its pair of outputs.

### generate_json

```python
pipeline.generate_json(path='/my/path/to/file.json', max_elements=50)
```

Builds a JSON-compatible history dictionary; optionally writes it to `path`. Call it after a successful run. `max_elements` limits sequence serialization and can omit large contents from the compact record. Create the destination directory before writing a file there.

### info

`pipeline.info` contains the recorded step history and context. For durable reporting, save a JSON history for each run rather than relying on mutable in-memory state using: 

```python
pipeline.generate_json(path='/my/path/to/file.json', max_elements=50)
```

## AlmKanalStep

```python
from almkanal import AlmKanalStep
```

Subclass the base `AlmKanalStep` class and implement `run(data, info)` and `reports(data, report, info)`. The first returns a dictionary with a `"data"` entry plus metadata; the second receives the processed data and updated context and may add report content.

Use `must_be_before`, `must_be_after`, and `allow_repeated` for supported construction-time checks. See [validation](validation.md) for their exact scope.


## Further documentation

For detailed documentation of individual classes and parameters, see the automatically generated API reference for AlmKanal, AlmKanalStep, and the preprocessing steps. The API reference is generated directly from the Python docstrings.

For practical examples, see VPixx alignment, WAV alignment, and Methods reporting.