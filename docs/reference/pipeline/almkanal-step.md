# AlmKanalStep

`AlmKanalStep` is the base class for custom preprocessing steps. Subclasses implement `run(data, info)` to process the data and `reports(data, report, info)` to update the MNE report.

The `run` method must return a dictionary containing a `data` key. Other entries can be used to record information about the processing step. The `must_be_before`, `must_be_after`, and `allow_repeated` attributes control the supported ordering and repetition checks.

::: almkanal.almkanal.AlmKanalStep
    options:
      members:
        - check_can_run
        - run
        - reports
      show_root_heading: true
      show_source: false
