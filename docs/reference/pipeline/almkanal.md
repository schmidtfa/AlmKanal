# AlmKanal

`AlmKanal` combines an ordered list of preprocessing steps into a pipeline. It checks the supported ordering constraints when the pipeline is created and runs the steps on MNE-Python data.

Create a pipeline with `AlmKanal(steps=[...])`, then pass the data to `pipeline.run(data)`. The method returns the processed data and an MNE report.

::: almkanal.almkanal.AlmKanal
    options:
      members:
        - run
        - generate_json
      show_root_heading: true
      show_source: false
