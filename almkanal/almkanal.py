# %%
from copy import deepcopy

import mne
from attrs import define, field


# %%
@define
class AlmKanalStep:
    """This is the base class for all almkanal steps."""

    must_be_before: tuple = field(default=None, init=False)
    must_be_after: tuple = field(default=None, init=False)

    def _check_dependencies(self, steps: list['AlmKanalStep']) -> None:
        pre: list[AlmKanalStep] = []
        post: list[AlmKanalStep] = []

        found_me = False
        for cur_step in steps:
            if cur_step is self:
                found_me = True
                continue

            if not found_me:
                pre.append(cur_step)
            else:
                post.append(cur_step)

        # For each dependency in must_be_before, check only if it is present somewhere.
        for dep_name in self.must_be_before or ():
            # If any step in the entire pipeline is of type dep_name:
            if any(step.__class__.__name__ == dep_name for step in steps) and not any(
                step.__class__.__name__ == dep_name for step in post
            ):
                # Then at least one such step must be in the post list.
                raise ValueError(
                    f'Dependency Error: {self.__class__.__name__} expects that, '
                    f"if a step of type '{dep_name}' is present in the pipeline, "
                    'it should follow this step. However, no such step was found after it.\n'
                    f'Current pipeline order: {[step.__class__.__name__ for step in steps]}'
                )

        # For each dependency in must_be_later, check only if it is present somewhere.
        for dep_name in self.must_be_after or ():
            if any(step.__class__.__name__ == dep_name for step in steps) and not any(
                step.__class__.__name__ == dep_name for step in pre
            ):
                # Then at least one such step must be in the pre list.
                raise ValueError(
                    f'Dependency Error: {self.__class__.__name__} expects that, '
                    f"if a step of type '{dep_name}' is present in the pipeline, "
                    'it should precede this step. However, no such step was found before it.\n'
                    f'Current pipeline order: {[step.__class__.__name__ for step in steps]}'
                )

    def check_can_run(self, steps: list['AlmKanalStep']) -> None:
        self._check_dependencies(steps)

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict) -> dict:
        """Apply the processing step to the given data.

        Child classes must implement this method.
        """
        raise NotImplementedError('Child classes should implement the apply() method.')

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: dict) -> None:
        """
        Update the provided mne.Report object based on the current data.

        Child classes must implement this method, and are forced to add something
        to the report using mne.Report's methods (e.g., add_raw or add_epochs).
        """
        raise NotImplementedError('Child classes should implement the reports() method.')


@define
class AlmKanal:  # TODO: Think about Thomas's smart idea of doing this AlmKanal(AlmKanalSteps)
    """
    Initializes the pipeline.

    Parameters:
    data (mne.io.Raw or mne.Epochs): The M/EEG data to process.
    pick_params: A dictionary of parameters for channel/type picking via mne.pick_types.
                   For example: {"meg": True, "eeg": False, "stim": False}
    steps (list of callables): A list of functions that accept and return an M/EEG data object.
    """

    steps: list[AlmKanalStep] = field()
    pick_params: dict = field(default=None)
    info: dict = field(init=False)

    def __attrs_post_init__(self) -> None:
        # Validate ordering constraints for each step.

        for step in self.steps:
            step.check_can_run(self.steps)

        # Save metadata about the steps.
        self.info = {
            'steps_order_valid': True,
            'steps': [step.__class__.__name__ for step in self.steps],
            'steps_info': {},  # This will be updated when the pipeline is run.
        }

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs) -> mne.io.BaseRaw | mne.BaseEpochs:
        """Applies each preprocessing step in sequence and returns the processed data, along with a report."""

        # Apply channel/type picking if parameters are provided.
        if self.pick_params is not None:
            # Use mne.pick_types to get channel indices.
            picks = mne.pick_types(data.info, **self.pick_params)
            data.pick(picks)

        report = mne.Report(title='Pipeline Report')
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='raw_data')
        elif isinstance(data, mne.BaseEpochs):
            report.add_epochs(data, title='raw_data', psd=False)
        else:
            raise ValueError('Input data must be an instance of mne.io.BaseRaw or mne.BaseEpochs')

        current_data = data
        context: dict = {'Picks': self.pick_params}  # Shared context dictionary for passing extra info between steps.
        for step in self.steps:
            result = step.run(current_data, deepcopy(context))
            if not isinstance(result, dict) or 'data' not in result:
                raise ValueError(f"Step {step.__class__.__name__} must return a dictionary with a 'data' key.")
            current_data = result['data']
            # Extract extra information from the step's result and update the shared context.
            extra_info = {k: v for k, v in result.items() if k != 'data'}
            context[step.__class__.__name__] = extra_info
            # Call the step's reports method, passing the updated context.
            step.reports(current_data, report, context)
        # Save the shared context in self.info.
        self.info['steps_info'] = context
        return current_data, report

    def __call__(self, data: mne.io.BaseRaw | mne.BaseEpochs) -> mne.io.BaseRaw | mne.BaseEpochs:
        """Enables the pipeline instance to be called like a function."""
        return self.run(data)
