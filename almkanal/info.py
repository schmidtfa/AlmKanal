from typing import Literal, overload

from attrs import define, field


@define
class StepInfo:
    step: str
    info: dict


@define
class AlmKanalInfo:
    pick_params: dict | None = None
    processing_history: list[StepInfo] = field(factory=list)

    def add(self, step_info: StepInfo) -> None:
        self.processing_history.append(step_info)

    def get_steps(self, step: str) -> list[StepInfo]:
        return [step_info for step_info in self.processing_history if step_info.step == step]

    def get_step(
        self,
        step: str,
        occurrence: int = -1,
    ) -> StepInfo | None:
        steps = self.get_steps(step)

        if not steps:
            return None

        return steps[occurrence]

    @overload
    def get_step_info(
        self,
        step: str,
        occurrence: int = -1,
        *,
        required: Literal[True],
    ) -> dict: ...

    @overload
    def get_step_info(
        self,
        step: str,
        occurrence: int = -1,
        *,
        required: Literal[False] = False,
    ) -> dict | None: ...

    def get_step_info(
        self,
        step: str,
        occurrence: int = -1,
        *,
        required: bool = False,
    ) -> dict | None:
        step_info = self.get_step(step, occurrence)

        if step_info is None:
            if required:
                raise ValueError(f'Required processing step {step!r} was not found.')
            return None

        return step_info.info
