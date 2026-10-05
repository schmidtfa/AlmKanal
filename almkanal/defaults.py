"""Explicit, context-local lab defaults for constructing reproducible pipelines.

Profiles are copy-protected snapshots. Accessors return independent sections,
so even edits to nested dictionaries cannot change an installed profile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from inspect import Parameter, signature
from numbers import Integral, Real
from types import UnionType
from typing import Any, Literal, ParamSpec, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from attrs import asdict, evolve, field, fields, frozen

from almkanal import _defaults_schema as schema
from almkanal.data_utils.info_generator import build_json
from almkanal.info import StepInfo

P = ParamSpec('P')
R = TypeVar('R')


def _matches_type(value: Any, annotation: Any) -> bool:  # noqa: C901, PLR0911
    """Validate profile types without imposing scientific parameter ranges."""
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        return any(_matches_type(value, option) for option in args)
    if origin is Literal:
        return value in args
    if annotation is float:
        return isinstance(value, Real) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, Integral) and not isinstance(value, bool)
    if origin in (dict, Mapping):
        return isinstance(value, origin) and (
            not args or all(_matches_type(k, args[0]) and _matches_type(v, args[1]) for k, v in value.items())
        )
    if origin in (list, tuple, Sequence):
        if not isinstance(value, origin) or isinstance(value, str | bytes):
            return False
        if not args:
            return True
        if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            return len(value) == len(args) and all(_matches_type(v, a) for v, a in zip(value, args, strict=True))
        return all(_matches_type(v, args[0]) for v in value)
    return isinstance(value, annotation)


@frozen
class Defaults:
    """A named defaults profile. ``Defaults()`` is the general profile.

    Customize using ``with_overrides(filter={'lowpass': 80.0})``. Unknown
    sections/options and incompatible types fail immediately. Parameter values
    within a section replace existing values; dictionaries are not deep-merged.
    """

    name: str = 'generic'
    version: str = '1'
    _sections: dict[str, Any] = field(
        factory=lambda: {name: cls() for name, cls in schema.SECTION_TYPES.items()},
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('Profile name must be a non-empty string.')
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError('Profile version must be a non-empty string.')

    @classmethod
    def generic(cls) -> Defaults:
        """General defaults: no train removal, assumed audio delay, or drift."""
        return cls()

    @classmethod
    def salzburg(cls) -> Defaults:
        """The historical Salzburg hardware and train-artifact assumptions."""
        return cls().with_overrides(
            name='salzburg',
            ica={'train': True},
            trf={
                'hw_delay_s': 0.0165,
                'fallback_drift_us_per_s': 499.0,
                'audio_channels': ('MISC007', 'MISC008'),
            },
            eye={
                'trigger_ch_name': 'STI101',
                'tpixx_fs': 2000,
                'distance': 82,
                'screen_width': 63,
                'screen_rect': [0, 0, 1920, 1080],
            },
            ica_eog={'left_eog_chs': ['MEG0121', 'MEG0311'], 'right_eog_chs': ['MEG1211', 'MEG1411']},
        )

    def with_overrides(
        self, *, name: str | None = None, version: str | None = None, **sections: Mapping[str, Any]
    ) -> Defaults:
        """Return an independent profile with selected parameters replaced."""
        unknown = sections.keys() - self._sections.keys()
        if unknown:
            raise ValueError(f'Unknown defaults sections: {sorted(unknown)}')
        updated = deepcopy(self._sections)
        for section, changes in sections.items():
            if not isinstance(changes, Mapping):
                raise TypeError(f'Defaults section {section!r} must be a mapping.')
            hints = get_type_hints(type(updated[section]))
            for key, value in changes.items():
                if key not in hints:
                    raise ValueError(f'Unknown defaults option: {section}.{key}')
                if not _matches_type(value, hints[key]):
                    raise TypeError(
                        f'Invalid type for {section}.{key}: expected {hints[key]}, got {type(value).__name__}'
                    )
            updated[section] = evolve(updated[section], **deepcopy(dict(changes)))
        result = type(self)(
            name=self.name if name is None else name, version=self.version if version is None else version
        )
        object.__setattr__(result, '_sections', updated)
        return result

    def section(self, name: str) -> dict[str, Any]:
        """Return a detached dictionary of one section's effective defaults."""
        if name not in self._sections:
            raise ValueError(f'Unknown defaults section: {name}')
        return deepcopy(asdict(self._sections[name], recurse=False))

    def to_dict(self) -> dict[str, Any]:
        """Return an independent snapshot, including profile name/version."""
        return {'name': self.name, 'version': self.version, **{key: self.section(key) for key in self._sections}}

    def _value(self, section: str, key: str) -> Any:
        return deepcopy(getattr(self._sections[section], key))

    @property
    def filter(self) -> schema.FilterDefaults:
        return deepcopy(self._sections['filter'])

    @property
    def resample(self) -> schema.ResampleDefaults:
        return deepcopy(self._sections['resample'])

    @property
    def ica(self) -> schema.ICADefaults:
        return deepcopy(self._sections['ica'])

    @property
    def events(self) -> schema.EventsDefaults:
        return deepcopy(self._sections['events'])

    @property
    def physio(self) -> schema.PhysioCleanerDefaults:
        return deepcopy(self._sections['physio'])

    @property
    def maxwell(self) -> schema.MaxwellDefaults:
        return deepcopy(self._sections['maxwell'])

    @property
    def ransac(self) -> schema.EEGRANSACDefaults:
        return deepcopy(self._sections['ransac'])

    @property
    def rereference(self) -> schema.ReReferenceDefaults:
        return deepcopy(self._sections['rereference'])

    @property
    def epochs(self) -> schema.EpochsDefaults:
        return deepcopy(self._sections['epochs'])

    @property
    def forward_model(self) -> schema.ForwardModelDefaults:
        return deepcopy(self._sections['forward_model'])

    @property
    def spatial_filter(self) -> schema.SpatialFilterDefaults:
        return deepcopy(self._sections['spatial_filter'])

    @property
    def source_reconstruction(self) -> schema.SourceReconstructionDefaults:
        return deepcopy(self._sections['source_reconstruction'])

    @property
    def trf(self) -> schema.EpochTRFDefaults:
        return deepcopy(self._sections['trf'])

    @property
    def pipeline(self) -> schema.PipelineDefaults:
        return deepcopy(self._sections['pipeline'])

    @property
    def eye(self) -> schema.EyeDefaults:
        return deepcopy(self._sections['eye'])

    @property
    def ica_eog(self) -> schema.ICAEOGDefaults:
        return deepcopy(self._sections['ica_eog'])

    @property
    def ica_train(self) -> schema.ICATrainDefaults:
        return deepcopy(self._sections['ica_train'])

    @property
    def trials(self) -> schema.TrialDefaults:
        return deepcopy(self._sections['trials'])

    @property
    def alignment(self) -> schema.AlignmentDefaults:
        return deepcopy(self._sections['alignment'])

    @property
    def audio(self) -> schema.AudioDefaults:
        return deepcopy(self._sections['audio'])


_active_defaults: ContextVar[Defaults] = ContextVar('almkanal_defaults', default=Defaults.generic())


def get_defaults() -> Defaults:
    """Return the active profile in this execution context."""
    return _active_defaults.get()


def configure(defaults: Defaults) -> None:
    """Select defaults for subsequent construction/calls in this context.

    Existing steps retain their settings. New worker processes must explicitly
    select a profile or receive already-constructed, serialized steps.
    """
    if not isinstance(defaults, Defaults):
        raise TypeError('defaults must be a Defaults instance.')
    _active_defaults.set(defaults)


def _select_profile(
    profile: Defaults | str,
    name: str | None,
    version: str | None,
    sections: Mapping[str, Mapping[str, Any]],
) -> Defaults:
    if isinstance(profile, str):
        profiles = {'generic': Defaults.generic, 'salzburg': Defaults.salzburg}
        if profile not in profiles:
            raise ValueError(
                f'Unknown defaults profile {profile!r}. Choose generic, salzburg, or pass a Defaults object.'
            )
        profile = profiles[profile]()
    if not isinstance(profile, Defaults):
        raise TypeError('profile must be a Defaults instance or a built-in profile name.')
    if sections or name is not None or version is not None:
        profile = profile.with_overrides(name=name, version=version, **sections)
    return profile


def set_defaults(
    profile: Defaults | str = 'generic',
    *,
    name: str | None = None,
    version: str | None = None,
    **sections: Mapping[str, Any],
) -> Defaults:
    """Select a profile and user-specific settings in one call.

    For example, ``set_defaults('salzburg', ica={'train': False})``. With no
    arguments, reset to general defaults. Overrides start from the selected
    profile, not prior calls. To update the current settings, explicitly pass
    ``get_defaults()`` as the profile. Return the activated profile for reuse.
    Invalid settings leave the current configuration untouched.
    """
    selected = _select_profile(profile, name, version, sections)
    configure(selected)
    return selected


@contextmanager
def use_defaults(
    profile: Defaults | str = 'generic',
    *,
    name: str | None = None,
    version: str | None = None,
    **sections: Mapping[str, Any],
) -> Iterator[Defaults]:
    """Temporarily apply the same profile/overrides accepted by set_defaults."""
    selected = _select_profile(profile, name, version, sections)
    token = _active_defaults.set(selected)
    try:
        yield selected
    finally:
        _active_defaults.reset(token)


def default_field(section: str, key: str, **kwargs: Any) -> Any:
    """Create an attrs field that snapshots an omitted argument's default."""
    return field(factory=lambda: get_defaults()._value(section, key), metadata={'defaults': (section, key)}, **kwargs)


@frozen
class _ProfileDefault:
    section: str
    key: str

    def __repr__(self) -> str:
        return f'DEFAULT({self.section}.{self.key})'


def function_defaults(
    section: str, *, aliases: Mapping[str, tuple[str, str]] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Resolve omitted function arguments while preserving explicit arguments.

    Signature binding distinguishes omitted values from explicit None/False/0,
    including positional arguments. Signatures expose DEFAULT(section.option)
    to make the dynamic defaults visible in help and introspection.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        sig = signature(function)
        keys = {attribute.name for attribute in fields(schema.SECTION_TYPES[section])}
        mapping = {
            name: (section, name)
            for name, parameter in sig.parameters.items()
            if name in keys and parameter.default is not Parameter.empty
        }
        mapping.update(aliases or {})
        for name in mapping:
            if name not in sig.parameters or sig.parameters[name].default is Parameter.empty:
                raise ValueError(f'Cannot configure required or unknown argument: {function.__name__}.{name}')

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            profile = get_defaults()
            for name, (section_name, key) in mapping.items():
                if name not in bound.arguments:
                    bound.arguments[name] = profile._value(section_name, key)
            return function(*bound.args, **bound.kwargs)

        parameters = [
            parameter.replace(default=_ProfileDefault(*mapping[name])) if name in mapping else parameter
            for name, parameter in sig.parameters.items()
        ]
        wrapped.__signature__ = sig.replace(parameters=parameters)  # type: ignore[attr-defined]
        return wrapped

    return decorate


def step_with_defaults(function: Callable[P, R]) -> Callable[P, R]:
    """Run helpers/callbacks under the step's construction-time profile."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        step = cast('Any', args[0] if args else kwargs['self'])
        profile = step._defaults_profile
        with use_defaults(profile):
            result = function(*args, **kwargs)
        if isinstance(result, dict) and 'data' in result:
            parameters = {
                attribute.name: getattr(step, attribute.name)
                for attribute in fields(type(step))
                if 'defaults' in attribute.metadata
            }
            # Retain serializable settings, without copying large MNE data
            # objects supplied as explicit forward/covariance/data inputs.
            snapshot = build_json([StepInfo(step='parameters', info=parameters)])['processing_history'][0]['info']
            result['defaults'] = {
                'profile': profile.name,
                'version': profile.version,
                'parameters': snapshot,
            }
        return result

    return wrapped
