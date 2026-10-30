"""Canonical step names for legacy pipeline references."""


def canonical_step_name(name: str) -> str:
    """Resolve a historical step name without changing stored history."""
    return 'EpochStimuls' if name == 'EpochTRF' else name
