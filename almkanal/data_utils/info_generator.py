import importlib
import importlib.util
from typing import Any, no_type_check

# --- Optional NumPy import without try/except ---
np: Any | None = None
NUMPY_GENERIC: tuple[type, ...] = ()
NUMPY_NDARRAY: tuple[type, ...] = ()

if importlib.util.find_spec('numpy') is not None:
    np = importlib.import_module('numpy')
    NUMPY_GENERIC = (np.generic,)
    NUMPY_NDARRAY = (np.ndarray,)

_DROP = object()


def build_json(  # noqa C901
    raw_dict: dict,
    *,
    restrict_to_steps: bool = True,
    max_seq_elems: int = 40,  # threshold AND summary length
) -> dict:
    """
    Build JUST the cleaned `steps_info` mapping.

    - Keeps structure (no *_info flattening).
    - Converts NumPy scalars/arrays to JSON-safe types.
    - For sequences/arrays with length/size > max_seq_elems, returns a SUMMARY:
        { "_summary": true, "why": "truncated", "length"/"size": N, "shape"/"dtype" (if ndarray), "sample": [...] }
      where 'sample' contains the first `max_seq_elems` sanitized elements.
    - Drops non-JSON-serializable objects (e.g., MNE ICA instance).
    """

    @no_type_check
    def sanitize(x: Any) -> dict:  # noqa C901
        # JSON-native primitives
        if x is None or isinstance(x, bool | int | float | str):
            return x

        if isinstance(x, NUMPY_GENERIC):
            return x.item()
        if isinstance(x, NUMPY_NDARRAY):
            size = int(x.size)
            if size > max_seq_elems:
                return {
                    'size': size,
                    'shape': tuple(map(int, x.shape)),
                    'dtype': str(x.dtype),
                }
            # small enough: convert to list and recurse
            return [sanitize(i) for i in x.tolist()]

        # Dict → sanitize values; drop non-serializable
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():  # keys assumed to be strings
                sv = sanitize(v)
                if sv is not _DROP:
                    out[k] = sv
            return out

        # Sequences (list/tuple/set) → summarize if large; else sanitize members
        if isinstance(x, list | tuple | set):
            seq = list(x)
            n = len(seq)
            if n > max_seq_elems:
                return {
                    'length': n,
                }
            return [sanitize(i) for i in seq if sanitize(i) is not _DROP]

        # Everything else (e.g., custom objects) → drop
        return _DROP

    # Build subset (optionally ordered/filtered by `steps`)
    si = raw_dict.get('steps_info', {}) or {}
    items = [(s, si[s]) for s in (raw_dict.get('steps') or []) if s in si] if restrict_to_steps else list(si.items())

    cleaned = {}
    for step, payload in items:
        sp = sanitize(payload or {})
        if sp is not _DROP:
            cleaned[step] = sp
    return cleaned
