"""RFC 6902 JSON Patch generation (structured diff).

The :func:`make_patch` entry point compares two arbitrary JSON values and
returns a deterministic list of RFC 6902 ``add`` / ``remove`` / ``replace``
operations that, fed to :func:`app.patch.apply_patch` in order, transforms a
deep copy of ``source`` exactly into ``target``.

* Objects are compared member by member, recursively, and every pointer token
  is escaped per RFC 6901 (``~`` -> ``~0``, ``/`` -> ``~1``).
* Arrays are aligned with a longest-common-subsequence computed from JSON
  equality; matched elements keep their position and are recursed into, while
  surplus source elements become ``remove`` operations and surplus target
  elements ``add`` operations, so a small edit never replaces the whole array.
  Between two matched anchors, surplus source and target elements at the same
  offset are paired and recursed into (or replaced at the leaf), instead of
  being emitted as a remove plus an add.
* A type change or scalar change at any position is a single ``replace``.
* Structurally equal values yield no operations; identical input yields ``[]``.
"""

from __future__ import annotations

from typing import Any

from .jsonutil import escape_pointer, json_equal


def make_patch(source: Any, target: Any) -> list[dict]:
    """Return RFC 6902 operations transforming ``source`` into ``target``."""
    operations: list[dict] = []
    _diff(source, target, "", operations)
    return operations


def _diff(source: Any, target: Any, path: str, operations: list[dict]) -> None:
    """Append operations for ``source`` -> ``target`` at JSON Pointer path."""
    if json_equal(source, target):
        return
    if isinstance(source, dict) and isinstance(target, dict):
        _diff_object(source, target, path, operations)
    elif isinstance(source, list) and isinstance(target, list):
        _diff_array(source, target, path, operations)
    else:
        # Type change (object <-> array <-> scalar) or scalar change.
        operations.append({"op": "replace", "path": path, "value": target})


def _diff_object(
    source: dict, target: dict, path: str, operations: list[dict]
) -> None:
    # Sorted member names give deterministic output; object operations address
    # independent paths, so the order cannot make a later path unresolvable.
    for key in sorted(source.keys() | target.keys()):
        member_path = f"{path}/{escape_pointer(key)}"
        if key not in source:
            operations.append({"op": "add", "path": member_path, "value": target[key]})
        elif key not in target:
            operations.append({"op": "remove", "path": member_path})
        else:
            _diff(source[key], target[key], member_path, operations)


def _lcs_pairs(source: list, target: list) -> list[tuple[int, int]]:
    """Return matched ``(source_index, target_index)`` pairs of an LCS.

    Equality is structural JSON equality. Backtracking prefers, in order,
    the diagonal match, then skipping a source element, then skipping a
    target element, which makes the alignment fully deterministic.
    """
    m, n = len(source), len(target)
    lengths = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if json_equal(source[i], target[j]):
                lengths[i][j] = lengths[i + 1][j + 1] + 1
            else:
                lengths[i][j] = max(lengths[i + 1][j], lengths[i][j + 1])

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < m and j < n:
        if json_equal(source[i], target[j]):
            pairs.append((i, j))
            i += 1
            j += 1
        elif lengths[i + 1][j] >= lengths[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _diff_array(
    source: list, target: list, path: str, operations: list[dict]
) -> None:
    pairs = _lcs_pairs(source, target)

    # Surplus index ranges for each segment: one segment per matched anchor
    # (the elements just before that anchor) plus a trailing segment.
    segments: list[tuple[range, range]] = []
    prev_i = prev_j = -1
    for i, j in pairs:
        segments.append((range(prev_i + 1, i), range(prev_j + 1, j)))
        prev_i, prev_j = i, j
    segments.append(
        (range(prev_i + 1, len(source)), range(prev_j + 1, len(target)))
    )

    # Emit left to right; ``cursor`` is the current array index of the first
    # element of the segment being processed, after earlier segments applied.
    cursor = 0
    for segment_index, (s_indices, t_indices) in enumerate(segments):
        paired = min(len(s_indices), len(t_indices))
        s_extra = len(s_indices) - paired
        t_extra = len(t_indices) - paired

        # 1. Elements paired at the same offset are changed in place: recurse
        #    while they still occupy their unshifted positions.
        for k in range(paired):
            _diff(
                source[s_indices[k]],
                target[t_indices[k]],
                f"{path}/{cursor + k}",
                operations,
            )

        # 2. Surplus source elements sit directly after the paired ones;
        #    remove them highest index first so lower indices never shift.
        for offset in range(s_extra - 1, -1, -1):
            operations.append(
                {"op": "remove", "path": f"{path}/{cursor + paired + offset}"}
            )

        # 3. Surplus target elements are inserted after the paired ones.
        #    Before an anchor, insert in reverse at a single fixed index
        #    (the anchor's current position); after the last anchor, append.
        if t_extra:
            if segment_index < len(pairs):
                insert_path = f"{path}/{cursor + paired}"
                for k in range(t_extra - 1, -1, -1):
                    operations.append(
                        {
                            "op": "add",
                            "path": insert_path,
                            "value": target[t_indices[paired + k]],
                        }
                    )
            else:
                for k in range(t_extra):
                    operations.append(
                        {
                            "op": "add",
                            "path": f"{path}/-",
                            "value": target[t_indices[paired + k]],
                        }
                    )

        # Advance past this segment and the matched anchor (if any); matched
        # elements are equal by construction and need no operation.
        cursor += paired + t_extra
        if segment_index < len(pairs):
            cursor += 1
