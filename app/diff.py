"""RFC 6902 JSON Patch generation (diff).

:func:`diff` computes a list of RFC 6902 ``add`` / ``remove`` /
``replace`` operations that turns ``source`` into ``target``; feeding the
result straight back to :func:`app.patch.apply_patch` yields exactly
``target``.

Strategy:

* objects are compared member by member, recursively, with keys processed
  in sorted order so the output is deterministic; member names are escaped
  as JSON Pointer tokens (RFC 6901) before being appended to the path;
* scalars, or any type change, yield a single ``replace`` at object level
  (and ``remove``/``add`` at array-element level);
* arrays are aligned with a longest common subsequence. The alignment
  weight between two elements reflects their contents: equal elements
  dominate (weight 2) and same-kind containers whose recursive patch is
  itself *applicable* align on a common line (weight 1), so near-equal
  objects/arrays are patched in place instead of replaced wholesale.
  Unequal scalars never align (there is nothing inside a scalar to recurse
  into); elements present on only one side, or two unequal scalars, are
  expressed as ``remove``/``add``. The bottom-up feasibility rule means a
  container whose aligned contents change JSON kind -- e.g. an aligned
  array pairing a list element with an object element -- is never aligned:
  that pair is rebuilt with ``remove``/``add`` rather than generating
  operations aimed down an incompatible sub-path. Every array pair's
  weighted LCS is built once and memoized (JSON trees are acyclic, so the
  analyses form a DAG), ties are broken deterministically to keep equal
  elements left-aligned, and operations are emitted gap by gap -- removes
  right-to-left, adds left-to-right before the (right-shifted) anchor, then
  the matched anchor recursed into -- an order that stays valid as every
  operation is applied.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .jsonutil import json_equal

# Cell weight for an equal element pair (dominates any container alignment).
_W_EQUAL = 2
# Cell weight for an unequal but in-place-patchable same-kind container pair.
_W_ALIGNABLE = 1


def escape_pointer(token: str) -> str:
    """Escape one reference token as per RFC 6901."""
    return token.replace("~", "~0").replace("/", "~1")


def _same_kind_containers(a: Any, b: Any) -> bool:
    return (isinstance(a, dict) and isinstance(b, dict)) or (
        isinstance(a, list) and isinstance(b, list)
    )


def diff(source: Any, target: Any) -> list[dict]:
    """Return RFC 6902 operations transforming ``source`` into ``target``.

    The operations are ordered so that applying them one by one with
    :func:`app.patch.apply_patch` results in a value structurally equal
    to ``target``. Equal inputs yield an empty list.
    """
    operations: list[dict] = []
    # Memoized per-pair array analyses: identity pair -> (feasible, pairs,
    # weights). JSON values form a tree, so an element pair is always
    # strictly deeper than the arrays that contain it: the dependency graph
    # of pair analyses is a DAG and plain bottom-up memoization is safe.
    analyses: dict[tuple[int, int], tuple[bool, list, list]] = {}

    def analyze_arrays(a: list, b: list) -> tuple[bool, list, list]:
        key = (id(a), id(b))
        cached = analyses.get(key)
        if cached is not None:
            return cached
        m, n = len(a), len(b)
        weights = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                x, y = a[i], b[j]
                if json_equal(x, y):
                    weights[i][j] = _W_EQUAL
                elif _same_kind_containers(x, y) and alignable_containers(x, y):
                    # Same-kind containers that can be patched in place
                    # align on a common line even though they are unequal;
                    # unequal scalars never match (they are removed/added).
                    weights[i][j] = _W_ALIGNABLE

        table = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                w = weights[i - 1][j - 1]
                diagonal = table[i - 1][j - 1] + w if w else 0
                table[i][j] = max(table[i - 1][j], table[i][j - 1], diagonal)

        pairs: list[tuple[int, int]] = []
        i, j = m, n
        while i > 0 and j > 0:
            w = weights[i - 1][j - 1]
            if (
                w
                and table[i][j] == table[i - 1][j - 1] + w
                # Diagonal only when it strictly beats skipping, keeping
                # equal elements left-aligned; ties resolve up before left.
                and table[i][j] > table[i - 1][j]
                and table[i][j] > table[i][j - 1]
            ):
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif table[i - 1][j] >= table[i][j - 1]:
                i -= 1
            else:
                j -= 1
        pairs.reverse()

        feasible_arrays = bool(pairs)
        result = (feasible_arrays, pairs, weights)
        analyses[key] = result
        return result

    def alignable_containers(a: Any, b: Any) -> bool:
        """Feasibility of aligning two same-kind, unequal containers."""
        key = (id(a), id(b))
        cached = analyses.get(key)
        if cached is not None:
            return cached[0]
        if isinstance(a, dict):
            # Scalar/kind-change members are always a single replace; only a
            # shared same-kind container member with an impossible inner
            # alignment blocks aligning the object.
            for k in a.keys() & b.keys():
                av, bv = a[k], b[k]
                if (
                    _same_kind_containers(av, bv)
                    and not json_equal(av, bv)
                    and not alignable_containers(av, bv)
                ):
                    return False
            return True
        return analyze_arrays(a, b)[0]

    def emit_member(src: Any, tgt: Any, path: str) -> None:
        """Emit ops for a pair reached at the root or as an object member."""
        if json_equal(src, tgt):
            return
        if isinstance(src, dict) and isinstance(tgt, dict):
            for key in sorted(src.keys() | tgt.keys()):
                token = f"{path}/{escape_pointer(key)}"
                if key not in tgt:
                    operations.append({"op": "remove", "path": token})
                elif key not in src:
                    operations.append(
                        {"op": "add", "path": token, "value": deepcopy(tgt[key])}
                    )
                elif json_equal(src[key], tgt[key]):
                    continue
                elif _same_kind_containers(src[key], tgt[key]):
                    if alignable_containers(src[key], tgt[key]):
                        emit_member(src[key], tgt[key], token)
                    else:
                        # Same-kind containers whose inner alignment is
                        # impossible (an element/member kind conflict).
                        operations.append(
                            {"op": "replace", "path": token, "value": deepcopy(tgt[key])}
                        )
                else:
                    # Scalar change, or scalar<->container/object<->array
                    # kind change: one whole-member replace.
                    operations.append(
                        {"op": "replace", "path": token, "value": deepcopy(tgt[key])}
                    )
        elif isinstance(src, list) and isinstance(tgt, list):
            emit_array(src, tgt, path)
        else:
            operations.append(
                {"op": "replace", "path": path, "value": deepcopy(tgt)}
            )

    def emit_anchor(src: Any, tgt: Any, path: str) -> None:
        """Emit ops for an element occupying an aligned array slot."""
        if json_equal(src, tgt):
            return
        if _same_kind_containers(src, tgt) and alignable_containers(src, tgt):
            if isinstance(src, dict):
                emit_member(src, tgt, path)
            else:
                emit_array(src, tgt, path)
        else:
            # An aligned slot that is not a feasible container pair cannot
            # occur via the LCS (only equals and feasible containers match);
            # this branch is retained defensively as remove/add.
            operations.append({"op": "remove", "path": path})
            operations.append(
                {"op": "add", "path": path, "value": deepcopy(tgt)}
            )

    def emit_array(src: list, tgt: list, path: str) -> None:
        _feasible, pairs, _weights = analyze_arrays(src, tgt)
        cursor = 0
        src_pos = 0
        tgt_pos = 0
        for src_index, tgt_index in pairs:
            removed = src_index - src_pos
            inserted = tgt_index - tgt_pos
            # Source-only elements: remove right-to-left at the gap start so
            # earlier indices stay valid.
            for _ in range(removed):
                operations.append({"op": "remove", "path": f"{path}/{cursor}"})
            # Target-only elements: insert left-to-right at the gap. They
            # land at cursor, cursor+1, ... and push the anchor right.
            for k in range(inserted):
                operations.append(
                    {
                        "op": "add",
                        "path": f"{path}/{cursor + k}",
                        "value": deepcopy(tgt[tgt_pos + k]),
                    }
                )
            # The matched anchor now sits at ``cursor + inserted``; recurse
            # into it. This cannot change the surrounding array's length or
            # later indices.
            emit_anchor(
                src[src_index],
                tgt[tgt_index],
                f"{path}/{cursor + inserted}",
            )
            cursor += inserted + 1
            src_pos = src_index + 1
            tgt_pos = tgt_index + 1

        # Trailing gap after the last anchor (or the whole array).
        removed = len(src) - src_pos
        inserted = len(tgt) - tgt_pos
        for _ in range(removed):
            operations.append({"op": "remove", "path": f"{path}/{cursor}"})
        for k in range(inserted):
            operations.append(
                {
                    "op": "add",
                    "path": f"{path}/{cursor + k}",
                    "value": deepcopy(tgt[tgt_pos + k]),
                }
            )

    emit_member(source, target, "")
    return operations
