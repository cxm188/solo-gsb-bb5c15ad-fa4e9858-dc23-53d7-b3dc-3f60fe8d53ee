import random

from fastapi.testclient import TestClient

from app.diff import diff
from app.jsonutil import json_equal
from app.main import app
from app.patch import apply_patch

client = TestClient(app)


def call_diff(source, target):
    return client.post("/diff", json={"source": source, "target": target})


def assert_roundtrip(source, target, operations=None):
    """The operations must be directly consumable by /patch and yield target."""
    if operations is None:
        r = call_diff(source, target)
        assert r.status_code == 200
        operations = r.json()["operations"]
    patched = apply_patch(source, operations)
    assert json_equal(patched, target)
    return operations


# -------------------------------------------------------------- identical


def test_identical_objects_returns_empty_list():
    src = {"a": 1, "b": [1, 2, {"c": None}]}
    r = call_diff(src, {"b": [1, 2, {"c": None}], "a": 1})
    assert r.status_code == 200
    assert r.json() == {"operations": []}


def test_identical_scalars_returns_empty_list():
    assert diff(1, 1) == []
    assert diff("x", "x") == []
    assert diff(None, None) == []
    assert diff([], []) == []
    assert diff({}, {}) == []


def test_int_and_float_one_are_identical():
    # Equality follows the service-wide rule: 1 and 1.0 are the same number.
    assert diff(1, 1.0) == []


# ---------------------------------------------------------------- objects


def test_object_added_members_are_add_ops():
    operations = assert_roundtrip({"a": 1}, {"a": 1, "b": 2, "c": 3})
    assert operations == [
        {"op": "add", "path": "/b", "value": 2},
        {"op": "add", "path": "/c", "value": 3},
    ]


def test_object_removed_members_are_remove_ops():
    operations = assert_roundtrip({"a": 1, "b": 2, "c": 3}, {"b": 2})
    assert operations == [
        {"op": "remove", "path": "/a"},
        {"op": "remove", "path": "/c"},
    ]


def test_object_changed_member_is_replace():
    operations = assert_roundtrip({"a": 1}, {"a": 9})
    assert operations == [{"op": "replace", "path": "/a", "value": 9}]


def test_object_member_type_change_is_replace():
    operations = assert_roundtrip({"a": 1}, {"a": [1]})
    assert operations == [{"op": "replace", "path": "/a", "value": [1]}]


def test_object_null_vs_missing_not_confused():
    add_null = assert_roundtrip({}, {"a": None})
    assert add_null == [{"op": "add", "path": "/a", "value": None}]
    remove_null = assert_roundtrip({"a": None}, {})
    assert remove_null == [{"op": "remove", "path": "/a"}]
    assert_roundtrip({"a": None}, {"a": 0})


def test_nested_objects_are_recursed_not_replaced():
    source = {"a": {"x": 1, "y": 2}, "b": 3}
    target = {"a": {"x": 1, "y": 9, "z": 4}, "b": 3}
    operations = assert_roundtrip(source, target)
    assert operations == [
        {"op": "replace", "path": "/a/y", "value": 9},
        {"op": "add", "path": "/a/z", "value": 4},
    ]


def test_object_member_names_are_pointer_escaped():
    source = {"a/b": 1, "~x": 2, "/~": 3}
    target = {"a/b": 10, "~x": 2, "~1~0y": 4}
    operations = assert_roundtrip(source, target)
    # Keys are walked in sorted (code-point) order: "/" < "a" < "~".
    assert operations == [
        {"op": "remove", "path": "/~1~0"},
        {"op": "replace", "path": "/a~1b", "value": 10},
        {"op": "add", "path": "/~01~00y", "value": 4},
    ]
    # Untouched escaped member must survive.
    assert apply_patch(source, operations)["~x"] == 2


def test_object_keys_processed_in_sorted_order_is_deterministic():
    source = {k: 0 for k in ("z", "m", "a")}
    target = {}
    operations = diff(source, target)
    assert [op["path"] for op in operations] == ["/a", "/m", "/z"]


# ----------------------------------------------------------------- arrays


def test_array_append_only():
    operations = assert_roundtrip([1, 2], [1, 2, 3])
    assert operations == [{"op": "add", "path": "/2", "value": 3}]


def test_array_prepend_shifts_without_whole_replace():
    operations = assert_roundtrip([1, 2, 3], [0, 1, 2, 3])
    assert operations == [{"op": "add", "path": "/0", "value": 0}]


def test_array_remove_only():
    operations = assert_roundtrip([1, 2, 3], [1, 3])
    assert operations == [{"op": "remove", "path": "/1"}]


def test_array_mixed_insert_remove_with_common_prefix_and_suffix():
    operations = assert_roundtrip([1, 2, 3, 4], [1, 9, 3, 4, 5])
    assert operations == [
        {"op": "remove", "path": "/1"},
        {"op": "add", "path": "/1", "value": 9},
        {"op": "add", "path": "/4", "value": 5},
    ]


def test_array_disjoint_contents_emits_remove_then_add():
    operations = assert_roundtrip([1, 2], [3, 4])
    assert operations == [
        {"op": "remove", "path": "/0"},
        {"op": "remove", "path": "/0"},
        {"op": "add", "path": "/0", "value": 3},
        {"op": "add", "path": "/1", "value": 4},
    ]


def test_array_common_elements_recurse_in_place():
    source = [{"a": 1}, {"a": 2}]
    target = [{"a": 1}, {"a": 20}]
    operations = assert_roundtrip(source, target)
    assert operations == [{"op": "replace", "path": "/1/a", "value": 20}]


def test_array_scalar_value_change_is_remove_and_add():
    # Unequal scalars cannot be aligned by the LCS (they carry no internal
    # structure to recurse into), so a changed value is remove + add.
    operations = assert_roundtrip([1], [2])
    assert operations == [
        {"op": "remove", "path": "/0"},
        {"op": "add", "path": "/0", "value": 2},
    ]
    # A scalar of a different JSON type likewise stays unaligned.
    assert_roundtrip([1, 2], ["a", 2])


def test_array_lcs_aligns_later_matching_element():
    source = [1, 2, 3]
    target = [2, 3, 4]
    operations = assert_roundtrip(source, target)
    assert operations == [
        {"op": "remove", "path": "/0"},
        {"op": "add", "path": "/2", "value": 4},
    ]


def test_array_deterministic_tie_break():
    # Multiple same-type alignments are possible; the output must be stable.
    assert diff([1, 2], [3, 4]) == [
        {"op": "remove", "path": "/0"},
        {"op": "remove", "path": "/0"},
        {"op": "add", "path": "/0", "value": 3},
        {"op": "add", "path": "/1", "value": 4},
    ]
    assert diff([1, 2], [3, 4]) == diff([1, 2], [3, 4])


def test_empty_to_nonempty_and_back():
    forward = assert_roundtrip([], [1, 2])
    assert forward == [
        {"op": "add", "path": "/0", "value": 1},
        {"op": "add", "path": "/1", "value": 2},
    ]
    assert_roundtrip([1, 2], [])


def test_array_with_objects_inserted_between_recursed_anchors():
    source = [{"k": 1}, {"k": 2}, {"k": 3}]
    target = [{"k": 1}, {"k": 0}, {"k": 2}, {"k": 30}, {"k": 3}]
    operations = assert_roundtrip(source, target)
    # Equal objects stay anchored; the new objects are pure inserts, so no
    # element is replaced wholesale.
    assert operations == [
        {"op": "add", "path": "/1", "value": {"k": 0}},
        {"op": "add", "path": "/3", "value": {"k": 30}},
    ]


def test_array_modified_objects_are_recursed_not_replaced():
    source = [{"k": 1}, {"k": 2}]
    target = [{"k": 10}, {"k": 2}]
    operations = assert_roundtrip(source, target)
    assert operations == [{"op": "replace", "path": "/0/k", "value": 10}]


def test_nested_array_inside_object():
    source = {"items": ["a", "b", "c"]}
    target = {"items": ["a", "x", "c", "d"]}
    operations = assert_roundtrip(source, target)
    # "b" -> "x" cannot align (unequal scalars); "d" is a trailing append.
    assert operations == [
        {"op": "remove", "path": "/items/1"},
        {"op": "add", "path": "/items/1", "value": "x"},
        {"op": "add", "path": "/items/3", "value": "d"},
    ]


# ------------------------------------------------------------------ root


def test_root_scalar_change_is_root_replace():
    operations = assert_roundtrip(1, 2)
    assert operations == [{"op": "replace", "path": "", "value": 2}]


def test_root_type_change_is_root_replace():
    assert_roundtrip({"a": 1}, [1, 2])
    assert_roundtrip([1, 2], {"a": 1})
    assert_roundtrip("x", None)
    assert_roundtrip(False, 0)


def test_root_equal_arrays_unchanged():
    assert diff([1, {"a": [2]}], [1, {"a": [2]}]) == []


# -------------------------------------------------------------- end to end


def test_endpoint_response_shape_and_patch_interop():
    source = {"a": [1, 2, 3], "b": {"x": 1}, "gone": 1}
    target = {"a": [1, 2, 3, 4], "b": {"x": 2, "y": 5}, "new": None}
    diff_resp = call_diff(source, target)
    assert diff_resp.status_code == 200
    operations = diff_resp.json()["operations"]
    patch_resp = client.post(
        "/patch", json={"document": source, "operations": operations}
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json() == {"document": target}


def test_operations_only_use_supported_shapes():
    rng = random.Random(1234)

    def random_value(depth=0):
        if depth >= 3:
            return rng.choice([None, True, False, 1, 2, "a", "b"])
        kind = rng.choice(["scalar", "object", "array"])
        if kind == "scalar":
            return rng.choice([None, True, 1, 1.0, "s", "t"])
        if kind == "array":
            return [random_value(depth + 1) for _ in range(rng.randrange(5))]
        return {
            k: random_value(depth + 1)
            for k in rng.sample(["a", "b", "c", "a/b", "~z"], rng.randrange(5))
        }

    for _ in range(300):
        source = random_value()
        target = random_value()
        operations = diff(source, target)
        for op in operations:
            assert set(op) == {"op", "path", "value"} or set(op) == {"op", "path"}
            assert op["op"] in ("add", "remove", "replace")
            assert isinstance(op["path"], str)
        assert json_equal(apply_patch(source, operations), target)
        # Diffing twice must be deterministic.
        assert diff(source, target) == operations


def test_diff_does_not_mutate_inputs():
    source = {"a": [1, {"k": 2}]}
    target = {"a": [1, {"k": 3}]}
    snapshot_source = {"a": [1, {"k": 2}]}
    snapshot_target = {"a": [1, {"k": 3}]}
    diff(source, target)
    assert source == snapshot_source
    assert target == snapshot_target


def test_missing_fields_rejected():
    assert client.post("/diff", json={"source": {}}).status_code == 422
    assert client.post("/diff", json={"target": {}}).status_code == 422
    assert client.post("/diff", json={}).status_code == 422


def test_other_endpoints_still_healthy():
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post(
        "/merge",
        json={"base": {"a": 1}, "left": {"a": 2}, "right": {"a": 1}},
    )
    assert r.json() == {"merged": {"a": 2}}
