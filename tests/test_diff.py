import random

from fastapi.testclient import TestClient

from app.diff import make_patch
from app.main import app
from app.patch import apply_patch

client = TestClient(app)


def diff(source, target):
    return client.post("/diff", json={"source": source, "target": target})


def check_roundtrip(source, target):
    """Generated ops must apply (via the real /patch engine) onto target."""
    operations = make_patch(source, target)
    result = apply_patch(source, operations)
    assert result == target, (source, target, operations, result)
    return operations


# ------------------------------------------------------------- identical


def test_identical_objects_return_empty():
    r = diff({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]})
    assert r.status_code == 200
    assert r.json() == {"operations": []}


def test_identical_scalars_return_empty():
    for value in [None, True, False, 0, 1, 1.5, "x", [], {}]:
        assert make_patch(value, value) == []


def test_int_and_float_one_are_equal_so_no_operations():
    assert make_patch({"a": 1}, {"a": 1.0}) == []


def test_empty_operations_apply_to_anything():
    check_roundtrip([1, {"a": 2}], [1, {"a": 2}])


# ------------------------------------------------------- scalars / types


def test_scalar_change_is_single_replace():
    assert make_patch(1, 2) == [{"op": "replace", "path": "", "value": 2}]


def test_string_scalar_change():
    assert make_patch("a", "b") == [{"op": "replace", "path": "", "value": "b"}]


def test_null_to_scalar_is_replace():
    assert make_patch(None, 1) == [{"op": "replace", "path": "", "value": 1}]


def test_type_changes_use_replace():
    cases = [
        ({"a": 1}, [1]),
        ([1], {"a": 1}),
        (1, "1"),
        (1, True),
        (True, False),
        ([1], 1),
        ({"a": 1}, None),
    ]
    for source, target in cases:
        operations = make_patch(source, target)
        assert operations == [{"op": "replace", "path": "", "value": target}]
        check_roundtrip(source, target)


def test_nested_type_change_uses_replace_at_member_path():
    operations = make_patch({"a": {"b": 1}}, {"a": [1]})
    assert operations == [{"op": "replace", "path": "/a", "value": [1]}]


def test_bool_is_not_silently_replaced_with_int():
    # 1 vs 1.0 are equal, but true vs 1 are different JSON values.
    operations = make_patch({"a": True}, {"a": 1})
    assert operations == [{"op": "replace", "path": "/a", "value": 1}]


# -------------------------------------------------------------- objects


def test_object_add_remove_replace_members():
    operations = make_patch(
        {"a": 1, "b": 2, "c": 3}, {"a": 10, "c": 3, "d": 4}
    )
    assert operations == [
        {"op": "replace", "path": "/a", "value": 10},
        {"op": "remove", "path": "/b"},
        {"op": "add", "path": "/d", "value": 4},
    ]
    check_roundtrip({"a": 1, "b": 2, "c": 3}, {"a": 10, "c": 3, "d": 4})


def test_object_members_sorted_for_deterministic_output():
    operations = make_patch({"z": 1, "a": 1}, {"z": 2, "a": 2})
    assert [op["path"] for op in operations] == ["/a", "/z"]


def test_nested_object_recursion():
    source = {"a": {"x": 1, "y": 2}, "b": [1]}
    target = {"a": {"x": 1, "y": 20, "z": 3}, "b": [1]}
    operations = make_patch(source, target)
    assert operations == [
        {"op": "replace", "path": "/a/y", "value": 20},
        {"op": "add", "path": "/a/z", "value": 3},
    ]
    check_roundtrip(source, target)


def test_object_to_empty_removes_every_member_not_root_replace():
    operations = make_patch({"a": 1, "b": 2}, {})
    assert operations == [
        {"op": "remove", "path": "/a"},
        {"op": "remove", "path": "/b"},
    ]


def test_empty_object_to_object_uses_adds():
    assert make_patch({}, {"a": 1}) == [
        {"op": "add", "path": "/a", "value": 1}
    ]


def test_add_null_member_value():
    operations = make_patch({}, {"a": None})
    assert operations == [{"op": "add", "path": "/a", "value": None}]
    check_roundtrip({}, {"a": None})


def test_pointer_escapes_slash_and_tilde_in_member_names():
    source = {"a/b": 1, "~x": 2, "/~": 3}
    target = {"a/b": 10, "~x": 20, "new/key~": 30}
    operations = make_patch(source, target)
    assert operations == [
        {"op": "remove", "path": "/~1~0"},
        {"op": "replace", "path": "/a~1b", "value": 10},
        {"op": "add", "path": "/new~1key~0", "value": 30},
        {"op": "replace", "path": "/~0x", "value": 20},
    ]
    check_roundtrip(source, target)


def test_empty_string_member_name():
    operations = make_patch({"": 1}, {"": 2})
    assert operations == [{"op": "replace", "path": "/", "value": 2}]
    check_roundtrip({"": 1}, {"": 2})


# --------------------------------------------------------------- arrays


def test_array_append_uses_add():
    assert make_patch([1, 2], [1, 2, 3]) == [
        {"op": "add", "path": "/-", "value": 3}
    ]


def test_array_prepend_uses_add_at_zero():
    assert make_patch([2, 3], [1, 2, 3]) == [
        {"op": "add", "path": "/0", "value": 1}
    ]


def test_array_remove_element():
    assert make_patch([1, 2, 3], [1, 3]) == [
        {"op": "remove", "path": "/1"}
    ]


def test_array_replace_element_in_place():
    assert make_patch([1, 2, 3], [1, 9, 3]) == [
        {"op": "replace", "path": "/1", "value": 9}
    ]


def test_array_insert_in_middle():
    operations = make_patch([1, 3], [1, 2, 3])
    assert operations == [{"op": "add", "path": "/1", "value": 2}]


def test_array_does_not_replace_whole_array_on_small_edit():
    # The common elements must be matched by LCS and never appear as values.
    big = [{"k": i} for i in range(10)]
    edited = [dict(item) for item in big]
    edited.append({"k": 10})
    operations = make_patch(big, edited)
    assert operations == [{"op": "add", "path": "/-", "value": {"k": 10}}]


def test_array_common_objects_recurse_in_place():
    source = [{"a": 1}, {"a": 2}]
    target = [{"a": 1}, {"a": 20}]
    operations = make_patch(source, target)
    assert operations == [
        {"op": "replace", "path": "/1/a", "value": 20}
    ]
    check_roundtrip(source, target)


def test_array_remove_and_insert_between_anchors():
    source = [1, 2, 3, 4]
    target = [1, 7, 3, 4]
    operations = make_patch(source, target)
    # 2 and 7 are paired at the same offset between anchors 1 and 3.
    assert operations == [{"op": "replace", "path": "/1", "value": 7}]
    check_roundtrip(source, target)


def test_array_gap_with_extra_source_and_target():
    source = [1, 2, 8, 3]
    target = [1, 7, 9, 3]
    operations = check_roundtrip(source, target)
    assert operations == [
        {"op": "replace", "path": "/1", "value": 7},
        {"op": "replace", "path": "/2", "value": 9},
    ]


def test_array_gap_more_source_than_target_removes_surplus():
    source = [1, 2, 8, 9, 3]
    target = [1, 7, 3]
    operations = check_roundtrip(source, target)
    assert operations == [
        {"op": "replace", "path": "/1", "value": 7},
        {"op": "remove", "path": "/3"},
        {"op": "remove", "path": "/2"},
    ]


def test_array_gap_more_target_than_source_inserts_surplus():
    source = [1, 2, 3]
    target = [1, 7, 8, 9, 3]
    operations = check_roundtrip(source, target)
    assert operations == [
        {"op": "replace", "path": "/1", "value": 7},
        {"op": "add", "path": "/2", "value": 9},
        {"op": "add", "path": "/2", "value": 8},
    ]


def test_array_multiple_removals_high_index_first():
    operations = make_patch([1, 2, 3, 4], [1, 4])
    assert operations == [
        {"op": "remove", "path": "/2"},
        {"op": "remove", "path": "/1"},
    ]
    check_roundtrip([1, 2, 3, 4], [1, 4])


def test_array_multiple_insertions_before_anchor():
    operations = make_patch([3], [1, 2, 3])
    assert operations == [
        {"op": "add", "path": "/0", "value": 2},
        {"op": "add", "path": "/0", "value": 1},
    ]
    check_roundtrip([3], [1, 2, 3])


def test_array_replace_all_when_no_common_subsequence():
    operations = make_patch([1, 2], [3, 4])
    assert operations == [
        {"op": "replace", "path": "/0", "value": 3},
        {"op": "replace", "path": "/1", "value": 4},
    ]
    check_roundtrip([1, 2], [3, 4])


def test_array_shrink_to_empty():
    operations = make_patch([1, 2], [])
    assert operations == [
        {"op": "remove", "path": "/1"},
        {"op": "remove", "path": "/0"},
    ]
    check_roundtrip([1, 2], [])


def test_array_grow_from_empty_uses_dash_append():
    operations = make_patch([], [1, 2])
    assert operations == [
        {"op": "add", "path": "/-", "value": 1},
        {"op": "add", "path": "/-", "value": 2},
    ]
    check_roundtrip([], [1, 2])


def test_array_with_duplicate_values_is_deterministic():
    operations_a = make_patch([1, 1, 2], [1, 2, 2])
    operations_b = make_patch([1, 1, 2], [1, 2, 2])
    assert operations_a == operations_b
    check_roundtrip([1, 1, 2], [1, 2, 2])


def test_nested_arrays():
    source = {"rows": [[1, 2], [3, 4]]}
    target = {"rows": [[1, 2, 0], [3, 4]]}
    operations = make_patch(source, target)
    assert operations == [
        {"op": "add", "path": "/rows/0/-", "value": 0}
    ]
    check_roundtrip(source, target)


def test_array_element_type_changes_via_paired_replace():
    operations = make_patch([1, 2], [[1], 2])
    assert operations == [{"op": "replace", "path": "/0", "value": [1]}]
    check_roundtrip([1, 2], [[1], 2])


# ------------------------------------------------------- root documents


def test_root_object_to_array():
    operations = make_patch({"a": 1}, [1, 2])
    assert operations == [{"op": "replace", "path": "", "value": [1, 2]}]
    r = client.post(
        "/patch",
        json={"document": {"a": 1}, "operations": operations},
    )
    assert r.status_code == 200
    assert r.json() == {"document": [1, 2]}


def test_root_scalar_to_object():
    check_roundtrip(42, {"a": 1})


# --------------------------------------------------------------- through API


def test_diff_endpoint_roundtrip_example():
    source = {"a": [1, 2, 3], "b": {"x": 1}, "gone": True}
    target = {"a": [1, 2, 3, 4], "b": {"x": 2, "y": 3}, "new": "v"}
    r = diff(source, target)
    assert r.status_code == 200
    operations = r.json()["operations"]
    applied = client.post(
        "/patch", json={"document": source, "operations": operations}
    )
    assert applied.status_code == 200
    assert applied.json()["document"] == target


def test_diff_response_operations_are_directly_patch_compatible():
    source = {"list": [{"id": 1, "v": 0}, {"id": 2}]}
    target = {"list": [{"id": 1, "v": 1}, {"id": 2}, {"id": 3}]}
    operations = diff(source, target).json()["operations"]
    applied = client.post(
        "/patch", json={"document": source, "operations": operations}
    )
    assert applied.status_code == 200
    assert applied.json()["document"] == target


def test_diff_missing_fields_are_422():
    assert client.post("/diff", json={"source": 1}).status_code == 422
    assert client.post("/diff", json={"target": 1}).status_code == 422
    assert client.post("/diff", json={}).status_code == 422


def test_diff_accepts_null_values():
    r = diff(None, {"a": None})
    assert r.status_code == 200
    operations = r.json()["operations"]
    applied = client.post(
        "/patch", json={"document": None, "operations": operations}
    )
    assert applied.json()["document"] == {"a": None}


def test_diff_does_not_mutate_inputs():
    source = {"a": [1, 2]}
    target = {"a": [1, 2, 3]}
    make_patch(source, target)
    assert source == {"a": [1, 2]}
    assert target == {"a": [1, 2, 3]}


def test_other_endpoints_still_work():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    r = client.post(
        "/merge",
        json={"base": {"a": 1}, "left": {"a": 2}, "right": {"a": 1}},
    )
    assert r.status_code == 200
    assert r.json() == {"merged": {"a": 2}}


# ---------------------------------------------------------- property tests


def _random_value(rng, depth):
    if depth <= 0:
        return rng.choice([None, True, False, 0, 1, 2, 1.5, "a", "b", "a/b", "~x"])
    kind = rng.randrange(3)
    if kind == 0:
        return rng.choice(
            [None, True, False, 0, 1, 2, 1.5, "a", "b", "a/b", "~x"]
        )
    if kind == 1:
        return [
            _random_value(rng, depth - 1)
            for _ in range(rng.randrange(4))
        ]
    keys = ["a", "b", "c", "a/b", "~x", ""]
    return {
        key: _random_value(rng, depth - 1)
        for key in rng.sample(keys, rng.randrange(4))
    }


def test_randomized_roundtrips():
    rng = random.Random(20260918)
    for _ in range(500):
        source = _random_value(rng, 3)
        target = _random_value(rng, 3)
        operations = make_patch(source, target)
        assert apply_patch(source, operations) == target
        # Deterministic: same inputs always give identical operations.
        assert make_patch(source, target) == operations
        # All operations are among the ops the /patch engine supports.
        assert all(
            op["op"] in ("add", "remove", "replace") for op in operations
        )


def test_randomized_roundtrips_deep_arrays():
    rng = random.Random(42)
    for _ in range(200):
        source = [_random_value(rng, 2) for _ in range(rng.randrange(6))]
        target = [_random_value(rng, 2) for _ in range(rng.randrange(6))]
        operations = make_patch(source, target)
        assert apply_patch(source, operations) == target


def test_equal_inputs_always_produce_empty_ops_under_fuzz():
    rng = random.Random(7)
    for _ in range(100):
        value = _random_value(rng, 3)
        assert make_patch(value, value) == []
