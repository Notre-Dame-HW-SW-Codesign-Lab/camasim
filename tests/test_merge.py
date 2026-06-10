"""Unit tests for the merge strategies (pure logic, no EvaCAM)."""

from camasim._merge import horizontal_and, vertical_union


def test_horizontal_and_all_hits():
    # (matched, energy, latency) per column split.
    results = [(True, 1.0, 2.0), (True, 0.5, 1.0)]
    matched, energy, latency = horizontal_and(results)
    assert matched is True
    assert energy == 1.5
    assert latency == 3.0


def test_horizontal_and_one_miss_fails():
    results = [(True, 1.0, 2.0), (False, 0.5, 1.0)]
    matched, energy, latency = horizontal_and(results)
    assert matched is False
    # Costs still accumulate across all splits even on a miss.
    assert energy == 1.5
    assert latency == 3.0


def test_horizontal_and_single_split():
    assert horizontal_and([(True, 2.0, 3.0)]) == (True, 2.0, 3.0)


def test_vertical_union_flattens_across_queries():
    # Each inner list is the matched row indices for one query.
    assert vertical_union([[0, 2], [], [5]]) == [0, 2, 5]


def test_vertical_union_empty():
    assert vertical_union([[], []]) == []
