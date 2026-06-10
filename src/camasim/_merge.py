"""
Merge strategies for CAM search results.

horizontal_merge: combines results across column splits for a single row
vertical_merge: combines matched row indices across all rows
"""


def horizontal_and(col_results: list[tuple[bool, float, float]]) -> tuple[bool, float, float]:
    """A row matches only if it matches in ALL column splits."""
    matched = all(r[0] for r in col_results)
    energy = sum(r[1] for r in col_results)
    latency = sum(r[2] for r in col_results)
    return matched, energy, latency


def vertical_union(per_row_matches: list[list[int]]) -> list[int]:
    """Collect all matched row indices."""
    return [idx for matches in per_row_matches for idx in matches]
