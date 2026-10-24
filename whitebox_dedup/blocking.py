"""
Blocking strategies to generate candidate duplicate pairs efficiently.

Without blocking, comparing every pair in N records requires N*(N-1)/2 comparisons.
Blocking groups records into buckets where duplicates are likely to be found,
then only compares records within the same bucket.
"""
import math
from typing import Iterator
import pandas as pd
from .schema import EntitySchema, FieldRole


def generate_candidate_pairs(
    df: pd.DataFrame,
    schema: EntitySchema,
    geo_radius_m: float = 50.0,
) -> list[tuple[int, int]]:
    """
    Generate candidate duplicate pairs using available blocking strategies.
    Returns list of (row_index_a, row_index_b) tuples.
    Deduplicates pairs (always returns (i, j) where i < j).
    """
    pairs: set[tuple[int, int]] = set()

    # Strategy 1: First-letter blocking on name fields
    name_cols = schema.fields_by_role(FieldRole.NAME)
    if name_cols:
        pairs.update(_name_block(df, name_cols[0]))

    # Strategy 2: Exact phone blocking
    phone_cols = schema.fields_by_role(FieldRole.PHONE)
    if phone_cols:
        pairs.update(_exact_block(df, phone_cols[0]))

    # Strategy 3: Exact ID blocking (to detect hard duplicates)
    id_cols = schema.fields_by_role(FieldRole.ID)
    if id_cols:
        pairs.update(_exact_block(df, id_cols[0]))

    # Strategy 4: Spatial blocking (haversine <= geo_radius_m)
    lat_cols = schema.fields_by_role(FieldRole.LATITUDE)
    lon_cols = schema.fields_by_role(FieldRole.LONGITUDE)
    if lat_cols and lon_cols:
        pairs.update(_geo_block(df, lat_cols[0], lon_cols[0], geo_radius_m))

    return sorted(pairs)


def _name_block(df: pd.DataFrame, col: str) -> set[tuple[int, int]]:
    """Block by first 2 characters of name field."""
    if col not in df.columns:
        return set()
    buckets: dict[str, list[int]] = {}
    for idx, val in df[col].items():
        key = str(val or "")[:2].lower()
        if key:
            buckets.setdefault(key, []).append(idx)
    return _pairs_from_buckets(buckets)


def _exact_block(df: pd.DataFrame, col: str) -> set[tuple[int, int]]:
    """Block by exact value match (detects hard duplicates)."""
    if col not in df.columns:
        return set()
    buckets: dict[str, list[int]] = {}
    for idx, val in df[col].items():
        key = str(val or "").strip().lower()
        if key:
            buckets.setdefault(key, []).append(idx)
    # Only keep buckets with > 1 record (actual duplicates possible)
    return _pairs_from_buckets({k: v for k, v in buckets.items() if len(v) > 1})


def _geo_block(df: pd.DataFrame, lat_col: str, lon_col: str, radius_m: float) -> set[tuple[int, int]]:
    """
    Spatial blocking: pair records within radius_m metres of each other.
    Uses a simple grid cell approach: approximately 0.001 degree = 111m resolution.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        return set()

    cell_deg = radius_m / 111_000.0
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, row in df[[lat_col, lon_col]].iterrows():
        try:
            lat, lon = float(row[lat_col]), float(row[lon_col])
            cell = (int(lat / cell_deg), int(lon / cell_deg))
            buckets.setdefault(cell, []).append(idx)
        except (ValueError, TypeError):
            continue

    # Also check adjacent cells
    pairs: set[tuple[int, int]] = set()
    cells = list(buckets.keys())
    for i, cell in enumerate(cells):
        members = buckets[cell]
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                neighbor = (cell[0] + di, cell[1] + dj)
                if neighbor in buckets:
                    for a in members:
                        for b in buckets[neighbor]:
                            if a < b:
                                pairs.add((a, b))
    return pairs


def _pairs_from_buckets(buckets: dict) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((min(members[i], members[j]), max(members[i], members[j])))
    return pairs
