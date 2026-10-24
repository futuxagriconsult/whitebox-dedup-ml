"""
Pairwise feature extraction for entity deduplication.

All features take two record dicts (row_a, row_b) and an EntitySchema,
and return a float value. Missing columns return None (feature skipped).
"""
import math
import re
from typing import Optional
import jellyfish


def jw_name_full(row_a: dict, row_b: dict, name_cols: list[str]) -> Optional[float]:
    """Jaro-Winkler similarity on concatenated name tokens."""
    if not name_cols:
        return None
    a = " ".join(str(row_a.get(c, "") or "").strip() for c in name_cols).lower()
    b = " ".join(str(row_b.get(c, "") or "").strip() for c in name_cols).lower()
    if not a or not b:
        return None
    return jellyfish.jaro_winkler_similarity(a, b)


def jw_name_first(row_a: dict, row_b: dict, name_cols: list[str]) -> Optional[float]:
    """Jaro-Winkler on first token of primary name field."""
    if not name_cols:
        return None
    a_tokens = str(row_a.get(name_cols[0], "") or "").split()
    b_tokens = str(row_b.get(name_cols[0], "") or "").split()
    if not a_tokens or not b_tokens:
        return None
    return jellyfish.jaro_winkler_similarity(a_tokens[0].lower(), b_tokens[0].lower())


def jw_name_last(row_a: dict, row_b: dict, name_cols: list[str]) -> Optional[float]:
    """Jaro-Winkler on last token of primary name field."""
    if not name_cols:
        return None
    a_tokens = str(row_a.get(name_cols[0], "") or "").split()
    b_tokens = str(row_b.get(name_cols[0], "") or "").split()
    if not a_tokens or not b_tokens:
        return None
    return jellyfish.jaro_winkler_similarity(a_tokens[-1].lower(), b_tokens[-1].lower())


def haversine_m(row_a: dict, row_b: dict, lat_col: str, lon_col: str) -> Optional[float]:
    """Great-circle distance in metres between two GPS coordinates."""
    try:
        lat1, lon1 = float(row_a[lat_col]), float(row_a[lon_col])
        lat2, lon2 = float(row_b[lat_col]), float(row_b[lon_col])
    except (KeyError, TypeError, ValueError):
        return None
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize_phone(phone: str) -> str:
    return re.sub(r"[\s\-().+]", "", str(phone or ""))


def exact_phone(row_a: dict, row_b: dict, phone_col: str) -> Optional[float]:
    """1.0 if phone numbers match after normalisation, 0.0 otherwise."""
    a = _normalize_phone(row_a.get(phone_col, ""))
    b = _normalize_phone(row_b.get(phone_col, ""))
    if not a or not b:
        return None
    return 1.0 if a == b else 0.0


def exact_id(row_a: dict, row_b: dict, id_col: str) -> Optional[float]:
    """1.0 if unique ID values match, 0.0 if both non-null and different."""
    a = str(row_a.get(id_col, "") or "").strip()
    b = str(row_b.get(id_col, "") or "").strip()
    if not a or not b:
        return None
    return 1.0 if a.lower() == b.lower() else 0.0


def same_admin(row_a: dict, row_b: dict, col: str) -> Optional[float]:
    """1.0 if same administrative unit, 0.0 otherwise."""
    a = str(row_a.get(col, "") or "").strip().lower()
    b = str(row_b.get(col, "") or "").strip().lower()
    if not a or not b:
        return None
    return 1.0 if a == b else 0.0


def dob_match(row_a: dict, row_b: dict, dob_col: str) -> Optional[float]:
    """1.0 if date of birth strings match exactly (after stripping)."""
    a = str(row_a.get(dob_col, "") or "").strip()
    b = str(row_b.get(dob_col, "") or "").strip()
    if not a or not b:
        return None
    return 1.0 if a == b else 0.0


def same_gender(row_a: dict, row_b: dict, gender_col: str) -> Optional[float]:
    """1.0 if gender values match (normalised to first letter)."""
    a = str(row_a.get(gender_col, "") or "")[:1].upper()
    b = str(row_b.get(gender_col, "") or "")[:1].upper()
    if not a or not b:
        return None
    return 1.0 if a == b else 0.0


def same_enumerator(row_a: dict, row_b: dict, enum_col: str) -> Optional[float]:
    """1.0 if records were collected by the same field agent."""
    a = str(row_a.get(enum_col, "") or "").strip()
    b = str(row_b.get(enum_col, "") or "").strip()
    if not a or not b:
        return None
    return 1.0 if a == b else 0.0


def extract_all(row_a: dict, row_b: dict, schema) -> dict[str, Optional[float]]:
    """
    Extract all available features for a pair given the EntitySchema.
    Returns a dict of feature_name -> value (None if feature unavailable).
    """
    from .schema import FieldRole
    result: dict[str, Optional[float]] = {}
    active = schema.active_features()

    name_cols = schema.fields_by_role(FieldRole.NAME)
    if "jw_name_full"  in active: result["jw_name_full"]  = jw_name_full(row_a, row_b, name_cols)
    if "jw_name_first" in active: result["jw_name_first"] = jw_name_first(row_a, row_b, name_cols)
    if "jw_name_last"  in active: result["jw_name_last"]  = jw_name_last(row_a, row_b, name_cols)

    id_cols = schema.fields_by_role(FieldRole.ID)
    if "exact_id" in active and id_cols:
        result["exact_id"] = exact_id(row_a, row_b, id_cols[0])

    phone_cols = schema.fields_by_role(FieldRole.PHONE)
    if "exact_phone" in active and phone_cols:
        result["exact_phone"] = exact_phone(row_a, row_b, phone_cols[0])

    lat_cols = schema.fields_by_role(FieldRole.LATITUDE)
    lon_cols = schema.fields_by_role(FieldRole.LONGITUDE)
    if "haversine_m" in active and lat_cols and lon_cols:
        result["haversine_m"] = haversine_m(row_a, row_b, lat_cols[0], lon_cols[0])

    a1_cols = schema.fields_by_role(FieldRole.ADMIN_1)
    if "same_admin_1" in active and a1_cols:
        result["same_admin_1"] = same_admin(row_a, row_b, a1_cols[0])

    a2_cols = schema.fields_by_role(FieldRole.ADMIN_2)
    if "same_admin_2" in active and a2_cols:
        result["same_admin_2"] = same_admin(row_a, row_b, a2_cols[0])

    dob_cols = schema.fields_by_role(FieldRole.DATE_OF_BIRTH)
    if "dob_match" in active and dob_cols:
        result["dob_match"] = dob_match(row_a, row_b, dob_cols[0])

    gender_cols = schema.fields_by_role(FieldRole.GENDER)
    if "same_gender" in active and gender_cols:
        result["same_gender"] = same_gender(row_a, row_b, gender_cols[0])

    enum_cols = schema.fields_by_role(FieldRole.ENUMERATOR)
    if "same_enumerator" in active and enum_cols:
        result["same_enumerator"] = same_enumerator(row_a, row_b, enum_cols[0])

    return result
