"""
EntitySchema declares the field layout of an entity dataset so the deduplication
pipeline knows which features it can compute.

Fields are tagged with semantic roles using simple flags:
  name_field=True    -> used for string similarity (Jaro-Winkler)
  unique_key=True    -> hard unique identifier (exact match = definite duplicate)
  dob=True           -> date of birth (exact match = hard signal)
  gender=True        -> gender concordance feature
  phone=True         -> phone normalisation + exact match (auto-detected from type)
  admin_level_1=True -> state / region / province boundary
  admin_level_2=True -> LGA / district / county boundary
  enumerator=True    -> records collected by same field agent (amplifies suspicion)

Any field not tagged is ignored by the deduplication engine.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import re
import pandas as pd
import yaml


class FieldRole(str, Enum):
    NAME = "name"
    ID = "id"
    PHONE = "phone"
    DATE_OF_BIRTH = "dob"
    GENDER = "gender"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    ADMIN_1 = "admin_level_1"
    ADMIN_2 = "admin_level_2"
    ENUMERATOR = "enumerator"
    IGNORED = "ignored"


@dataclass
class EntityField:
    name: str
    role: FieldRole
    unique_key: bool = False
    auto_inferred: bool = False


class EntitySchema:
    """
    Describes which columns in a dataset serve which semantic roles.

    Examples
    --------
    NIN registry (no GPS needed):
    >>> schema = EntitySchema(fields={
    ...     "nin":           EntityField("nin",           FieldRole.ID, unique_key=True),
    ...     "full_name":     EntityField("full_name",     FieldRole.NAME),
    ...     "date_of_birth": EntityField("date_of_birth", FieldRole.DATE_OF_BIRTH),
    ...     "gender":        EntityField("gender",        FieldRole.GENDER),
    ...     "state_origin":  EntityField("state_origin",  FieldRole.ADMIN_1),
    ... })

    Agricultural roster (full feature set):
    >>> schema = EntitySchema(fields={
    ...     "full_name":   EntityField("full_name",   FieldRole.NAME),
    ...     "phone":       EntityField("phone",        FieldRole.PHONE),
    ...     "state":       EntityField("state",        FieldRole.ADMIN_1),
    ...     "lga":         EntityField("lga",          FieldRole.ADMIN_2),
    ...     "latitude":    EntityField("latitude",     FieldRole.LATITUDE),
    ...     "longitude":   EntityField("longitude",    FieldRole.LONGITUDE),
    ... })

    Auto-infer from CSV (no YAML needed):
    >>> schema = EntitySchema.infer("records.csv")
    """

    # Column name hints for auto-inference
    _ROLE_HINTS: list[tuple[FieldRole, list[str]]] = [
        (FieldRole.ID,          ["nin", "nid", "national_id", "passport", "voter_id", "hospital_id", "applicant_id", "patient_id", "uid"]),
        (FieldRole.PHONE,       ["phone", "tel", "mobile", "gsm", "contact"]),
        (FieldRole.DATE_OF_BIRTH, ["dob", "date_of_birth", "birth_date", "birthdate"]),
        (FieldRole.GENDER,      ["gender", "sex"]),
        (FieldRole.LATITUDE,    ["lat", "latitude", "gps_lat", "y_coord"]),
        (FieldRole.LONGITUDE,   ["lon", "lng", "longitude", "gps_lon", "gps_lng", "x_coord"]),
        (FieldRole.ADMIN_1,     ["state", "region", "province", "county", "territory"]),
        (FieldRole.ADMIN_2,     ["lga", "district", "ward", "city", "municipality"]),
        (FieldRole.ENUMERATOR,  ["enumerator", "agent", "collector", "surveyor", "field_agent"]),
        (FieldRole.NAME,        ["name", "full_name", "fullname", "first_name", "last_name", "surname"]),
    ]

    def __init__(self, fields: dict[str, EntityField]):
        self.fields = fields

    @classmethod
    def infer(cls, source: str | Path | pd.DataFrame, sample_rows: int = 200) -> "EntitySchema":
        """Auto-infer schema from a CSV or DataFrame column names."""
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source, nrows=sample_rows)
        else:
            df = source.head(sample_rows)

        fields: dict[str, EntityField] = {}
        for col in df.columns:
            col_lower = col.lower().replace(" ", "_")
            role = FieldRole.IGNORED
            unique_key = False

            for r, hints in cls._ROLE_HINTS:
                if any(h in col_lower for h in hints):
                    role = r
                    break

            if role == FieldRole.ID:
                # Check if it's high-cardinality -> treat as unique_key
                unique_rate = df[col].nunique() / max(len(df), 1)
                unique_key = unique_rate > 0.9

            if role != FieldRole.IGNORED:
                fields[col] = EntityField(name=col, role=role, unique_key=unique_key, auto_inferred=True)

        return cls(fields=fields)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EntitySchema":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "EntitySchema":
        fields: dict[str, EntityField] = {}
        for name, cfg in data.get("fields", {}).items():
            fields[name] = EntityField(
                name=name,
                role=FieldRole(cfg.get("role", "ignored")),
                unique_key=cfg.get("unique_key", False),
            )
        return cls(fields=fields)

    def fields_by_role(self, role: FieldRole) -> list[str]:
        return [n for n, f in self.fields.items() if f.role == role]

    def has_role(self, role: FieldRole) -> bool:
        return any(f.role == role for f in self.fields.values())

    def active_features(self) -> set[str]:
        """Returns the set of features that can be computed given available roles."""
        active = set()
        if self.has_role(FieldRole.NAME):
            active.update(["jw_name_full", "jw_name_first", "jw_name_last"])
        if self.has_role(FieldRole.ID):
            active.add("exact_id")
        if self.has_role(FieldRole.PHONE):
            active.add("exact_phone")
        if self.has_role(FieldRole.LATITUDE) and self.has_role(FieldRole.LONGITUDE):
            active.add("haversine_m")
        if self.has_role(FieldRole.ADMIN_1):
            active.add("same_admin_1")
        if self.has_role(FieldRole.ADMIN_2):
            active.add("same_admin_2")
        if self.has_role(FieldRole.DATE_OF_BIRTH):
            active.add("dob_match")
        if self.has_role(FieldRole.GENDER):
            active.add("same_gender")
        if self.has_role(FieldRole.ENUMERATOR):
            active.add("same_enumerator")
        return active
