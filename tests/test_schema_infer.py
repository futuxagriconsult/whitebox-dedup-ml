"""
Tests for EntitySchema: auto-inference, role detection, and active feature computation.
"""
import pandas as pd
import pytest
from whitebox_dedup.schema import EntitySchema, EntityField, FieldRole


# ---- Auto-inference: role detection ----

def test_infer_detects_nin_as_id_role():
    df = pd.DataFrame({"nin": [f"NIN{i:08d}" for i in range(100)]})
    schema = EntitySchema.infer(df)
    assert "nin" in schema.fields
    assert schema.fields["nin"].role == FieldRole.ID


def test_infer_detects_unique_nin_as_unique_key():
    df = pd.DataFrame({"nin": [f"NIN{i:08d}" for i in range(100)]})
    schema = EntitySchema.infer(df)
    assert schema.fields["nin"].unique_key is True


def test_infer_detects_non_unique_id_not_as_unique_key():
    """If there are many duplicates in the ID column, unique_key should be False."""
    df = pd.DataFrame({"nin": ["NIN00000001"] * 100})
    schema = EntitySchema.infer(df)
    assert schema.fields["nin"].unique_key is False


def test_infer_detects_latitude_column():
    df = pd.DataFrame({
        "latitude": [10.0, 11.0, 12.0],
        "longitude": [7.0, 8.0, 9.0],
    })
    schema = EntitySchema.infer(df)
    assert "latitude" in schema.fields
    assert schema.fields["latitude"].role == FieldRole.LATITUDE


def test_infer_detects_longitude_column():
    df = pd.DataFrame({
        "latitude": [10.0, 11.0, 12.0],
        "longitude": [7.0, 8.0, 9.0],
    })
    schema = EntitySchema.infer(df)
    assert "longitude" in schema.fields
    assert schema.fields["longitude"].role == FieldRole.LONGITUDE


def test_infer_detects_name_column():
    df = pd.DataFrame({"full_name": ["Aminu Bello", "Fatima Mohammed"]})
    schema = EntitySchema.infer(df)
    assert "full_name" in schema.fields
    assert schema.fields["full_name"].role == FieldRole.NAME


def test_infer_detects_phone_column():
    df = pd.DataFrame({"phone": ["08012345678", "09098765432"]})
    schema = EntitySchema.infer(df)
    assert "phone" in schema.fields
    assert schema.fields["phone"].role == FieldRole.PHONE


def test_infer_detects_dob_column():
    df = pd.DataFrame({"date_of_birth": ["1990-01-01", "1985-06-15"]})
    schema = EntitySchema.infer(df)
    assert "date_of_birth" in schema.fields
    assert schema.fields["date_of_birth"].role == FieldRole.DATE_OF_BIRTH


def test_infer_detects_gender_column():
    df = pd.DataFrame({"gender": ["M", "F", "M"]})
    schema = EntitySchema.infer(df)
    assert "gender" in schema.fields
    assert schema.fields["gender"].role == FieldRole.GENDER


def test_infer_detects_state_as_admin_1():
    df = pd.DataFrame({"state": ["Kano", "Kaduna"]})
    schema = EntitySchema.infer(df)
    assert "state" in schema.fields
    assert schema.fields["state"].role == FieldRole.ADMIN_1


def test_infer_detects_lga_as_admin_2():
    df = pd.DataFrame({"lga": ["Kano Municipal", "Kaduna North"]})
    schema = EntitySchema.infer(df)
    assert "lga" in schema.fields
    assert schema.fields["lga"].role == FieldRole.ADMIN_2


def test_infer_ignores_unknown_columns():
    df = pd.DataFrame({"random_col_xyz": [1, 2, 3]})
    schema = EntitySchema.infer(df)
    assert "random_col_xyz" not in schema.fields


def test_infer_marks_fields_as_auto_inferred():
    df = pd.DataFrame({"full_name": ["Aminu Bello"]})
    schema = EntitySchema.infer(df)
    assert schema.fields["full_name"].auto_inferred is True


# ---- active_features ----

def test_active_features_name_only():
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
    })
    active = schema.active_features()
    assert "jw_name_full" in active
    assert "jw_name_first" in active
    assert "jw_name_last" in active
    assert "exact_phone" not in active
    assert "haversine_m" not in active


def test_active_features_with_gps():
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
        "latitude": EntityField("latitude", FieldRole.LATITUDE),
        "longitude": EntityField("longitude", FieldRole.LONGITUDE),
    })
    active = schema.active_features()
    assert "haversine_m" in active


def test_active_features_gps_requires_both_lat_and_lon():
    """haversine_m must NOT be active if only latitude (no longitude) is present."""
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
        "latitude": EntityField("latitude", FieldRole.LATITUDE),
    })
    active = schema.active_features()
    assert "haversine_m" not in active


def test_active_features_with_id():
    schema = EntitySchema(fields={
        "nin": EntityField("nin", FieldRole.ID, unique_key=True),
    })
    active = schema.active_features()
    assert "exact_id" in active


def test_active_features_with_phone():
    schema = EntitySchema(fields={
        "phone": EntityField("phone", FieldRole.PHONE),
    })
    active = schema.active_features()
    assert "exact_phone" in active


def test_active_features_with_dob():
    schema = EntitySchema(fields={
        "dob": EntityField("dob", FieldRole.DATE_OF_BIRTH),
    })
    active = schema.active_features()
    assert "dob_match" in active


def test_active_features_full_nin_schema():
    schema = EntitySchema(fields={
        "nin":           EntityField("nin",           FieldRole.ID, unique_key=True),
        "full_name":     EntityField("full_name",     FieldRole.NAME),
        "date_of_birth": EntityField("date_of_birth", FieldRole.DATE_OF_BIRTH),
        "gender":        EntityField("gender",        FieldRole.GENDER),
        "state_origin":  EntityField("state_origin",  FieldRole.ADMIN_1),
    })
    active = schema.active_features()
    assert "exact_id" in active
    assert "jw_name_full" in active
    assert "dob_match" in active
    assert "same_gender" in active
    assert "same_admin_1" in active
    assert "haversine_m" not in active  # no GPS in NIN schema


def test_active_features_full_agricultural_schema():
    schema = EntitySchema(fields={
        "full_name":  EntityField("full_name",  FieldRole.NAME),
        "phone":      EntityField("phone",      FieldRole.PHONE),
        "state":      EntityField("state",      FieldRole.ADMIN_1),
        "lga":        EntityField("lga",        FieldRole.ADMIN_2),
        "latitude":   EntityField("latitude",   FieldRole.LATITUDE),
        "longitude":  EntityField("longitude",  FieldRole.LONGITUDE),
    })
    active = schema.active_features()
    assert "jw_name_full" in active
    assert "exact_phone" in active
    assert "haversine_m" in active
    assert "same_admin_1" in active
    assert "same_admin_2" in active


# ---- fields_by_role ----

def test_fields_by_role_returns_correct_columns():
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
        "nickname":  EntityField("nickname",  FieldRole.NAME),
        "phone":     EntityField("phone",     FieldRole.PHONE),
    })
    name_fields = schema.fields_by_role(FieldRole.NAME)
    assert "full_name" in name_fields
    assert "nickname" in name_fields
    assert "phone" not in name_fields


def test_fields_by_role_empty_when_no_match():
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
    })
    gps_fields = schema.fields_by_role(FieldRole.LATITUDE)
    assert gps_fields == []


# ---- has_role ----

def test_has_role_returns_true_when_present():
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
    })
    assert schema.has_role(FieldRole.NAME) is True


def test_has_role_returns_false_when_absent():
    schema = EntitySchema(fields={
        "full_name": EntityField("full_name", FieldRole.NAME),
    })
    assert schema.has_role(FieldRole.PHONE) is False


# ---- from_dict ----

def test_from_dict_creates_correct_schema():
    data = {
        "fields": {
            "nin":       {"role": "id",   "unique_key": True},
            "full_name": {"role": "name", "unique_key": False},
        }
    }
    schema = EntitySchema.from_dict(data)
    assert "nin" in schema.fields
    assert schema.fields["nin"].role == FieldRole.ID
    assert schema.fields["nin"].unique_key is True
    assert schema.fields["full_name"].role == FieldRole.NAME
