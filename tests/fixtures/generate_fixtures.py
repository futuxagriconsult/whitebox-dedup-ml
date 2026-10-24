"""
Synthetic fixture generator for whitebox-dedup tests.

Run this script to regenerate all fixture CSV files:
  python tests/fixtures/generate_fixtures.py

All data is fully synthetic. No real personal records are used or stored.
Seed is fixed at 42 for reproducibility.
"""
import csv
import random
import math
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).parent

# Try to import faker; fall back to minimal synthetic data if unavailable
try:
    from faker import Faker
    faker = Faker(["en_NG"])
    faker.seed_instance(SEED)
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False

rng = random.Random(SEED)

# ---- Nigerian-flavoured name pools ----
FIRST_NAMES = [
    "Aminu", "Bello", "Fatima", "Musa", "Aisha", "Ibrahim", "Halima", "Usman",
    "Zainab", "Suleiman", "Hauwa", "Kabir", "Ramatu", "Yusuf", "Aminatu",
    "Aliyu", "Maryam", "Mustapha", "Falmata", "Adamu", "Saratu", "Garba",
    "Bilkisu", "Haruna", "Ladi", "Tanimu", "Hadiza", "Sani", "Rakiya", "Lawal",
]
LAST_NAMES = [
    "Abubakar", "Mohammed", "Abdullahi", "Yakubu", "Hassan", "Danladi", "Shehu",
    "Umar", "Isah", "Tukur", "Makama", "Rabiu", "Bello", "Idris", "Danmusa",
    "Waziri", "Gwarzo", "Dauda", "Lawan", "Maikudi",
]
STATES = ["Kano", "Kaduna", "Sokoto", "Zamfara", "Katsina", "Bauchi", "Gombe", "Jigawa", "Kebbi", "Niger"]
LGAS = ["Kano Municipal", "Kaduna North", "Sokoto North", "Gusau", "Daura", "Bauchi", "Gombe", "Dutse", "Birnin Kebbi", "Bida"]
GENDERS = ["M", "F"]

# GPS bounding box: northern Nigeria approximate
LAT_MIN, LAT_MAX = 10.0, 14.0
LON_MIN, LON_MAX = 3.5, 13.5


def random_name() -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def random_phone() -> str:
    return f"0{rng.randint(70, 90)}{rng.randint(10000000, 99999999)}"


def random_gps() -> tuple[float, float]:
    lat = rng.uniform(LAT_MIN, LAT_MAX)
    lon = rng.uniform(LON_MIN, LON_MAX)
    return round(lat, 6), round(lon, 6)


def random_dob() -> str:
    year = rng.randint(1960, 2000)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def random_nin() -> str:
    return f"NIN{rng.randint(10000000, 99999999)}"


def perturb_name(name: str) -> str:
    """Introduce a small typo into a name (simulates data entry error)."""
    tokens = name.split()
    token = rng.choice(tokens)
    if len(token) > 3:
        i = rng.randint(1, len(token) - 2)
        token = token[:i] + rng.choice("aeiou") + token[i + 1:]
    idx = tokens.index(rng.choice(tokens))
    tokens[idx] = token
    return " ".join(tokens)


def nearby_gps(lat: float, lon: float, max_m: float = 30.0) -> tuple[float, float]:
    """Return a GPS coordinate within max_m metres."""
    delta_deg = max_m / 111_000.0
    dlat = rng.uniform(-delta_deg, delta_deg)
    dlon = rng.uniform(-delta_deg, delta_deg)
    return round(lat + dlat, 6), round(lon + dlon, 6)


# ---- Fixture 1: pairs_farmer_labelled.csv ----

def generate_pairs_farmer_labelled(n_dup: int = 50, n_dist: int = 50) -> None:
    path = OUTPUT_DIR / "pairs_farmer_labelled.csv"
    rows = []

    # Duplicate pairs
    for _ in range(n_dup):
        name = random_name()
        phone = random_phone()
        lat, lon = random_gps()
        state = rng.choice(STATES)
        # Duplicate: same or very similar name, same phone or close GPS
        variant = rng.choice(["same_name", "perturb_name"])
        name_b = name if variant == "same_name" else perturb_name(name)
        phone_b = phone if rng.random() > 0.3 else random_phone()
        lat_b, lon_b = nearby_gps(lat, lon, max_m=25) if rng.random() > 0.4 else (lat, lon)
        rows.append({
            "name_a": name, "phone_a": phone, "lat_a": lat, "lon_a": lon, "state_a": state,
            "name_b": name_b, "phone_b": phone_b, "lat_b": lat_b, "lon_b": lon_b, "state_b": state,
            "is_duplicate": 1,
        })

    # Distinct pairs
    for _ in range(n_dist):
        name_a = random_name()
        # Ensure different last name for distinct
        name_b = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        # Keep regenerating if names are too similar
        attempts = 0
        while name_a.split()[0] == name_b.split()[0] and attempts < 5:
            name_b = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            attempts += 1
        gps_a = random_gps()
        gps_b = random_gps()
        rows.append({
            "name_a": name_a, "phone_a": random_phone(), "lat_a": gps_a[0], "lon_a": gps_a[1], "state_a": rng.choice(STATES),
            "name_b": name_b, "phone_b": random_phone(), "lat_b": gps_b[0], "lon_b": gps_b[1], "state_b": rng.choice(STATES),
            "is_duplicate": 0,
        })

    fieldnames = ["name_a", "phone_a", "lat_a", "lon_a", "state_a",
                  "name_b", "phone_b", "lat_b", "lon_b", "state_b", "is_duplicate"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path} ({len(rows)} rows)")


# ---- Fixture 2: pairs_nin_labelled.csv ----

def generate_pairs_nin_labelled(n_dup: int = 30, n_dist: int = 30) -> None:
    path = OUTPUT_DIR / "pairs_nin_labelled.csv"
    rows = []

    # Duplicate pairs
    for _ in range(n_dup):
        nin = random_nin()
        name = random_name()
        dob = random_dob()
        gender = rng.choice(GENDERS)
        same_nin = rng.random() > 0.4  # 60% have same NIN (hard signal), 40% same DOB + similar name
        nin_b = nin if same_nin else random_nin()
        name_b = name if rng.random() > 0.3 else perturb_name(name)
        dob_b = dob  # same DOB for duplicates
        rows.append({
            "nin_a": nin, "name_a": name, "dob_a": dob, "gender_a": gender,
            "nin_b": nin_b, "name_b": name_b, "dob_b": dob_b, "gender_b": gender,
            "is_duplicate": 1,
        })

    # Distinct pairs
    for _ in range(n_dist):
        rows.append({
            "nin_a": random_nin(), "name_a": random_name(), "dob_a": random_dob(), "gender_a": rng.choice(GENDERS),
            "nin_b": random_nin(), "name_b": random_name(), "dob_b": random_dob(), "gender_b": rng.choice(GENDERS),
            "is_duplicate": 0,
        })

    fieldnames = ["nin_a", "name_a", "dob_a", "gender_a", "nin_b", "name_b", "dob_b", "gender_b", "is_duplicate"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path} ({len(rows)} rows)")


# ---- Fixture 3: registry_farmer_200.csv ----
# Known duplicate pairs (farmer_id pairs):
#   (F001, F002), (F015, F016), (F030, F031), (F045, F046),
#   (F060, F061), (F075, F076), (F090, F091)
# Total: 7 duplicate pairs embedded in 200 records.

KNOWN_FARMER_DUPS = [
    ("F001", "F002"),
    ("F015", "F016"),
    ("F030", "F031"),
    ("F045", "F046"),
    ("F060", "F061"),
    ("F075", "F076"),
    ("F090", "F091"),
]


def generate_registry_farmer_200() -> None:
    path = OUTPUT_DIR / "registry_farmer_200.csv"
    records = []
    dup_ids = {fid for pair in KNOWN_FARMER_DUPS for fid in pair}

    # Build a lookup of the duplicate pairs: second ID -> first record
    dup_source: dict[str, dict] = {}
    dup_targets: dict[str, str] = {b: a for a, b in KNOWN_FARMER_DUPS}

    farmer_id_counter = 1

    def next_fid() -> str:
        nonlocal farmer_id_counter
        fid = f"F{farmer_id_counter:03d}"
        farmer_id_counter += 1
        return fid

    used_ids: set[str] = set()

    for i in range(200):
        fid = next_fid()
        used_ids.add(fid)

        if fid in dup_targets:
            # This record is the duplicate of a previous one
            source_fid = dup_targets[fid]
            src = next(r for r in records if r["farmer_id"] == source_fid)
            # Slightly perturb the name to simulate a data entry variant
            records.append({
                "farmer_id": fid,
                "full_name": perturb_name(src["full_name"]),
                "phone_number": src["phone_number"],
                "state": src["state"],
                "lga": src["lga"],
                "latitude": nearby_gps(src["latitude"], src["longitude"], max_m=20)[0],
                "longitude": nearby_gps(src["latitude"], src["longitude"], max_m=20)[1],
            })
        else:
            lat, lon = random_gps()
            state = rng.choice(STATES)
            lga = rng.choice(LGAS)
            records.append({
                "farmer_id": fid,
                "full_name": random_name(),
                "phone_number": random_phone(),
                "state": state,
                "lga": lga,
                "latitude": lat,
                "longitude": lon,
            })

    fieldnames = ["farmer_id", "full_name", "phone_number", "state", "lga", "latitude", "longitude"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Written: {path} ({len(records)} rows)")
    print(f"  Known duplicate pairs: {KNOWN_FARMER_DUPS}")


# ---- Fixture 4: registry_nin_300.csv ----
# Known exact NIN duplicates (NIN appears exactly twice):
#   DUP_NIN_001, DUP_NIN_002, DUP_NIN_003, DUP_NIN_004, DUP_NIN_005
# Plus some near-duplicate names without NIN match.

KNOWN_DUP_NINS = [
    "DUP_NIN_001",
    "DUP_NIN_002",
    "DUP_NIN_003",
    "DUP_NIN_004",
    "DUP_NIN_005",
]


def generate_registry_nin_300() -> None:
    path = OUTPUT_DIR / "registry_nin_300.csv"
    records = []

    # First pass: add 290 unique NIN records
    for i in range(290):
        records.append({
            "nin": random_nin(),
            "full_name": random_name(),
            "date_of_birth": random_dob(),
            "gender": rng.choice(GENDERS),
            "state_of_origin": rng.choice(STATES),
        })

    # Insert 5 exact NIN duplicates (each NIN appears twice)
    # Replace records at known positions with duplicate NIN records
    dup_positions_a = [10, 40, 70, 100, 130]
    for nin_val, pos_a in zip(KNOWN_DUP_NINS, dup_positions_a):
        name = random_name()
        dob = random_dob()
        gender = rng.choice(GENDERS)
        state = rng.choice(STATES)
        records[pos_a] = {
            "nin": nin_val,
            "full_name": name,
            "date_of_birth": dob,
            "gender": gender,
            "state_of_origin": state,
        }
        # Second occurrence: same NIN, slightly different name
        records.append({
            "nin": nin_val,
            "full_name": perturb_name(name),
            "date_of_birth": dob,
            "gender": gender,
            "state_of_origin": state,
        })

    fieldnames = ["nin", "full_name", "date_of_birth", "gender", "state_of_origin"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Written: {path} ({len(records)} rows)")
    print(f"  Known duplicate NINs: {KNOWN_DUP_NINS}")


def main() -> None:
    print(f"Generating fixtures in: {OUTPUT_DIR}")
    print(f"Random seed: {SEED}")
    print()
    generate_pairs_farmer_labelled()
    generate_pairs_nin_labelled()
    generate_registry_farmer_200()
    generate_registry_nin_300()
    print()
    print("All fixtures generated successfully.")


if __name__ == "__main__":
    main()
