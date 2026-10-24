export function jaroWinkler(s1: string, s2: string): number {
  if (s1 === s2) return 1.0;
  const len1 = s1.length;
  const len2 = s2.length;
  if (len1 === 0 || len2 === 0) return 0.0;

  const matchDistance = Math.floor(Math.max(len1, len2) / 2) - 1;
  const s1Matches = new Array(len1).fill(false);
  const s2Matches = new Array(len2).fill(false);

  let matches = 0;
  for (let i = 0; i < len1; i++) {
    const start = Math.max(0, i - matchDistance);
    const end = Math.min(i + matchDistance + 1, len2);
    for (let j = start; j < end; j++) {
      if (s2Matches[j] || s1[i] !== s2[j]) continue;
      s1Matches[i] = true;
      s2Matches[j] = true;
      matches++;
      break;
    }
  }

  if (matches === 0) return 0.0;

  let transpositions = 0;
  let k = 0;
  for (let i = 0; i < len1; i++) {
    if (!s1Matches[i]) continue;
    while (!s2Matches[k]) k++;
    if (s1[i] !== s2[k]) transpositions++;
    k++;
  }

  const jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3;
  let prefix = 0;
  for (let i = 0; i < Math.min(4, Math.min(len1, len2)); i++) {
    if (s1[i] === s2[i]) prefix++;
    else break;
  }

  return jaro + prefix * 0.1 * (1 - jaro);
}

export function haversineM(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371000.0; // Earth radius in meters
  const toRad = (deg: number) => (deg * Math.PI) / 180.0;

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}

export function extractPairFeatures(
  a: Record<string, string | number | null | undefined>,
  b: Record<string, string | number | null | undefined>
): Record<string, number | null> {
  const feats: Record<string, number | null> = {};

  // Name similarities
  const nameA = String(a["full_name"] ?? a["name"] ?? "").trim().toLowerCase();
  const nameB = String(b["full_name"] ?? b["name"] ?? "").trim().toLowerCase();

  if (nameA && nameB) {
    feats["jw_name_full"] = jaroWinkler(nameA, nameB);
    const tokensA = nameA.split(/\s+/);
    const tokensB = nameB.split(/\s+/);
    feats["jw_name_first"] = jaroWinkler(tokensA[0] ?? "", tokensB[0] ?? "");
    feats["jw_name_last"] = jaroWinkler(
      tokensA[tokensA.length - 1] ?? "",
      tokensB[tokensB.length - 1] ?? ""
    );
  } else {
    feats["jw_name_full"] = null;
    feats["jw_name_first"] = null;
    feats["jw_name_last"] = null;
  }

  // Exact Phone Match
  const phoneA = String(a["phone"] ?? a["phone_number"] ?? "").replace(/\D/g, "");
  const phoneB = String(b["phone"] ?? b["phone_number"] ?? "").replace(/\D/g, "");
  if (phoneA.length >= 7 && phoneB.length >= 7) {
    feats["exact_phone"] = phoneA === phoneB ? 1.0 : 0.0;
  } else {
    feats["exact_phone"] = null;
  }

  // Haversine GPS distance
  const latA = Number(a["latitude"] ?? a["lat"]);
  const lonA = Number(a["longitude"] ?? a["lon"]);
  const latB = Number(b["latitude"] ?? b["lat"]);
  const lonB = Number(b["longitude"] ?? b["lon"]);

  if (!isNaN(latA) && !isNaN(lonA) && !isNaN(latB) && !isNaN(lonB)) {
    feats["haversine_m"] = haversineM(latA, lonA, latB, lonB);
  } else {
    feats["haversine_m"] = null;
  }

  return feats;
}
