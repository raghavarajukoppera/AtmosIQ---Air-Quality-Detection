from __future__ import annotations

from typing import Mapping


# CPCB AQI breakpoint tables.
_AQI_INDEX_BANDS = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
_POLLUTANT_BREAKPOINTS = {
    "pm25": [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 350)],  # ug/m3
    "pm10": [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 600)],  # ug/m3
    "no2": [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 600)],  # ug/m3
    "so2": [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 2000)],  # ug/m3
    "co": [(0.0, 1.0), (1.1, 2.0), (2.1, 10.0), (10.1, 17.0), (17.1, 34.0), (34.1, 50.0)],  # mg/m3
    "o3": [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 1000)],  # ug/m3
}

_MOLECULAR_WEIGHTS = {
    "co": 28.01,
    "no2": 46.0055,
    "so2": 64.066,
    "o3": 48.0,
}


def _normalize_unit(unit: str | None) -> str:
    if unit is None:
        return ""

    normalized = unit.lower()
    for micro in ("\u00b5", "\u03bc"):
        normalized = normalized.replace(micro, "u")
    normalized = normalized.replace("\u00b3", "3")
    normalized = normalized.replace("^3", "3").replace(" ", "")
    return normalized


def _ppb_to_ugm3(ppb: float, pollutant: str) -> float:
    return ppb * _MOLECULAR_WEIGHTS[pollutant] / 24.45


def convert_to_standard(parameter: str, value: float, unit: str | None) -> float:
    """Convert raw pollutant value to AQI standard units.

    Standards used:
    - pm25/pm10/no2/so2/o3 in ug/m3
    - co in mg/m3
    """
    p = parameter.lower()
    u = _normalize_unit(unit)

    if p in ("pm25", "pm10"):
        if "mg/m3" in u:
            return value * 1000.0
        return value

    if p in ("no2", "so2", "o3"):
        if "ppb" in u:
            return _ppb_to_ugm3(value, p)
        if "mg/m3" in u:
            return value * 1000.0
        return value

    if p == "co":
        if "ppb" in u:
            return _ppb_to_ugm3(value, p) / 1000.0
        if "ug/m3" in u:
            return value / 1000.0
        return value

    return value


def compute_sub_index(parameter: str, standardized_value: float) -> float:
    if standardized_value < 0:
        standardized_value = 0.0

    p = parameter.lower()
    breakpoints = _POLLUTANT_BREAKPOINTS[p]

    for (bp_low, bp_high), (idx_low, idx_high) in zip(breakpoints, _AQI_INDEX_BANDS):
        if bp_low <= standardized_value <= bp_high:
            if bp_high == bp_low:
                return float(idx_high)
            return ((idx_high - idx_low) / (bp_high - bp_low)) * (standardized_value - bp_low) + idx_low

    return 500.0


def compute_aqi_from_values(values: Mapping[str, float], units: Mapping[str, str | None] | None = None) -> float | None:
    sub_indices: list[float] = []
    for pollutant in _POLLUTANT_BREAKPOINTS:
        value = values.get(pollutant)
        if value is None:
            continue
        if value != value:  # NaN check
            continue
        unit = units.get(pollutant) if units else None
        standardized = convert_to_standard(pollutant, float(value), unit)
        sub_indices.append(compute_sub_index(pollutant, standardized))

    if not sub_indices:
        return None
    return float(max(sub_indices))
