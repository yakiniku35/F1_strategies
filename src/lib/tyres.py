"""
Tyre compound utilities for F1 race replay.
"""

TYRE_COMPOUNDS = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}


def get_tyre_compound_int(compound_str):
    """Convert tyre compound string to integer."""
    if compound_str is None:
        return -1
    return TYRE_COMPOUNDS.get(str(compound_str).upper(), -1)


def get_tyre_compound_str(compound_int):
    """Convert tyre compound integer back to string."""
    for name, value in TYRE_COMPOUNDS.items():
        if value == compound_int:
            return name
    return "UNKNOWN"
