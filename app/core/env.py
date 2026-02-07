import os


def env_int(name: str, default: int) -> int:
    """Read an integer env var with fallback on missing/invalid values."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a float env var with fallback on missing/invalid values."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
