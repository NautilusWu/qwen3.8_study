"""Small statistics helpers."""


def mean(values):
    """Arithmetic mean of a non-empty list of numbers."""
    if not values:
        raise ValueError("mean of an empty list")
    return sum(values) / len(values)


def median(values):
    """Median of a non-empty list of numbers."""
    values = sorted(values)
    return values[len(values) // 2]
