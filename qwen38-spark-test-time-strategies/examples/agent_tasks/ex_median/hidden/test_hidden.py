import pytest

from stats import mean, median


def test_even():
    assert median([4, 1, 3, 2]) == 2.5
    assert median([10, 20]) == 15


def test_odd_unchanged():
    assert median([5, 1, 9]) == 5
    assert median([7]) == 7


def test_empty():
    with pytest.raises(ValueError):
        median([])


def test_input_not_modified():
    xs = [3, 1, 2, 4]
    median(xs)
    assert xs == [3, 1, 2, 4]


def test_mean_unchanged():
    assert mean([1, 2, 3, 4]) == 2.5
