#!/usr/bin/env python3
from calc_markov import *


def test_time_to_index():
    assert time_to_index("06.00") == 0
    assert time_to_index("06.30") == 1
    assert time_to_index("07.25") == 2
    assert time_to_index("07.35") == 3



if __name__ == "__main__":
    test_time_to_index()
