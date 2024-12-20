'Advent of Code {{ year }} Day {{ day }}'

import math
from collections import Counter, defaultdict, deque
from functools import lru_cache
from itertools import combinations

import regex as re
from aocd import data
from bidict import bidict
from loguru import logger

from waoc.utils import chunked, diffs, firstint, ints, nats

TEST_DATA = """"""
TEST_ANS_A = None
TEST_ANS_B = None


def solve(test=False, parta=True):
    dat = TEST_DATA if test else data

    lines = dat.splitlines()
    print(lines)

    res = None

    if test:
        test_ans = TEST_ANS_A if parta else TEST_ANS_B
        print(f"********** RESULT: {res == test_ans} **********")

    return res
