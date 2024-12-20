import heapq
import re
from collections import Counter
from itertools import pairwise


def ints_oneline(line):
    return [int(num) for num in re.findall(r'(-?\d+)', line)]


def nats_oneline(line):
    return [int(num) for num in re.findall(r'(\d+)', line)]


def ints(data):
    if '\n' in data:
        lines = data.splitlines()
        return [ints_oneline(line) for line in lines]

    return ints_oneline(data)


def nats(data: str):
    if '\n' in data:
        lines = data.splitlines()
        return [nats_oneline(line) for line in lines]

    return nats_oneline(data)


def firstint(line):
    num = re.search(r'-?\d+', line)
    return int(num.group(0))


def toints(slist):
    """Convert a list of integers-as-string into actual integers"""
    # return [int(x) for x in slist]
    return list(map(int, slist))


def chunked(bigstring, size=None):
    if size:
        lines = bigstring.splitlines()
        return [lines[i : i + 3] for i in range(0, len(lines), size)]

    return bigstring.split('\n\n')


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))  # Recursively flatten sublists
        else:
            result.append(item)  # Add non-list items directly
    return result


def flatten_gen(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def most_common(lst, count=1):
    ctr = Counter(lst)
    common = ctr.most_common(count)
    if count == 1:
        return common[0][0]
    return [elt[0] for elt in common]


def least_common(lst, count=1):
    ctr = Counter(lst)
    reverse_ctr = sorted(ctr.items(), key=lambda x: x[1])[:count]
    if count == 1:
        return reverse_ctr[0][0]
    return [elt[0] for elt in reverse_ctr]


def nlargest(n, lst, key=None):
    return heapq.nlargest(n=n, iterable=lst, key=key)


def diffs(lst):
    return [b - a for a, b in pairwise(lst)]


def make_grid(nrows, ncols, default=None):
    return [[default for _ in range(ncols)] for _ in range(nrows)]


def get_grid(rows, sep=None):
    if sep is None:
        return [list(row) for row in rows]
    return [row.split(sep=sep) for row in rows]


def grid_cols(grid):
    return [[row[i] for row in grid] for i in range(len(grid))]


def grid_replace(grid, target, val):
    for row, _ in enumerate(grid):
        for col, _ in enumerate(grid[row]):
            if grid[row][col] == target:
                grid[row][col] = val


DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DDIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def get_neighbors(grid, pos, diag=False):
    R = len(grid)
    C = len(grid[0])
    pr, pc = pos
    nbs = []
    deltas = DIRS + DDIRS if diag else DIRS
    for dr, dc in deltas:
        if 0 <= pr + dr < R and 0 <= pc + dc < C:
            nbs.append((pr + dr, pc + dc))
    return nbs


def get_diag_coords(x1, y1, x2, y2):
    """Note - x,y coords are actually col, row"""
    if abs(x2 - x1) != abs(y2 - y1):
        print(f"Not a diagonal. ({x1},{y1}) -- ({x2},{y2})")
    if x2 < x1 and y2 < y1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    if x1 < x2 and y1 < y2:
        return [(x1 + step, y1 + step) for step in range(x2 - x1 + 1)]
    if x1 > x2 and y1 < y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    if x1 < x2 and y1 > y2:
        return [(x1 + step, y1 - step) for step in range(x2 - x1 + 1)]


def binary_search(pred, lo, hi=None):
    """Finds the first n in [lo, hi) such that pred(n) holds.

    hi == None -> infty
    """
    assert not pred(lo)

    if hi is None:
        hi = max(lo, 1)
        while not pred(hi):
            hi *= 2

    assert pred(hi)

    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


#### TESTS ####


def test_nats():
    ex1 = """3   4
          24   333"""
    res1 = nats(ex1)
    assert res1 == [[3, 4], [24, 333]]

    ex2 = "-23 and 7 - 3 + 0"
    res2 = nats(ex2)
    assert res2 == [[23, 7, 3, 0]]


def test_ints():
    ex1 = """3   4
          24   333"""
    res1 = ints(ex1)
    assert res1 == [[3, 4], [24, 333]]

    ex2 = "-23 and 7 - 3 + 0"
    res2 = ints(ex2)
    assert res2 == [[-23, 7, 3, 0]]
