"""Fit the DFPS error model from `dfpsgen` output.

Usage: python fit_dfps.py <dfpsgen.csv> <out vyuha.dfps> [gap] [min_count] [qscale]

For every position the deepest exact score stands in for the true value. For each shallower
depth d (at least `gap` plies shallower) with an exact score, the error is deep - score_d,
measured in expected-score space q = 1/(1+exp(-s/qscale)) when qscale > 0 (default 150),
else in internal score units. Errors are grouped like Model in src/dfps.h:
  depth d, instability bucket (|score_d - score_{d-2}|: <10, <30, <80, more; unknown -> 1),
  signed score_d bucket (<-800, <-300, <-100, <100, <300, <800, more), internal score units.
Per bucket the output line is "d ib vb sigma count bias": bias = mean error (shallow scores are
systematically optimistic for the side to move), sigma = spread around that mean.
Depths beyond the data get sigma from a power law in d fitted per bucket, and bias 0 (count 0).
Note: the deep score has its own error, so sigmas for d close to the deepest depth are too small;
use a gap of 4 or more. Standard library only.
"""
import csv
import math
import sys
from collections import defaultdict

EXACT = 3
DECISIVE = 30000


def ib(instab):
    if instab is None:
        return 1
    return 0 if instab < 10 else 1 if instab < 30 else 2 if instab < 80 else 3


def vb(v):
    edges = (-800, -300, -100, 100, 300, 800)
    return next((i for i, e in enumerate(edges) if v < e), len(edges))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    gap = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    min_count = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    qscale = float(sys.argv[5]) if len(sys.argv) > 5 else 150.0
    conv = (lambda s: 1 / (1 + math.exp(-s / qscale))) if qscale > 0 else (lambda s: float(s))

    rows = defaultdict(dict)  # index -> depth -> (score, bound, move)
    with open(src, newline='') as fh:
        for r in csv.DictReader(fh):
            rows[r['index']][int(r['depth'])] = (int(r['score']), int(r['bound']), r['bestmove'])

    s1 = defaultdict(float)
    s2 = defaultdict(float)
    cnt = defaultdict(int)
    flips = defaultdict(lambda: [0, 0])
    for depths in rows.values():
        exact = [d for d, (s, b, _) in depths.items() if b == EXACT]
        if not exact:
            continue
        dmax = max(exact)
        deep, _, deep_move = depths[dmax]
        if abs(deep) >= DECISIVE:
            continue
        for d, (s, b, m) in depths.items():
            if d > dmax - gap or b != EXACT or abs(s) >= DECISIVE:
                continue
            prev = depths.get(d - 2)
            instab = abs(s - prev[0]) if prev and prev[1] == EXACT else None
            key = (d, ib(instab), vb(s))
            e = conv(deep) - conv(s)
            s1[key] += e
            s2[key] += e * e
            cnt[key] += 1
            flips[d][0] += m != deep_move
            flips[d][1] += 1

    measured = {}
    for key, n in cnt.items():
        if n >= min_count:
            mean = s1[key] / n
            measured[key] = (math.sqrt(max(1e-12, s2[key] / n - mean * mean)), mean)

    # power law log(sigma) = a + k*log(d) per bucket, for the depths the data does not reach
    laws = {}
    for i in range(4):
        for v in range(7):
            pts = [(d, measured[(d, i, v)][0], cnt[(d, i, v)]) for d in range(1, 64) if (d, i, v) in measured]
            if len(pts) < 3:
                continue
            sw = sum(w for _, _, w in pts)
            mx = sum(w * math.log(d) for d, _, w in pts) / sw
            my = sum(w * math.log(s) for _, s, w in pts) / sw
            vx = sum(w * (math.log(d) - mx) ** 2 for d, _, w in pts)
            if vx <= 0:
                continue
            slope = sum(w * (math.log(d) - mx) * (math.log(s) - my) for d, s, w in pts) / vx
            laws[(i, v)] = (my - slope * mx, slope, max(d for d, _, _ in pts))

    floor = 0.002 if qscale > 0 else 4.0
    with open(dst, 'w') as out:
        print('# DFPS error model: d ib vb sigma count bias (fit_dfps.py, gap %d)' % gap, file=out)
        print('# count 0 = sigma extrapolated by a power law in depth fitted to the measured rows', file=out)
        print('space q %d' % qscale if qscale > 0 else 'space cp', file=out)
        written = extrapolated = 0
        for d in range(1, 64):
            for i in range(4):
                for v in range(7):
                    key = (d, i, v)
                    if key in measured:
                        sg, bias = measured[key]
                        print('%d %d %d %.5g %d %.5g' % (d, i, v, sg, cnt[key], bias), file=out)
                        written += 1
                    elif (i, v) in laws and d > laws[(i, v)][2]:
                        a, slope, _ = laws[(i, v)]
                        print('%d %d %d %.5g 0 0' % (d, i, v, max(floor, math.exp(a + slope * math.log(d)))), file=out)
                        extrapolated += 1
        for (i, v), (a, slope, top) in sorted(laws.items()):
            print('# bucket ib %d vb %d: sigma = %.4g * d^%.3f (measured to d %d)' % (i, v, math.exp(a), slope, top), file=out)

    print('positions %d, measured buckets %d, extrapolated %d -> %s' % (len(rows), written, extrapolated, dst))
    for d in sorted(flips):
        f, n = flips[d]
        print('depth %2d: best move differs from deepest in %5.1f%% of %d positions' % (d, 100.0 * f / n, n))


if __name__ == '__main__':
    main()
