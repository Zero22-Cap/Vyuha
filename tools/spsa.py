"""SPSA tuning of Vyuha's search constants (the TUNE_LIST table in src/engine.cpp).

Usage: python spsa.py <engine.exe> <state.json> [pairs_total] [--params A,B,...]
Each batch plays game pairs (same opening, colours swapped) between theta+c*delta and theta-c*delta
with fastchess, then moves theta along delta by the result (fishtest's schedule: c_k = c/k^0.101,
a_k = a/(A+k)^0.602, A = N/10, c_end = 10% of the default, r_end = 0.002). The state file holds theta
and the pair count, so a stopped run resumes where it ended. Control files next to the state file:
<state>.PAUSE waits, <state>.STOP ends after the current batch. Standard library only.
"""
import json
import os
import random
import re
import subprocess
import sys
import time

FC = 'E:/Chess/tools/fastchess/fastchess-windows-x86-64/fastchess.exe'
BOOK = 'E:/Chess/lab/book/balanced-openings.epd'
TC = os.environ.get('SPSA_TC', '4+0.04')
CONC = int(os.environ.get('SPSA_CONC', '12'))
BATCH = int(os.environ.get('SPSA_BATCH', '12'))  # game pairs per batch
# small integer ranges tune badly with SPSA's rounding: left out unless named
SKIP = {'RfpDepth', 'NmpBase', 'LmpBase', 'CorrMinorW', 'CorrContW', 'TtPvLmr', 'SmpVote', 'RootLmrSub', 'PvLmrSub', 'RootVerify'}  # the last seven switch features (0 = off)


def tunelist(exe):
    p = subprocess.run([exe], input='tunelist\nquit\n', capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        f = line.split()
        if len(f) == 4 and f[0] != 'tunelist':
            out[f[0]] = dict(default=int(f[1]), lo=int(f[2]), hi=int(f[3]))
    return out


def play(exe, plus, minus, pairs, tag):
    def eng(name, vals):
        return ['-engine', 'cmd=' + exe, 'name=' + name] + ['option.%s=%d' % (k, v) for k, v in vals.items()]
    cmd = [FC] + eng('plus', plus) + eng('minus', minus) + [
        '-each', 'tc=' + TC, 'option.Threads=1', 'option.Hash=16', 'option.SyzygyPath=<empty>',
        '-openings', 'file=' + BOOK, 'format=epd', 'order=random', '-srand', str(random.randrange(1 << 30)),
        '-games', '2', '-rounds', str(pairs), '-repeat', '-concurrency', str(CONC), '-recover']
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.findall(r'Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)', out)
    if not m:
        raise RuntimeError('no result from fastchess (%s):\n%s' % (tag, out[-2000:]))
    g, w, l, d = map(int, m[-1])
    return g, w, l, d


def main():
    exe, state_path = sys.argv[1], sys.argv[2]
    total = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 20000
    names = None
    if '--params' in sys.argv:
        names = sys.argv[sys.argv.index('--params') + 1].split(',')
    table = tunelist(exe)
    if not table:
        sys.exit('engine printed no tunelist')
    if os.path.exists(state_path):
        st = json.load(open(state_path))
    else:
        use = names or [n for n in table if n not in SKIP]
        st = {'pairs': 0, 'total': total, 'theta': {n: float(table[n]['default']) for n in use}, 'log': []}
    N = st['total']
    A, alpha, gamma, r_end = N / 10, 0.602, 0.101, 0.002
    cend = {n: max(1.0, 0.1 * abs(table[n]['default'])) for n in st['theta']}
    while st['pairs'] < N:
        if os.path.exists(state_path + '.STOP'):
            print('STOP file: ending', flush=True)
            break
        while os.path.exists(state_path + '.PAUSE'):
            time.sleep(30)
        k = st['pairs'] + 1
        plus, minus, delta, ck = {}, {}, {}, {}
        for n, v in st['theta'].items():
            lo, hi = table[n]['lo'], table[n]['hi']
            c = cend[n] * N ** gamma / k ** gamma
            d = random.choice((-1, 1))
            delta[n], ck[n] = d, c
            plus[n] = int(round(min(hi, max(lo, v + c * d))))
            minus[n] = int(round(min(hi, max(lo, v - c * d))))
        t0 = time.time()
        g, w, l, dr = play(exe, plus, minus, BATCH, 'pairs %d' % k)
        res = w - l
        for n in st['theta']:
            lo, hi = table[n]['lo'], table[n]['hi']
            a_end = r_end * cend[n] ** 2
            a = a_end * (A + N) ** alpha
            # a batch of B pairs counts as B iterations with the same delta
            ak = a / (A + k) ** alpha
            R = ak / ck[n] ** 2
            st['theta'][n] = min(hi, max(lo, st['theta'][n] + R * ck[n] * res * delta[n]))
        st['pairs'] += g // 2
        st['log'].append([st['pairs'], w, l, dr, round(time.time() - t0, 1)])
        tmp = state_path + '.tmp'
        json.dump(st, open(tmp, 'w'), indent=1)
        os.replace(tmp, state_path)
        if len(st['log']) % 10 == 0:
            print('pairs %d  %s' % (st['pairs'], ' '.join('%s=%.1f' % (n, v) for n, v in st['theta'].items())), flush=True)
    print('final', json.dumps({n: round(v) for n, v in st['theta'].items()}), flush=True)


if __name__ == '__main__':
    main()
