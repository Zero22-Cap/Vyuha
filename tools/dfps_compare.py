"""Equal-node decision test: DFPS vs the alpha-beta search of the same engine, judged by Stockfish.

Usage: python dfps_compare.py <vyuha.exe> <positions.epd> [nodes] [judge_nodes] [max_positions]
                              [workers] [extra DFPS setoptions as Name=Value ...]

Each worker runs one Vyuha process and one Stockfish process, both Threads 1. For every position
(ucinewgame before every search) Vyuha picks a move with alpha-beta and with DFPS at the same node
budget. Where the two moves differ, Stockfish scores each move on its own (go nodes judge_nodes
searchmoves <move>); the higher score wins, a difference under 10 cp is a tie. Using Stockfish as
the judge avoids favouring Vyuha's own alpha-beta. Standard library only.
"""
import math
import multiprocessing as mp
import subprocess
import sys

JUDGE = 'E:/Chess/lab/src/sf-lab2/src/stockfish.exe'


class Engine:
    def __init__(self, path, opts):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send('uci')
        self.wait('uciok')
        for k, v in opts:
            self.send('setoption name %s value %s' % (k, v))
        self.ready()

    def send(self, s):
        self.p.stdin.write(s + '\n')
        self.p.stdin.flush()

    def wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('engine exited')
            if line.startswith(token):
                return line

    def ready(self):
        self.send('isready')
        self.wait('readyok')

    def quit(self):
        self.send('quit')
        self.p.wait(timeout=30)

    def go(self, fen, cmd):
        """Returns (bestmove, last score in cp or None)."""
        self.send('ucinewgame')
        self.ready()
        self.send('position fen ' + fen)
        self.send(cmd)
        score = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('engine exited')
            if line.startswith('info') and ' score ' in line:
                t = line.split(' score ')[1].split()
                score = int(t[1]) if t[0] == 'cp' else (100000 - abs(int(t[1]))) * (1 if int(t[1]) > 0 else -1)
            if line.startswith('bestmove'):
                return line.split()[1], score


W = {}


def init(exe, nodes, judge_nodes, dfps_opts):
    W['v'] = Engine(exe, [('Threads', 1), ('Hash', 256), ('SyzygyPath', '<empty>')])
    W['sf'] = Engine(JUDGE, [('Threads', 1), ('Hash', 256), ('MateSingular', 'false')])
    W['nodes'], W['jn'], W['dfps'] = nodes, judge_nodes, dfps_opts


def work(fen):
    v = W['v']
    v.send('setoption name SearchMode value AB')
    ab, _ = v.go(fen, 'go nodes %d' % W['nodes'])
    v.send('setoption name SearchMode value DFPS')
    for k, val in W['dfps']:
        v.send('setoption name %s value %s' % (k, val))
    df, _ = v.go(fen, 'go nodes %d' % W['nodes'])
    if ab == df:
        return 0
    _, sa = W['sf'].go(fen, 'go nodes %d searchmoves %s' % (W['jn'], ab))
    _, sd = W['sf'].go(fen, 'go nodes %d searchmoves %s' % (W['jn'], df))
    if sa is None or sd is None or abs(sa - sd) < 10:
        return 1  # differ, tie
    return 2 if sd > sa else 3  # 2: DFPS better, 3: AB better


def main():
    exe, epd = sys.argv[1], sys.argv[2]
    nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    judge_nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 5000000
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    dfps_opts = [tuple(a.split('=', 1)) for a in sys.argv[7:]]

    fens = []
    with open(epd) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 4:
                fens.append(' '.join(f[:4]) + ' 0 1')
    fens = fens[:limit]

    # an engine silently ignores an option it does not know: check every option first
    chk = Engine(exe, [])
    for k, val in dfps_opts:
        chk.send('setoption name %s value %s' % (k, val))
    chk.send('isready')
    while True:
        line = chk.p.stdout.readline()
        if line.startswith('readyok'):
            break
        if 'unknown option' in line:
            raise SystemExit('engine rejected: ' + line.strip())
    chk.quit()

    c = [0, 0, 0, 0]
    with mp.Pool(workers, initializer=init, initargs=(exe, nodes, judge_nodes, dfps_opts)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, fens, chunksize=4), 1):
            c[r] += 1
            if i % 100 == 0 or i == len(fens):
                n = c[2] + c[3]
                z = (c[2] - c[3]) / math.sqrt(n) if n else 0.0
                print('%d positions: same move %d, differ-tie %d, DFPS better %d, AB better %d, z %+.2f'
                      % (i, c[0], c[1], c[2], c[3], z), flush=True)


if __name__ == '__main__':
    main()
