"""Sample positions from PGN files into an EPD file (one FEN per line).

Usage: python pgn_positions.py <out.epd> <every> <min_ply> <file.pgn> [file.pgn ...]
Takes every `every`-th position from ply `min_ply` on, skipping positions where the side to move
has no legal move. Needs python-chess (PYTHONPATH=C:/scdata/pylib).
"""
import sys

import chess.pgn


def main():
    out, every, min_ply = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seen = set()
    n = 0
    with open(out, 'w') as fo:
        for path in sys.argv[4:]:
            with open(path, encoding='utf-8', errors='replace') as fh:
                while True:
                    g = chess.pgn.read_game(fh)
                    if g is None:
                        break
                    b = g.board()
                    for ply, mv in enumerate(g.mainline_moves(), 1):
                        b.push(mv)
                        if ply < min_ply or (ply - min_ply) % every or b.is_game_over():
                            continue
                        key = b.epd()
                        if key in seen:
                            continue
                        seen.add(key)
                        fo.write(key + '\n')
                        n += 1
    print('%d positions -> %s' % (n, out))


if __name__ == '__main__':
    main()
