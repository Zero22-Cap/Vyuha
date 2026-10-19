# Vyuha

A UCI chess engine written from scratch in C++ as a personal research project. Version 0.4-dev.
The name is Sanskrit for a battle formation. The long-term goal is to solve chess without
endgame tablebases.

Vyuha has two sides: a conventional alpha-beta engine with an NNUE evaluation (the default), and
research on a search of its own, DFPS (Decision-Focused Probe Search), together with a net that
learns chess from zero through self-play.

## Strength (estimates, October 2026)

| Configuration | Estimate | Basis |
|---|---|---|
| Alpha-beta + main net | about 3300 | GUI rating |
| Self-learned net (zero knowledge) | about 2920 | -83 +/- 43 Elo vs Stockfish at UCI_Elo 3000, 200 games, 5+0.05 |
| DFPS vs alpha-beta, 600k nodes per move | -21 +/- 23 Elo | 200 games |
| DFPS vs alpha-beta, 10+0.1 | -55 +/- 20 Elo | 300 games |
| DFPS vs alpha-beta, 5k nodes per move | -123 +/- 23 Elo | 600 games |

## Alpha-beta search

- Principal variation search with iterative deepening and aspiration windows; quiescence search
  with SEE; mate-distance pruning; repetition and 50-move rules.
- Transposition table with 3-entry clusters, aging by search generation and a was-on-PV flag.
- Reverse futility pruning, razoring, null move, ProbCut, late-move pruning, futility, history and
  SEE pruning, late-move reductions adjusted by history.
- Singular extensions with double and negative extensions.
- Move ordering: hash move, captures by value and capture history, killers, counter moves, main
  history and continuation history (one and two plies back).
- Correction history by pawn structure and non-pawn material (minor-piece and continuation
  correction as options).
- Lazy SMP; Syzygy tablebases in search (WDL) and at the root (DTZ ranking) through Fathom.
- Time management from best-move stability, the best move's share of the nodes and falling scores.

## Evaluation

NNUE: 10 king buckets x 768 piece-square inputs (horizontally mirrored) -> 1024 x 2 -> SCReLU ->
8 output buckets by piece count, AVX-512 inference, a per-king-bucket refresh cache. Optional
uncertainty and policy heads exist (off by default; they did not gain strength). The main net
(`vyuha.nnue`) was trained with `train/train.py` on training data published by the Stockfish
project (Leela Chess Zero games rescored by Stockfish).

## DFPS - Decision-Focused Probe Search

`SearchMode DFPS` (source `src/dfps.h`). Alpha-beta spends its effort by depth; DFPS spends it
where it is most likely to change the move that is played.

- A small best-first tree is kept at the root. Its leaves are alpha-beta probes, so tactics a few
  plies deep are still seen. The root moves start from the root search's own per-move results.
- Every root move holds a belief (mean, sigma) in expected-score units; sigma comes from an error
  model fitted on Vyuha's own searches (`dfpsgen`, `tools/fit_dfps.py`, file `vyuha.dfps`).
- Beliefs are combined with Clark's approximation for the maximum of Gaussians. A knowledge-
  gradient rule chooses the next probe: deepen the best move, or test a challenger with a null
  window at the best move's depth (reduced by its rank, re-tested at full depth if it fails high).
- Results proven by the rules (mate, stalemate, draws) propagate as exact intervals; once they
  decide the root, the search stops.
- `dfpscal` measures how one-sided (null-window) results relate to full values. On 200 positions
  at depth 10, a move alpha-beta had refuted at the root scored more than 30 above the chosen move
  in 60 positions after a full re-search, and Stockfish preferred it 26 times to 15; most of those
  moves had been reduced 2-4 plies. Reducing root moves less everywhere cost Elo (-10 +/- 9), so the
  fix has to be selective. Root-level signals did not predict which positions hide such a move
  (AUC about 0.5).

Design notes, measurements and lessons: `docs/research-architecture-2026-10-16.md`.

### Related work

DFPS belongs to the family of selective, decision-theoretic game-tree searches. Closest to it:
E. B. Baum and W. D. Smith, "A Bayesian approach to relevance in game playing", Artificial
Intelligence 97 (1997); S. Russell and E. Wefald, "Do the Right Thing" (1991, value of
computation); H. Berliner, "The B* tree search algorithm" (1979); D. McAllester, "Conspiracy
numbers for min-max search" (1988); P. Frazier, W. Powell and S. Dayanik on the knowledge
gradient (2008); C. E. Clark, "The greatest of a finite set of random variables" (1961). What is
new here is the combination: alpha-beta probes as leaves, an error model fitted on the engine's
own searches, Clark backup with knowledge-gradient selection, null-window challenger tests at the
best move's depth, and proven intervals.

## Self-learning from zero

`selfplay <out.bin> <games> <softNodes> <maxDepth> [randomPlies] [nonet]` plays whole games with
a soft node limit per move and writes training records; `nonet` plays with material only, which
is how generation 0 starts from zero knowledge. Positions with a proven mate are kept as sure
wins or losses. `tools/selflearn2.sh` runs the loop: self-play with the best net (12 million
positions per generation), training on the last eight generations mixed (`tools/mix_records.py`,
ten passes), a match against the best net, and an anchor match against Stockfish at a fixed Elo.
The next generation's self-play runs while the GPU trains.

## Testing and tuning

- `tools/dfps_match.sh`, `tools/dfps_compare.py`: DFPS against alpha-beta in games and in
  equal-node decision tests judged by Stockfish.
- `tools/spsa.py`: SPSA tuning of the search constants (exposed as hidden UCI options; the
  `tunelist` command prints them).
- `tools/sprt_queue.sh`: a queue of SPRT matches for option changes.

## Build

Needs GCC with C++20 and a CPU with AVX-512 and BMI2 (developed on a Ryzen 7 7700X).

```bash
gcc -O3 -march=native -std=gnu11 -w -Iext/fathom -c ext/fathom/tbprobe.c -o ext/fathom/tbprobe.o
```

```bash
g++ -O3 -march=native -std=c++20 -static -pthread -Iext/fathom -o vyuha.exe src/engine.cpp ext/fathom/tbprobe.o
```

Put `vyuha.nnue` (and optionally `vyuha.dfps`) next to the program, or set `EvalFile` /
`DfpsModel`. `vyuha.exe bench` prints a node count (311108 for this version with the included net).

## UCI options

`Threads`, `Hash`, `EvalFile`, `SyzygyPath`, `SyzygyProbeLimit`, `SyzygyProbeDepth`,
`Syzygy50MoveRule`, `Move Overhead`, `SearchMode` (AB or DFPS), the `Dfps*` options of the DFPS
search, `NoNet` (material only), and the experimental `UncMode`, `PolicyMode`, `PolicyWeight`,
`PolicyDepth`, `PolicyFile`.

## Layout

| Path | Contents |
|---|---|
| `src/` | the engine: `engine.cpp`, `chess.h`, `nnue.h`, `dfps.h`, `selfplay.h` |
| `train/` | PyTorch trainers for the value net and the policy head |
| `tools/` | training-data tools, self-learning driver, test and tuning scripts |
| `ext/` | third-party code (see below) |
| `docs/` | research notes and design |
| `vyuha.nnue`, `vyuha.pol`, `vyuha.dfps` | main net, policy head, DFPS error model |

## Credits and third-party material

- **Fathom** (`ext/fathom`): Syzygy tablebase probing, MIT licence; see `ext/fathom/LICENSE`.
- **Training-data reader** (`ext/*.h`): from the nnue-pytorch project (GPL-3.0), used only by
  `tools/convert.cpp`.
- **Training data** for the main net: published by the Stockfish project, derived from Leela
  Chess Zero self-play games rescored by Stockfish. No Stockfish or Lc0 network is included.
- The alpha-beta search techniques are the well-known ones described on the Chess Programming
  Wiki and used by open-source engines such as Stockfish.

## Licence

GPL-3.0; see `LICENSE`.
