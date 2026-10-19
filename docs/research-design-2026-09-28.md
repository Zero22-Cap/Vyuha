# Vyuha — research survey and design to close the gap (2026-09-28)

## 1. Where the field is (Sept 2026)

| Source | What matters for us |
|---|---|
| Stockfish 19 (Sept 2026) | +44 Elo vs SF 18. SFNNv16: threat inputs pruned + pawn-pair inputs, secondary net retired, **quantization-aware training**, trained on **hundreds of billions** of Leela-rescored positions. Fishtest: 20,600 CPU-years, 10.2 B games |
| CCRL (Sept 2026) | Reckless 3822 (#2), PlentyChess 3809, Viridithas 3792, Obsidian 3766 — all alpha-beta + NNUE, SF-style search, SPSA-tuned |
| TCEC S29 superfinal | SF beat Reckless 59–41. Lc0 no longer the main challenger |
| Coda (2026) | Tested via OpenBench + Bullet. ~3500–3600 CCRL-equivalent after ~6 months. Threat-aware NNUE, 16 king buckets, 1024 pairwise FT, two int8 hidden layers, multi-source correction history |
| Monty (CPU MCTS) | CEDR ~3431; big recent gains but far below AB engines |
| Lc0 / Chessformer (ICLR 2026) | BT4 = "Leela-CF"; BT5 in development (+5 % inference speed). Searchless research improves puzzles, not play strength |
| AlphaEvolve (DeepMind, 2025) | LLM + automated evaluator + evolution beat 56-year-old matrix-multiply bound and improved Google production heuristics. The method transfers to engine heuristics, but **the evaluator for a chess patch is tens of thousands of games** |
| GPU alpha-beta | Papers report 11x on chess *without pruning*; with pruning the gain disappears. Claimed "3850 GPU engines" (NikolaChess) are unverified marketing. Our own measurement: GPU 3,600x too slow per node |
| KataGo uncertainty / our policy head | Uncertainty: −1 ± 17 here. Frozen policy head: −15 / −7. Not worth more time without joint training |

## 2. The honest constraint

The gap to Stockfish (−363) is ~85 % evaluation and ~15 % search tuning (depth/time per move already equal).
Stockfish's edge is **scale**: hundreds of billions of training positions and ~20,000 CPU-years of game
testing. One PC produces ~1,700 games/hour at 10+0.1, so one SPRT for a +3 Elo patch (~40k games) takes
~24 h. Ideas are cheap; **verification is the bottleneck**. No from-scratch engine has passed Stockfish;
Reckless, with a strong team and distributed testing, is still ~40 Elo behind.

## 3. Design (in order of Elo per hour)

### Phase A — Evaluation (GPU-bound, largest gain)
1. Data: 12–20 months of T80 binpacks (~20–40 B positions). Stream binpack -> GPU directly (no 72-byte
   files; the current pipeline would need >1 TB).
2. Trainer speed: 0.6 M -> 3–5 M positions/s (bf16, sparse FT gradient, fused feature kernel, or Bullet).
3. Architecture: 16 king buckets + **threat inputs** + pawn-pair inputs, FT 1024–1536 with pairwise
   multiplication, int8 hidden layers 16 -> 32 -> 1 per material bucket (layer stacks), QAT.
Expected: +150–250 Elo over today.

### Phase B — Inference speed (CPU)
Finny tables (accumulator cache per king bucket), int8 L1 with AVX-512 VNNI (Zen 4 has it), sparse L1
(skip zero activations), large pages for TT, NUMA-free thread pinning. Expected +25–40 % nps ≈ +20–35 Elo.

### Phase C — Search (proven elsewhere, port and verify)
Multi-source correction history (pawn, minor, non-pawn, continuation), pawn history, continuation history
at plies 1/2/4/6, ttPv flag, cutoff counts, fail-high score blending, hindsight depth adjustment,
SF-style time management, then **SPSA** on ~40 constants (overnight). Expected +80–150 Elo.

### Phase D — automated research loop (our own mini-fishtest, AlphaEvolve-style)
Candidate patches come from the published ideas of SF/Reckless/Viridithas and own variants;
fastchess SPRT at STC (14 concurrent games) decides; winners are merged; SPSA re-tunes weekly.
Throughput: ~1–2 verdicts per night. This is how Coda was built.

### Phase E — Your use cases (where "better than Stockfish" is actually reachable)
Syzygy (Fathom), MultiPV, experience file / persistent hash, a mate mode with a df-pn helper thread
(Huntsman already beats SF on mates <= M11 in matebench).

## 4. Realistic outcome

| Stage | Est. Elo (GUI scale, SF-Analysis = 3660) | Time |
|---|---|---|
| Today | ~3300 | done |
| A (data + trainer + deeper net) | ~3450–3500 | 1–2 weeks, mostly GPU time |
| + B + C + SPSA | ~3550–3600 | weeks–months of nightly testing |
| Beat Stockfish at equal hardware | not realistic on one PC | needs SF-scale data and testing |
| Beat Stockfish in a niche (short/medium mates, persistent analysis) | plausible | weeks |
