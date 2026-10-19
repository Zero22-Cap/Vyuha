# Vyuha — research survey for a new engine architecture (2026-10-16)

Question: which mathematical and algorithmic work could redefine how a chess engine works, with
the far goal of solving chess without tablebases and beating every other engine? When this survey
was written nothing below had been built or tested yet (sections 5a and 5b were added later). Every claim is from the cited source; where a source
is unreviewed (arXiv only, GitHub, forum) it says so.

## 1. Verdict in five lines

1. No paper found offers a search algorithm that would leap past Stockfish-class engines. The
   theory says alpha-beta of the Scout/PVS family is already the right class for chess.
2. The real advances of 2019–2026 are in **learning efficiency** (KataGo, Gumbel, search-contempt),
   **network design** (Chessformer) and **evidence that superhuman nets have systematic blind
   spots** (adversarial attacks on KataGo, adversarial endgame positions for Stockfish).
3. At the top, games are mostly draws (Stockfish 19: 92.4 % draws on CCRL 40/15). A stronger engine
   only wins when the opponent makes an error, so "beating everyone" means finding and exploiting
   errors, not only playing better moves.
4. Solving chess needs exact values plus a proof tree of roughly 4×10^22 positions per side. No
   method shrinks that below the theoretical minimum; better move ordering only approaches it.
5. The architecture in section 5 combines pieces that exist separately but have not been assembled
   in one chess engine: an exact (proof-carrying) search core, self-learned knowledge from zero,
   a GPU teacher used only at training time, and an adversarial loop that hunts blind spots.

## 2. Hard limits (mathematics that no architecture escapes)

| Fact | Source | Consequence |
|---|---|---|
| (4.82 ± 0.03)×10^44 legal positions | Tromp, ChessPositionRanking (GitHub) | Strong solution (every position) is out of reach |
| Minimal proof tree ≈ 2·√N ≈ 4.4×10^22 positions; a draw needs two proofs | Estimate by the Syzygy tablebase author, TalkChess, Jan 2025 (forum) | At 22 M positions/s: tens of millions of years on this PC |
| Generalised n×n chess is EXPTIME-complete | Fraenkel & Lichtenstein, J. Comb. Theory A, 1981 | No general polynomial shortcut exists; only structure specific to 8×8 can help |
| Alpha-beta minimal tree b^⌈d/2⌉ + b^⌊d/2⌋ − 1 with perfect ordering | Knuth & Moore, 1975 | Search effort is governed by move ordering, i.e. by knowledge |
| Alpha-beta's branching factor is asymptotically optimal on random trees | Pearl, CACM 1982 | The search *class* is not where gains hide |
| On synthetic trees with ancestor dependence, AlphaBeta has a much larger constant factor than Scout; same asymptotic branching factor | Boige, Boumaza, Scherrer, NeurIPS 2025 | Supports the Scout-family (PVS) that engines already use |
| Checkers (≈5×10^20 positions) was solved with 10-piece databases (3.9×10^13 positions, 237 GB) and a forward proof manager over ~18 years | Schaeffer et al., Science 2007 | Template for a solve: exact knowledge from the end + proof search from the start |
| Top-engine draw rate: Stockfish 19 92.4 % draws, CCRL 40/15, Sept 2026; top three within 12 Elo | CCRL | Wins come from opponent errors; balanced-opening matches barely separate engines |
| TCEC S29 Superfinal: Stockfish 59–41 Reckless (unbalanced openings) | Chessdom, Apr 2026 | Elite events force decisive games through sharp openings |

## 3. Survey by area

### 3.1 Search algorithms

| Work | Year | What it shows | Use for Vyuha |
|---|---|---|---|
| Alpha-beta / PVS (Knuth & Moore; Pearl) | 1975–82 | Near-optimal node count given good ordering | **Keep** as the core |
| Proof-number search; df-pn; survey "The first twenty years" (Kishimoto, Winands, Müller, Saito) | 1994–2012 | Dominant method for *solving* games; needs care on graphs | **Adopt** for the exact layer |
| Graph-history interaction, general solution (Kishimoto & Müller, AAAI) | 2004 | Transpositions + repetition rules make naive proofs wrong | **Required** for any proof store |
| UCT / MCTS; PUCT (AlphaZero) | 2006 / 2017 | Selective best-first search | Root only (see 3.1 shallow traps) |
| Shallow traps (Ramanujan, Sabharwal, Selman, ICAPS) | 2010 | Traps of depth 3–7 are frequent in chess, even in GM games; UCT cannot identify them, minimax can | Explains why pure MCTS is wrong for chess tactics |
| MCTS–minimax hybrids (Baier & Winands) | 2013–14 | Shallow minimax inside MCTS fixes part of the trap problem | Supports a hybrid, alpha-beta below |
| Monte-Carlo Graph Search (Czech, Korus, Kersting, ICAPS) | 2021 | Tree → DAG: information flows across transpositions, much less memory | Adopt if a best-first layer is built |
| Gumbel AlphaZero / MuZero (Danihelka et al., ICLR) | 2022 | Guaranteed policy improvement even with 2 simulations; sequential halving at root | **Adopt** for self-play and root allocation |
| RMCTS, optimized posterior policies (Frankston & Howard, arXiv) | 2026 | Breadth-first batched MCTS, >40× faster on one root (Connect-4, Othello, Dots-and-Boxes; not chess) | Training-time speed-up candidate |
| Rational metareasoning, MGSS* (Russell & Wefald); "Selecting computations" (Hay, Russell, Tolpin, Shimony, UAI) | 1989–2012 | Choose each search step by its value of information | Theory for a root controller; never competitive in chess yet |
| Bayesian game-tree search (Baum & Smith) | 1997 | Propagate value distributions, not scalars | Our uncertainty head was −1 ± 17; low priority |
| Minimax pathology (Nau 1982); real-valued minimax (Luštrek, Gams, Bratko 2006); "When is it better not to look ahead?" (Nau et al. 2010) | 1982–2010 | Deeper search can be worse on independent trees; dependences in real games remove most of it | Confirms deep search is sound for chess |

### 3.2 Learning

| Work | Year | What it shows | Use for Vyuha |
|---|---|---|---|
| AlphaZero (Silver et al.) | 2017–18 | Tabula-rasa self-play; used 5,000 TPUs for game generation | Concept only; scale impossible here |
| Expert Iteration (Anthony, Tian, Barber) | 2017 | Search = expert, network = apprentice | The loop Vyuha would run |
| KataGo, "Accelerating self-play learning" (Wu) | 2019 | 50× less compute: playout-cap randomisation, policy-target pruning, global pooling, auxiliary targets; beat ELF in 19 days on < 30 GPUs | **Adopt** its general techniques |
| Search-contempt (Joshi, arXiv, single author, unreviewed) | 2025 | Hybrid MCTS that prefers harder positions; up to 70 Elo stronger self-play games; claims training from zero on a consumer GPU becomes feasible | Promising, must be verified here |
| Scaling scaling laws with board games (Jones) | 2021 | 10× training compute replaces ~15× test-time compute (Hex); 9×9 Hex solved to perfect play in < 3 h on one RTX 2080 Ti | **Key**: spend GPU at training time, not in the tree |
| AlphaZero scaling and Zipf's law (Neumann & Gros, NeurIPS 2025) | 2024–25 | Nets learn states in order of frequency; in games where endgame states dominate, bigger nets can get *worse* | Warns about endgame-heavy training mixes |
| Chessformer (Monroe et al., ICLR 2026) | 2026 | Squares as tokens, Geometric Attention Bias, attention policy head; +100 Elo inside Lc0 | Teacher-net design |
| Grandmaster-level chess without search (Ruoss et al., NeurIPS 2024) | 2024 | 270 M transformer, Lichess blitz 2895, "perfect distillation still beyond reach" | Searchless is not the path to beating engines |
| DiffuSearch, implicit search via discrete diffusion (Ye et al., ICLR 2025) | 2025 | Beats one-step and MCTS-enhanced policies of its own scale; +540 Elo over one-step policy | Research curiosity; far below engine level |
| External and internal planning with language models (DeepMind) | 2024 | One model does world model + values; search distilled into the model | Not competitive with engines on CPU budgets |
| Learned look-ahead in Lc0 (Jenner et al., NeurIPS 2024); "The algorithm is not the behavior" (arXiv 2025) | 2024–25 | Lc0's net internally looks 2 moves ahead, but learned priors often override it | Nets compress search; they do not replace it |
| M2CTS, mixture of experts by game phase (Helfenstein, Czech et al.) | 2024–25 | Up to +122 Elo over a single-net baseline in their setting | Option for Vyuha's net later |

### 3.3 Exactness and solving

| Work | What it shows | Use for Vyuha |
|---|---|---|
| Checkers solved (Schaeffer et al. 2007) | Backward databases + forward proof manager with transposition-safe storage | The structure to copy, without precomputed tablebases |
| MCTS-Solver (Winands et al. 2008); score-bounded MCTS (Cazenave & Saffidine 2010) | Proven wins/losses/draws propagate inside a selective search | Proof tags in Vyuha's search |
| Combinatorial game theory in chess endgames (Elkies 1996; "Higher nimbers", 2002) | Some endgames split into independent subgames whose values add (mutual zugzwang, tempo counting) | Long-term solve lever: exact reasoning by decomposition, used by no engine |
| Logic-minimisation compression of tablebases (Gomboc & Shelton, ACG 2021) | Exact knowledge can be stored far more compactly | Compact proof store |
| Neural tablebase compression (mcarbonell, GitHub, unreviewed) | 99.93 % accuracy on 3-piece endings, i.e. not exact even there | Nets alone are never exact; proofs must back them |
| Stockfish or Lc0 against tablebases (Sadmine et al., ACG 2023) | Both still make endgame errors | Exam for Vyuha (Syzygy as grader only) |

### 3.4 Robustness and exploitation

| Work | What it shows | Use for Vyuha |
|---|---|---|
| Adversarial policies beat superhuman Go AIs (Wang et al., ICML 2023) | > 97 % wins against superhuman KataGo with < 14 % of its training compute; the adversary loses to amateurs; a human learned the trick | Superhuman nets have *systematic* holes |
| Can Go AIs be adversarially robust? (Tseng et al., AAAI 2025) | Three defences; none withstand freshly trained adversaries | Robustness is an open problem, i.e. an opening |
| Discovery of adversarial endgame chess positions (AdvChess, ICLR 2026 submission) | Positions where Stockfish deviates from perfect play are densely clustered; they transfer poorly across settings | Engines' errors are findable by local search |

### 3.5 Exotic directions

| Work | Status |
|---|---|
| Quantum NAND-tree evaluation in √N time (Farhi, Goldstone, Gutmann 2007) | Real mathematics, no usable hardware; skip |
| AlphaEvolve (DeepMind 2025): LLM + automatic evaluator + evolution | Needs a fast evaluator; one chess patch needs tens of thousands of games; skip for now |
| Human-like engines (Maia-1/2/3, Allie, Matilda 2026) | Different goal (human imitation); skip |

## 4. What earlier experiments already ruled out (do not redo)

| Idea | Measured result |
|---|---|
| GPU net inside the alpha-beta tree | ~3,600× too slow per node |
| Lc0 + Stockfish arbiter at the root | Who-is-right signal worth 0.34 cp/move |
| GPU root allocator | Stockfish's own fragility signals predict better (AUC 0.751 vs 0.679) |
| Instability / conspiracy numbers driving pruning | AUC ≈ 0.50; fishtest rejected 14 node-level conspiracy-number tests in 2022 |
| Uncertainty head scaling pruning | −1 ± 17 Elo |
| Frozen-accumulator policy head for ordering | −15 ± 18, −7 ± 18 Elo |
| Retro solver vs search for long mates | 3/25 vs 18/25 exact |

## 5. Proposed Vyuha architecture (for discussion, not yet built)

Four layers. Each is general (not a feature mode) and each has its own measurement.

| Layer | What it is | Grounded in | How it is measured |
|---|---|---|---|
| 1. Exact core | PVS alpha-beta whose every score carries a tag: proven bound or estimate. Proven values propagate, live in a transposition-safe table (GHI-correct) and in a permanent proof store. | Knuth–Moore, PNS survey, Kishimoto–Müller, MCTS-Solver, checkers proof | Bench unchanged; no Elo loss; count of proven positions; zero wrong proofs |
| 2. Self-learned knowledge | Start from zero: no Stockfish data, no tablebases. Expert-iteration self-play with KataGo's and Gumbel's efficiency techniques. Labels: proven value where one exists, otherwise deep-search value. | AlphaZero, ExIt, KataGo, Gumbel, search-contempt | Generation-vs-generation Elo; Syzygy exam (grader only); later vs SF-Analysis and Lc0 |
| 3. Teacher at training time | A larger transformer (Chessformer-style) trained on the GPU from the same self-play, used only to label positions and to distil into the fast CPU NNUE. The GPU never enters the search tree. | Jones 2021 trade-off; Chessformer; our GPU-in-tree measurement | NNUE loss and Elo with vs without teacher labels at equal GPU hours |
| 4. Adversarial loop | A search that hunts positions where Vyuha's evaluation is wrong (vs proof or much deeper search), feeding them back into training; the same tool maps other engines' error clusters for match preparation. | Wang et al. 2023; Tseng et al. 2025; AdvChess 2026 | Error rate on found positions falls across generations; score vs opponents from their error clusters |

Long-term (after the four layers work): exact subgame decomposition for endgames (Elkies) and a
best-first proof manager over the proof store, as in the checkers proof.

What this design deliberately does not do: pure MCTS (shallow traps), GPU in the tree (measured),
searchless or diffusion play (too weak), quantum methods (no hardware).

## 5a. DFPS — Decision-Focused Probe Search (own design, approved 2026-10-16)

DFPS was chosen as the search core in place of plain alpha-beta. Code: `src/dfps.h`, selected
with the UCI option `SearchMode DFPS` (default stays `AB`).

Design:
- A best-first tree near the root; its leaves are alpha-beta probes of a recorded depth (traps are
  still seen inside the probes). Probes run as jobs on the worker threads, sharing the TT.
- Each leaf holds a belief (mean, sigma) in expected-score space q = 1/(1+e^(-s/150)); sigma comes
  from an error model fitted on Vyuha's own data (`dfpsgen` + `tools/fit_dfps.py`, 6,000 positions
  to depth 20, tablebases off).
- Clark's approximation backs up the maximum of Gaussians; a knowledge-gradient controller picks
  what to refine next, weighed by expected nodes.
- Exact intervals (mate, stalemate, draws by rule) propagate; the search stops once they decide
  the root.

What the measurements changed (equal nodes, 200k, Threads 1, 400 positions from our own games,
Stockfish sf-lab2 at 5M nodes judging only the positions where the two moves differ):

| Version | DFPS better | AB better | z |
|---|---|---|---|
| v1 (wide-window child probes) | 13 | 88 (of 200) | -7.5 |
| fixed aspiration bug, wide sigma for bounds, root-only tree | 43 | 94 | -4.4 |
| + depth-bias correction | 39 | 118 | -6.3 |
| + null-window refinement one step deeper | 41 | 140 | -7.4 |
| **only the best move is deepened; others tested at the best move's depth** | **40** | **56** | **-1.6** |
| same, full tree (expansion below the root) | 31 | 108 | -6.5 |

In games (1,000 at 10+0.1, Threads 1, tablebases off) the parity version (root-only, 1-ply steps)
scored **-85 +/- 11 Elo** against alpha-beta: equal decisions per node are not equal strength per
second. Its node rate and time use are the next things to measure.

Lessons, each measured:
1. Expanding a leaf into all its children costs about 5x more than deepening it, and loses the
   cutoffs alpha-beta gets; below the root the tree does not pay (yet).
2. Values from different depths must not be compared: shallow scores are too extreme (at depth 5,
   scores of +100..+300 drop by 0.13 in q on average by depth 20; -300..-100 rise by 0.06), and
   correcting the bias made it worse, because the correction makes shallow alternatives attractive.
3. A one-sided (null-window) result is only valid against the threshold it was tested with; the
   working rule is alpha-beta's: test challengers at the best move's depth against its value, and
   give a real window only to a move that passes.

### 5b. DFPS, second round (2026-10-19)

Games against alpha-beta at a fixed node budget showed what equal-node decision tests had hidden:

| Change | 5k nodes/move | 600k nodes/move | 10+0.1 |
|---|---|---|---|
| state of 2026-10-17 | -478 +/- 39 | -68 +/- 16 | -85 +/- 11 |
| stopped searches keep finished probes (before: could play the first legal move) | -197 | | |
| probes go straight to their target depth | -140 | | |
| root moves seeded from the root search itself; probes one ply below the parent | -123 +/- 23 | -21 +/- 23 | -55 +/- 20 |

Checks: DFPS whose root search never ends (plain alpha-beta inside the DFPS frame) scored exactly
0 once alpha-beta's fail-high rule was added (it was -45 without it); at a fixed depth DFPS chose
as well as alpha-beta (24 vs 25 better, Stockfish judge) with 18% more nodes; inside real games the
large blunders were equally frequent. What remains at small budgets is overhead, not a bug.

Calibration of one-sided results (`dfpscal`, 200 positions, depth 10): after alpha-beta's root
search, every refuted move was searched again with a full window one ply below. The gap between a
refuted move's fail-soft bound and its full value has median 150 but a heavy tail: the chance that
the move is really better than the chosen one is about 25% when the bound equals the best value,
12% within 20, 7% within 80, 4% within 200 and still about 2% beyond. In 60 of 196 positions such a
move scored more than 30 above the chosen move; Stockfish preferred it 26 times, the chosen move
15 times (18 ties). These moves had mostly been reduced 2-4 plies at the root. Two consequences:
alpha-beta's root reductions are too aggressive (option `RootLmrSub`, under test), and a Gaussian
belief for a refuted move needs a much wider tail than the depth error model gives (this is why
"probability the best move is wrong" saturated near 1 and could not drive time management).

## 6. Order of work

1. Write the measurement harness first: Syzygy exam sets (grader only), mate sets, adversarial
   endgame sets, match settings vs SF-Analysis and Lc0.
2. Layer 1 (exact core) inside the current Vyuha; must cost no Elo.
3. Layer 2 at small scale: self-play from zero, first on reduced material, then full games.
4. Layer 3 teacher, compared at equal GPU hours.
5. Layer 4 adversarial loop.

Main risks: the self-learned engine starts far weaker than today's Vyuha (which uses Stockfish
data); search-contempt's efficiency claim is unreviewed; Layer 4 may find holes faster than
training can close them (as happened to KataGo's defences).

## 7. References

- Tromp, ChessPositionRanking: https://github.com/tromp/ChessPositionRanking
- Proof-tree estimate (Syzygy author), TalkChess: https://talkchess.com/forum2/viewtopic.php?start=20&t=84719
- Schaeffer et al., Checkers is solved, Science 2007: https://www.science.org/cms/asset/7f2147df-b2f1-4748-9e98-1ac3afccfb76/pap.pdf
- Boige, Boumaza, Scherrer, AlphaBeta is not as good as you think, NeurIPS 2025: https://arxiv.org/abs/2506.21996
- Kishimoto, Winands, Müller, Saito, Game-tree search using proof numbers: https://webdocs.cs.ualberta.ca/~mmueller/ps/ICGA2012PNS.pdf
- Kishimoto & Müller, GHI: http://webdocs.cs.ualberta.ca/~mmueller/ghi.html
- Ramanujan, Sabharwal, Selman, ICAPS 2010: https://ojs.aaai.org/index.php/ICAPS/article/view/13437
- Baier & Winands, MCTS–minimax hybrids: https://dke.maastrichtuniversity.nl/m.winands/documents/mcts-minimax_hybrids_final.pdf
- Czech, Korus, Kersting, Monte-Carlo Graph Search: https://arxiv.org/abs/2012.11045
- Danihelka et al., Gumbel, ICLR 2022: https://openreview.net/forum?id=bERaNdoegnO
- Frankston & Howard, RMCTS, 2026: https://arxiv.org/abs/2601.01301
- Hay, Russell, Tolpin, Shimony, Selecting computations: https://arxiv.org/pdf/1207.5879
- Nau et al., When is it better not to look ahead?: http://www.cs.umd.edu/~nau/papers/nau2010when.pdf
- Luštrek, Gams, Bratko, Is real-valued minimax pathological?: https://www.sciencedirect.com/science/article/pii/S0004370206000117
- Silver et al., AlphaZero: https://arxiv.org/abs/1712.01815
- Wu, KataGo: https://arxiv.org/abs/1902.10565
- Joshi, Search-contempt, 2025: https://arxiv.org/abs/2504.07757
- Jones, Scaling scaling laws with board games: https://arxiv.org/abs/2104.03113
- Neumann & Gros, AlphaZero scaling and Zipf's law: https://arxiv.org/abs/2412.11979
- Monroe et al., Chessformer, ICLR 2026: https://arxiv.org/abs/2605.19091
- Ruoss et al., Amortized planning / searchless chess: https://arxiv.org/abs/2402.04494
- Ye et al., DiffuSearch, ICLR 2025: https://arxiv.org/abs/2502.19805
- DeepMind, External and internal planning with language models: https://arxiv.org/abs/2412.12119
- Jenner et al., Learned look-ahead, NeurIPS 2024: https://openreview.net/forum?id=8zg9sO4ttV
- The algorithm is not the behavior, 2025: https://arxiv.org/abs/2508.21380
- Helfenstein, Czech et al., M2CTS: https://arxiv.org/abs/2401.16852
- Elkies, On numbers and endgames: https://arxiv.org/abs/math/9905198 ; Higher nimbers: https://arxiv.org/abs/math/0011253
- Gomboc & Shelton, Endgame compression via logic minimisation: https://rlair.cs.ucr.edu/papers/docs/acg21.pdf
- Sadmine et al., Stockfish or Lc0 against tablebases: https://webdocs.cs.ualberta.ca/~mmueller/ps/2023/ACG_2023_Stockfish.pdf
- Wang et al., Adversarial policies beat superhuman Go AIs: https://proceedings.mlr.press/v202/wang23g.html
- Tseng et al., Can Go AIs be adversarially robust?: https://arxiv.org/abs/2406.12843
- AdvChess, Discovery of adversarial endgame chess positions: https://openreview.net/forum?id=nIYtXmeY3F
- Farhi, Goldstone, Gutmann, Quantum NAND tree: https://arxiv.org/abs/quant-ph/0702144
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- CCRL 40/15: https://computerchess.org.uk/4040/rating_list_all.html ; TCEC S29: https://www.chessdom.com/stockfish-claims-its-19th-tcec-title-defeating-reckless-in-the-superfinal
