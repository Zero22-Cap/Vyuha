// Vyuha - search + UCI.
#include "chess.h"
#include "nnue.h"
#include "../ext/fathom/tbprobe.h"
#include <climits>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <malloc.h>
#ifdef _WIN32
#include <windows.h>
#endif

constexpr int INF = 32001, MATE = 32000, MATE_BOUND = MATE - 512, NONE = 32002, MAXPLY = 120;
// Tablebase wins sit just below the mate range. TB_BOUND marks "decisive" (TB or mate).
constexpr int TB_WIN_SCORE = 31000, TB_BOUND = 30000;
enum { B_NONE = 0, B_UPPER = 1, B_LOWER = 2, B_EXACT = 3 };

struct Options {
    int threads = 16;
    int hashMB = 8192;
    int uncMode = 0;       // uncertainty-scaled pruning/reductions
    int moveOverhead = 100;
    int policyMode = 0, policyWeight = 2000, policyDepth = 5;
    std::string syzygyPath = "H:/Chess/Tablebase/Common";
    int tbProbeLimit = 7, tbProbeDepth = 1;
    bool tb50 = true;
    std::string evalFile;
} OPT;

using Clock = std::chrono::steady_clock;
inline int64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- TT
struct TTEntry { uint16_t key; uint16_t move; int16_t score; int16_t eval; uint8_t depth; uint8_t genBound; };
struct TTCluster { TTEntry e[3]; uint16_t pad; };
static_assert(sizeof(TTCluster) == 32, "cluster");
struct TTable {
    TTCluster* t = nullptr; size_t n = 0; uint8_t gen = 0;
    void resize(size_t mb) {
        if (t) _aligned_free(t);
        n = mb * 1024 * 1024 / sizeof(TTCluster);
        t = (TTCluster*)_aligned_malloc(n * sizeof(TTCluster), 64);
        clear();
    }
    void clear() {
        int nt = std::max(1, OPT.threads);
        std::vector<std::thread> th;
        for (int i = 0; i < nt; i++)
            th.emplace_back([=, this]() { size_t a = n * i / nt, b = n * (i + 1) / nt; std::memset((void*)(t + a), 0, (b - a) * sizeof(TTCluster)); });
        for (auto& x : th) x.join();
        gen = 0;
    }
    TTCluster* cl(U64 k) const { return &t[(size_t)(((unsigned __int128)k * n) >> 64)]; }
    void prefetch(U64 k) const { __builtin_prefetch(cl(k)); }
    TTEntry* probe(U64 k, bool& hit) {
        TTCluster* c = cl(k); uint16_t k16 = uint16_t(k);
        for (auto& e : c->e) if (e.key == k16 && (e.genBound & 3)) { hit = true; return &e; }
        hit = false;
        TTEntry* r = &c->e[0];
        auto val = [&](const TTEntry& x) { return int(x.depth) - 8 * ((gen - (x.genBound >> 3)) & 31); };
        for (auto& e : c->e) {
            if (!(e.genBound & 3)) return &e;
            if (val(e) < val(*r)) r = &e;
        }
        return r;
    }
    // genBound: generation (5 bits) | was-PV flag (bit 2) | bound (2 bits)
    void store(TTEntry* e, U64 k, Move m, int score, int eval, int depth, int bound, bool pv = false) {
        uint16_t k16 = uint16_t(k);
        if (m || e->key != k16) e->move = m;
        if (bound == B_EXACT || e->key != k16 || depth + 4 > e->depth || ((e->genBound >> 3) != gen)) {
            e->key = k16; e->score = int16_t(score); e->eval = int16_t(eval);
            e->depth = uint8_t(std::max(0, depth)); e->genBound = uint8_t((gen << 3) | (pv << 2) | bound);
        }
    }
    int hashfull() const {
        int c = 0;
        for (int i = 0; i < 1000; i++) for (auto& e : t[i].e) c += (e.genBound & 3) && (e.genBound >> 3) == gen;
        return c / 3;
    }
} TT;

inline int to_tt(int s, int ply) { return s >= TB_BOUND ? s + ply : s <= -TB_BOUND ? s - ply : s; }
inline int from_tt(int s, int ply) { return s >= TB_BOUND ? s - ply : s <= -TB_BOUND ? s + ply : s; }

// ---------------------------------------------------------------- search state
int LMR[64][64];

// ---------------------------------------------------------------- tunable search constants
// Hidden UCI options (name, default, min, max) for SPSA tuning with tools/spsa.py; the defaults are
// the hand-set values. "tunelist" prints the table.
#define TUNE_LIST(X) \
    X(RfpMul, 80, 30, 160) X(RfpImp, 70, 0, 150) X(RfpDepth, 8, 4, 12) X(RazorMul, 260, 100, 500) \
    X(NmpBase, 4, 2, 6) X(NmpDiv, 200, 80, 400) X(NmpDepthMul, 20, 0, 50) X(NmpMargin, 180, 50, 350) \
    X(PcMargin, 200, 80, 350) X(PcImp, 60, 0, 150) X(LmpBase, 3, 1, 8) X(FutBase, 120, 40, 250) \
    X(FutMul, 100, 40, 200) X(HistPrune, 3000, 1000, 8000) X(SeeQuiet, 25, 5, 60) X(SeeNoisy, 100, 40, 200) \
    X(SeDepthMul, 16, 6, 40) X(DextMargin, 20, 0, 60) X(LmrBase, 99, 30, 160) X(LmrDiv, 210, 130, 330) \
    X(LmrHist, 8192, 3000, 20000) X(AspDelta, 15, 5, 40) X(HistMul, 300, 100, 600) X(HistSub, 250, 0, 600) \
    X(HistMax, 1800, 800, 3500) X(QsFut, 200, 50, 400) X(CorrMinorW, 0, 0, 256) X(CorrContW, 0, 0, 256) X(TtPvLmr, 0, 0, 2) X(SmpVote, 0, 0, 1) X(RootLmrSub, 0, 0, 3) X(PvLmrSub, 0, 0, 2) X(RootVerify, 0, 0, 300)
#define X(n, d, lo, hi) int n = d;
namespace tp { TUNE_LIST(X) }
#undef X
struct TuneParam { const char* name; int* v; int def, lo, hi; };
#define X(n, d, lo, hi) {#n, &tp::n, d, lo, hi},
const TuneParam TUNES[] = { TUNE_LIST(X) };
#undef X
const TuneParam* find_tune(const std::string& n) {
    for (const TuneParam& tq : TUNES) if (n == tq.name) return &tq;
    return nullptr;
}
void init_lmr() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++) LMR[d][m] = int(tp::LmrBase / 100.0 + std::log(d) * std::log(m) / (tp::LmrDiv / 100.0));
}
std::atomic<bool> STOP{false};
struct Limits { int64_t start = 0, soft = 0, hard = 0; int depth = MAXPLY - 1; uint64_t nodes = 0; bool infinite = false, timed = false; } LIM;

struct Stack { Move move = 0; int staticEval = NONE; int unc = 0; Move excluded = 0; Move killer = 0; int16_t* cont = nullptr; int dext = 0; int pcTo = -1; bool ttPv = false; };
// result of one root move in a search call (DFPS seeds its root children from these)
struct RootRec { Move move; int score, depth, bound, alpha; uint64_t nodes; };

struct Thread {
    int id = 0;
    Board pos[MAXPLY + 8];
    nnue::Acc acc[MAXPLY + 8];
    Stack ssBuf[MAXPLY + 12];
    Stack* ss = ssBuf + 4;
    int16_t mainHist[2][64 * 64];
    int16_t capHist[12][64][6];
    int16_t cont[12 * 64][12 * 64];
    int16_t contDummy[12 * 64];
    Move counter[12][64];
    int16_t corrPawn[2][16384];
    int16_t corrNp[2][2][16384];
    int16_t corrMinor[2][16384];
    int16_t corrCont[12 * 64][12 * 64];  // by the moves two plies and one ply back (piece, to)
    Move pv[MAXPLY + 2][MAXPLY + 2];
    int pvLen[MAXPLY + 2];
    U64 hist[1200];
    int base = 0, repBase = 0;  // repBase: positions after this index repeat as a draw at once
    bool probe = false;         // running a DFPS probe (its root is not the game root)
    std::vector<RootRec>* rootRec = nullptr;  // if set, every root move's result is appended here
    uint64_t nodes = 0, tbHits = 0;
    int seldepth = 0, rootDepth = 0;
    Move bestMove = 0; int bestScore = 0, completedDepth = 0;
    uint64_t rootNodes[4096];
    void clear() {
        std::memset(mainHist, 0, sizeof mainHist); std::memset(capHist, 0, sizeof capHist);
        std::memset(cont, 0, sizeof cont); std::memset(contDummy, 0, sizeof contDummy);
        std::memset(counter, 0, sizeof counter); std::memset(corrPawn, 0, sizeof corrPawn); std::memset(corrNp, 0, sizeof corrNp);
        std::memset(corrMinor, 0, sizeof corrMinor); std::memset(corrCont, 0, sizeof corrCont);
    }
};
std::vector<std::unique_ptr<Thread>> THREADS;
Board ROOT;
std::vector<U64> GAME_KEYS;
std::vector<Move> ROOT_ALLOWED;  // tablebase-filtered root moves (empty = no filter)
int ROOT_TB = 0;                 // root tablebase verdict: 0 none, 1 win, -1 loss, 2 draw

// ---------------------------------------------------------------- eval helpers
int material_eval(const Board& b) {
    static const int v[6] = {200, 600, 640, 1000, 1800, 0};  // on the net's scale (pawn ~200 internal)
    int s = 0;
    for (int pt = 0; pt < 5; pt++) s += v[pt] * (popcnt(b.pieces(WHITE, pt)) - popcnt(b.pieces(BLACK, pt)));
    return b.stm == WHITE ? s : -s;
}

void ensure_acc(Thread& t, int ply) {
    if (t.acc[ply].computed) return;
    int p = ply;
    while (p > 0 && !t.acc[p].computed) p--;
    if (!t.acc[p].computed) { nnue::refresh(t.acc[p], t.pos[p], 0); nnue::refresh(t.acc[p], t.pos[p], 1); t.acc[p].computed = true; }
    for (int q = p + 1; q <= ply; q++) { nnue::update(t.acc[q - 1], t.acc[q], t.pos[q]); t.acc[q].computed = true; }
}

int raw_eval(Thread& t, int ply, int* unc) {
    const Board& b = t.pos[ply];
    if (!nnue::NET) { if (unc) *unc = 40; return material_eval(b); }
    ensure_acc(t, ply);
    if (unc) *unc = nnue::uncertainty(t.acc[ply], b);
    return nnue::evaluate(t.acc[ply], b);
}

constexpr int CG = 256;
// the correction entry of the last two moves before ply (null if one of them is missing or a null move)
inline int16_t* cont_corr(Thread& t, int ply) {
    if (ply < 2) return nullptr;
    int a = t.ss[ply - 2].pcTo, b = t.ss[ply - 1].pcTo;
    return a >= 0 && b >= 0 ? &t.corrCont[a][b] : nullptr;
}
int corrected(Thread& t, const Board& b, int raw, int ply = -1) {
    int c = b.stm;
    int corr = t.corrPawn[c][b.pawnKey & 16383] + (t.corrNp[c][WHITE][b.npKey[WHITE] & 16383] + t.corrNp[c][BLACK][b.npKey[BLACK] & 16383]) / 2;
    if (tp::CorrMinorW) corr += t.corrMinor[c][b.minorKey & 16383] * tp::CorrMinorW / 128;
    if (tp::CorrContW && ply >= 0)
        if (int16_t* e = cont_corr(t, ply)) corr += *e * tp::CorrContW / 128;
    int v = raw + corr / CG;
    v = v * (200 - b.hm) / 200;
    return std::clamp(v, -TB_BOUND + 1, TB_BOUND - 1);
}
void corr_update(Thread& t, const Board& b, int diff, int depth, int ply) {
    int c = b.stm, w = std::min(16, depth + 1);
    auto upd = [&](int16_t& e) { int v = (e * (256 - w) + std::clamp(diff, -250, 250) * CG * w) / 256; e = int16_t(std::clamp(v, -CG * 64, CG * 64)); };
    upd(t.corrPawn[c][b.pawnKey & 16383]);
    upd(t.corrNp[c][WHITE][b.npKey[WHITE] & 16383]);
    upd(t.corrNp[c][BLACK][b.npKey[BLACK] & 16383]);
    if (tp::CorrMinorW) upd(t.corrMinor[c][b.minorKey & 16383]);
    if (tp::CorrContW)
        if (int16_t* e = cont_corr(t, ply)) upd(*e);
}

inline void grav(int16_t& h, int bonus) { h += bonus - h * std::abs(bonus) / 16384; }

bool is_draw(const Thread& t, int ply) {
    const Board& b = t.pos[ply];
    if (b.hm >= 100) return true;
    if (!(b.byPt[PAWN] | b.byPt[ROOK] | b.byPt[QUEEN]) && popcnt(b.occ() ^ b.byPt[KING]) <= 1) return true;
    int idx = t.base + ply, lo = std::max(0, idx - b.hm);
    int earlier = 0;
    for (int j = idx - 4; j >= lo; j -= 2)
        if (t.hist[j] == b.key) {
            if (j > t.repBase) return true;   // repeated inside the search tree
            if (++earlier >= 2) return true;  // third occurrence counting the game history
        }
    return false;
}

inline bool check_stop(Thread& t) {
    if (STOP.load(std::memory_order_relaxed)) return true;
    if (t.id == 0 && (t.nodes & 1023) == 0) {
        if (LIM.timed && t.rootDepth > 1 && now_ms() - LIM.start >= LIM.hard) STOP = true;
        if (LIM.nodes) {
            uint64_t tot = 0; for (auto& th : THREADS) tot += th->nodes;
            if (tot >= LIM.nodes) STOP = true;
        }
    }
    return STOP.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------- move ordering
struct Picker {
    MoveList ml; int nNoisy = 0, idx = 0, stage = 0;
    Move tt, killer, cm; bool skipQuiets = false, quietScored = false;
    Move bad[64]; int nBad = 0, badIdx = 0;
    Thread* t; int ply; const Board* b; int depth; Move polBest = 0;
    Picker(Thread& th, int p, Move ttm, bool qs, int d = 0) : t(&th), ply(p), b(&th.pos[p]), depth(d) {
        gen_moves(*b, ml, true, false); nNoisy = ml.n;
        if (!qs || b->checkers) gen_moves(*b, ml, false, true);
        tt = in_list(ttm) ? ttm : 0;
        killer = th.ss[p].killer; cm = 0;
        Move pm = th.ss[p - 1].move;
        if (p > 0 && pm && pm != NULLMOVE && b->sq[to_sq(pm)] != NO_PIECE) cm = th.counter[b->sq[to_sq(pm)]][to_sq(pm)];
        for (int i = 0; i < nNoisy; i++) {
            Move m = ml.m[i];
            int cap = mtype(m) == MF_EP ? PAWN : b->sq[to_sq(m)] == NO_PIECE ? 5 : ptype(b->sq[to_sq(m)]);
            int val = cap == 5 ? 0 : SEEVal[cap];
            if (mtype(m) == MF_PROMO) val += SEEVal[promo_pt(m)];
            ml.s[i] = val * 16 + th.capHist[b->sq[from_sq(m)]][to_sq(m)][cap];
        }
    }
    bool in_list(Move m) { if (!m) return false; for (int i = 0; i < ml.n; i++) if (ml.m[i] == m) return true; return false; }
    int pick(int lo, int hi) {
        int best = lo;
        for (int i = lo + 1; i < hi; i++) if (ml.s[i] > ml.s[best]) best = i;
        std::swap(ml.m[lo], ml.m[best]); std::swap(ml.s[lo], ml.s[best]);
        return lo;
    }
    void score_quiets() {
        int us = b->stm;
        int16_t* c1 = t->ss[ply - 1].cont; int16_t* c2 = t->ss[ply - 2].cont;
        for (int i = nNoisy; i < ml.n; i++) {
            Move m = ml.m[i]; int pc = b->sq[from_sq(m)], to = to_sq(m);
            int s = t->mainHist[us][from_sq(m) * 64 + to];
            if (c1) s += c1[pc * 64 + to];
            if (c2) s += c2[pc * 64 + to];
            if (mtype(m) == MF_PROMO) s -= 30000;
            ml.s[i] = s;
        }
        if (nnue::POLICY && nnue::NET && OPT.policyMode && depth >= OPT.policyDepth && ml.n - nNoisy > 1) {
            ensure_acc(*t, ply);
            float q[nnue::PD];
            nnue::policy_query(t->acc[ply], us, q);
            int k = b->king(us);
            bool kmir = ((k ^ (us ? 56 : 0)) & 7) >= 4;
            float lg[256], mx = -1e30f; int bi = -1;
            for (int i = nNoisy; i < ml.n; i++) {
                Move m = ml.m[i];
                lg[i] = mtype(m) == MF_PROMO ? -30.f : nnue::policy_logit(q, us, kmir, ptype(b->sq[from_sq(m)]), from_sq(m), to_sq(m));
                if (lg[i] > mx) { mx = lg[i]; bi = i; }
            }
            for (int i = nNoisy; i < ml.n; i++) ml.s[i] += int(OPT.policyWeight * std::max(-8.f, lg[i] - mx));
            if (bi >= 0) polBest = ml.m[bi];
        }
        quietScored = true;
    }
    Move next() {
        switch (stage) {
        case 0: stage = 1; idx = 0; if (tt) return tt; [[fallthrough]];
        case 1:
            while (idx < nNoisy) {
                int i = pick(idx, nNoisy); idx++;
                Move m = ml.m[i];
                if (m == tt) continue;
                if (!b->see_ge(m, -ml.s[i] / 64)) { if (nBad < 64) bad[nBad++] = m; continue; }
                return m;
            }
            stage = 2; [[fallthrough]];
        case 2:
            stage = 3;
            if (!skipQuiets && killer && killer != tt && !b->is_noisy(killer) && in_list(killer)) return killer;
            [[fallthrough]];
        case 3:
            stage = 4;
            if (!skipQuiets && cm && cm != tt && cm != killer && !b->is_noisy(cm) && in_list(cm)) return cm;
            [[fallthrough]];
        case 4:
            if (!skipQuiets) {
                if (!quietScored) { score_quiets(); idx = nNoisy; }
                while (idx < ml.n) {
                    int i = pick(idx, ml.n); idx++;
                    Move m = ml.m[i];
                    if (m == tt || m == killer || m == cm) continue;
                    return m;
                }
            }
            stage = 5; [[fallthrough]];
        case 5:
            while (badIdx < nBad) { Move m = bad[badIdx++]; if (m != tt) return m; }
            stage = 6; [[fallthrough]];
        default: return 0;
        }
    }
};

// ---------------------------------------------------------------- search
inline void make_child(Thread& t, int ply, Move m) {
    Board& c = t.pos[ply + 1];
    c = t.pos[ply];
    if (m == NULLMOVE) c.make_null(); else c.make(m);
    t.acc[ply + 1].computed = false;
    t.ss[ply].move = m;
    if (m == NULLMOVE) { t.ss[ply].cont = t.contDummy; t.ss[ply].pcTo = -1; }
    else {
        int pc = t.pos[ply].sq[from_sq(m)];
        t.ss[ply].cont = t.cont[pc * 64 + to_sq(m)];
        t.ss[ply].pcTo = pc * 64 + to_sq(m);
    }
    t.hist[t.base + ply + 1] = c.key;
}

int qsearch(Thread& t, int ply, int alpha, int beta) {
    t.nodes++;
    if (ply > t.seldepth) t.seldepth = ply;
    if (check_stop(t)) return 0;
    Board& b = t.pos[ply];
    if (is_draw(t, ply)) return 0;
    if (ply >= MAXPLY - 1) return b.checkers ? 0 : corrected(t, b, raw_eval(t, ply, nullptr));
    bool pvNode = beta - alpha > 1;
    bool hit; TTEntry* e = TT.probe(b.key, hit);
    int ttScore = hit ? from_tt(e->score, ply) : NONE;
    int ttBound = hit ? (e->genBound & 3) : 0;
    Move ttMove = hit ? e->move : 0;
    if (!pvNode && hit && (ttBound == B_EXACT || (ttBound == B_LOWER && ttScore >= beta) || (ttBound == B_UPPER && ttScore <= alpha)))
        return ttScore;
    int best, standPat = -INF, raw = NONE;
    if (b.checkers) best = -INF;
    else {
        raw = hit && e->eval != NONE ? e->eval : raw_eval(t, ply, nullptr);
        standPat = corrected(t, b, raw, ply);
        if (hit && (ttBound & (ttScore > standPat ? B_LOWER : B_UPPER))) standPat = ttScore;
        if (standPat >= beta) {
            if (!hit) TT.store(e, b.key, 0, to_tt(standPat, ply), raw, 0, B_LOWER);
            return standPat;
        }
        if (standPat > alpha) alpha = standPat;
        best = standPat;
    }
    Picker pk(t, ply, ttMove, true);
    Move bestMove = 0; int moveCount = 0;
    while (Move m = pk.next()) {
        if (!b.legal(m)) continue;
        moveCount++;
        if (!b.checkers) {
            if (mtype(m) != MF_PROMO) {
                int cap = mtype(m) == MF_EP ? PAWN : ptype(b.sq[to_sq(m)]);
                if (standPat + SEEVal[cap] + tp::QsFut <= alpha) { best = std::max(best, standPat + SEEVal[cap] + tp::QsFut); continue; }
            }
            if (!b.see_ge(m, 0)) continue;
        } else if (best > -TB_BOUND && moveCount > 3 && !b.is_noisy(m)) continue;
        make_child(t, ply, m);
        int score = -qsearch(t, ply + 1, -beta, -alpha);
        if (STOP) return 0;
        if (score > best) {
            best = score;
            if (score > alpha) { bestMove = m; if (score >= beta) break; alpha = score; }
        }
    }
    if (b.checkers && moveCount == 0) return -MATE + ply;
    TT.store(e, b.key, bestMove, to_tt(best, ply), raw, 0, best >= beta ? B_LOWER : B_UPPER);
    return best;
}

int search(Thread& t, int ply, int alpha, int beta, int depth, bool cutNode) {
    bool pvNode = beta - alpha > 1, root = ply == 0;
    Board& b = t.pos[ply];
    t.pvLen[ply] = ply;
    if (depth <= 0) return qsearch(t, ply, alpha, beta);
    t.nodes++;
    if (ply > t.seldepth) t.seldepth = ply;
    if (!root) {
        if (check_stop(t)) return 0;
        if (is_draw(t, ply)) return 0;
        if (ply >= MAXPLY - 1) return b.checkers ? 0 : corrected(t, b, raw_eval(t, ply, nullptr));
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }
    Stack* ss = &t.ss[ply];
    Move excluded = ss->excluded;
    bool hit = false; TTEntry* e = nullptr;
    int ttScore = NONE, ttBound = 0, ttDepth = -1; Move ttMove = 0;
    if (!excluded) {
        e = TT.probe(b.key, hit);
        if (hit) { ttScore = from_tt(e->score, ply); ttBound = e->genBound & 3; ttDepth = e->depth; ttMove = e->move; }
    }
    if (root && t.bestMove) ttMove = t.bestMove;
    // on or once on the PV (kept in the TT); a singular search keeps its node's flag
    bool ttPv = excluded ? ss->ttPv : pvNode || (hit && (e->genBound & 4));
    ss->ttPv = ttPv;
    if (!pvNode && hit && ttDepth >= depth && b.hm < 90 &&
        ((ttBound == B_LOWER && ttScore >= beta) || (ttBound == B_UPPER && ttScore <= alpha) || ttBound == B_EXACT))
        return ttScore;

    // Syzygy probe (WDL only, right after a capture or pawn move, no castling rights)
    int tbMin = -INF, tbMax = INF;
    if (!root && !excluded && TB_LARGEST && b.hm == 0 && !b.castle) {
        int pc = popcnt(b.occ()), lim = std::min<int>(OPT.tbProbeLimit, TB_LARGEST);
        if (pc <= lim && (pc < lim || depth >= OPT.tbProbeDepth)) {
            unsigned r = tb_probe_wdl(b.byC[WHITE], b.byC[BLACK], b.byPt[KING], b.byPt[QUEEN], b.byPt[ROOK], b.byPt[BISHOP],
                                      b.byPt[KNIGHT], b.byPt[PAWN], 0, 0, b.ep == NO_SQ ? 0 : b.ep, b.stm == WHITE);
            if (r != TB_RESULT_FAILED) {
                t.tbHits++;
                int v, bound;
                if (r == TB_WIN || (r == TB_CURSED_WIN && !OPT.tb50)) { v = TB_WIN_SCORE - ply; bound = B_LOWER; }
                else if (r == TB_LOSS || (r == TB_BLESSED_LOSS && !OPT.tb50)) { v = -TB_WIN_SCORE + ply; bound = B_UPPER; }
                else { v = r == TB_CURSED_WIN ? 1 : r == TB_BLESSED_LOSS ? -1 : 0; bound = B_EXACT; }
                if (bound == B_EXACT || (bound == B_LOWER ? v >= beta : v <= alpha)) {
                    TT.store(e, b.key, 0, to_tt(v, ply), NONE, std::min(MAXPLY - 1, depth + 6), bound, ttPv);
                    return v;
                }
                if (pvNode) {
                    if (bound == B_LOWER) { tbMin = v; alpha = std::max(alpha, v); }
                    else tbMax = v;
                }
            }
        }
    }

    bool inCheck = b.checkers != 0;
    int raw = NONE, eval = NONE, unc = 0;
    if (inCheck) { ss->staticEval = NONE; ss->unc = 0; }
    else if (excluded) { eval = ss->staticEval; unc = ss->unc; }
    else {
        raw = raw_eval(t, ply, &unc);
        ss->staticEval = eval = corrected(t, b, raw, ply);
        ss->unc = unc;
        if (hit && (ttBound & (ttScore > eval ? B_LOWER : B_UPPER))) eval = ttScore;
    }
    bool improving = false;
    if (!inCheck) {
        if (t.ss[ply - 2].staticEval != NONE) improving = ss->staticEval > t.ss[ply - 2].staticEval;
        else if (t.ss[ply - 4].staticEval != NONE) improving = ss->staticEval > t.ss[ply - 4].staticEval;
    }
    t.ss[ply + 1].killer = 0;
    t.ss[ply + 1].excluded = 0;
    // uncertainty scale (1024 = neutral): prune harder when the net is confident, softer when unsure
    int uscale = 1024;
    if (OPT.uncMode && !inCheck) uscale = std::clamp(640 + unc * 12, 700, 1600);

    if (!pvNode && !inCheck && !excluded) {
        int rfpMargin = (tp::RfpMul * depth - tp::RfpImp * improving) * uscale / 1024;
        if (depth <= tp::RfpDepth && eval - rfpMargin >= beta && eval < TB_BOUND) return (eval + beta) / 2;
        if (depth <= 3 && eval + tp::RazorMul * depth < alpha) {
            int v = qsearch(t, ply, alpha, alpha + 1);
            if (v <= alpha) return v;
        }
        if (depth >= 3 && eval >= beta && ss->staticEval >= beta - tp::NmpDepthMul * depth + tp::NmpMargin && t.ss[ply - 1].move != NULLMOVE &&
            b.non_pawn(b.stm) && beta > -TB_BOUND) {
            int R = tp::NmpBase + depth / 3 + std::min(3, (eval - beta) / tp::NmpDiv);
            make_child(t, ply, NULLMOVE);
            int v = -search(t, ply + 1, -beta, -beta + 1, depth - R, !cutNode);
            if (STOP) return 0;
            if (v >= beta) return v >= TB_BOUND ? beta : v;
        }
        int pcBeta = beta + tp::PcMargin - tp::PcImp * improving;
        if (depth >= 5 && std::abs(beta) < TB_BOUND && !(hit && ttDepth >= depth - 3 && ttScore < pcBeta)) {
            Picker pk(t, ply, ttMove && b.is_noisy(ttMove) ? ttMove : 0, true);
            while (Move m = pk.next()) {
                if (!b.is_noisy(m) || !b.legal(m) || !b.see_ge(m, pcBeta - ss->staticEval)) continue;
                make_child(t, ply, m);
                int v = -qsearch(t, ply + 1, -pcBeta, -pcBeta + 1);
                if (v >= pcBeta) v = -search(t, ply + 1, -pcBeta, -pcBeta + 1, depth - 4, !cutNode);
                if (STOP) return 0;
                if (v >= pcBeta) {
                    TT.store(e, b.key, m, to_tt(v, ply), raw, depth - 3, B_LOWER, ttPv);
                    return v;
                }
            }
        }
    }
    if (depth >= 4 && !ttMove && !excluded) depth--;

    Picker pk(t, ply, ttMove, false, depth);
    Move quiets[64], caps[32]; int nq = 0, nc = 0;
    int best = tbMin, moveCount = 0;
    Move bestMove = 0;
    while (Move m = pk.next()) {
        if (m == excluded || !b.legal(m)) continue;
        if (root && !t.probe && !ROOT_ALLOWED.empty() && std::find(ROOT_ALLOWED.begin(), ROOT_ALLOWED.end(), m) == ROOT_ALLOWED.end()) continue;
        moveCount++;
        bool noisy = b.is_noisy(m);
        int pc = b.sq[from_sq(m)], to = to_sq(m);
        int hist = 0;
        if (!noisy) {
            hist = t.mainHist[b.stm][from_sq(m) * 64 + to];
            if (t.ss[ply - 1].cont) hist += t.ss[ply - 1].cont[pc * 64 + to];
            if (t.ss[ply - 2].cont) hist += t.ss[ply - 2].cont[pc * 64 + to];
        }
        if (!root && best > -TB_BOUND && b.non_pawn(b.stm)) {
            int lmrDepth = std::max(0, depth - 1 - LMR[std::min(depth, 63)][std::min(moveCount, 63)]);
            if (!noisy) {
                if (moveCount >= (tp::LmpBase + depth * depth) / (2 - improving)) { pk.skipQuiets = true; continue; }
                if (!inCheck && lmrDepth <= 8 && ss->staticEval + (tp::FutBase + tp::FutMul * lmrDepth) * uscale / 1024 <= alpha) { pk.skipQuiets = true; continue; }
                if (lmrDepth <= 4 && hist < -tp::HistPrune * depth) continue;
                if (!b.see_ge(m, -tp::SeeQuiet * lmrDepth * lmrDepth)) continue;
            } else if (depth <= 8 && !b.see_ge(m, -tp::SeeNoisy * depth)) continue;
        }
        int ext = 0;
        if (!root && m == ttMove && !excluded && depth >= 7 && ttDepth >= depth - 3 && (ttBound & B_LOWER) &&
            std::abs(ttScore) < TB_BOUND && ply < 2 * t.rootDepth) {
            int sBeta = ttScore - tp::SeDepthMul * depth / 16;
            ss->excluded = m;
            int v = search(t, ply, sBeta - 1, sBeta, (depth - 1) / 2, cutNode);
            ss->excluded = 0;
            if (STOP) return 0;
            if (v < sBeta) {
                ext = 1;
                if (!pvNode && v < sBeta - tp::DextMargin && ss->dext <= 8) ext = 2;
            } else if (sBeta >= beta) return sBeta;
            else if (ttScore >= beta) ext = -1;
            else if (cutNode) ext = -1;
        }
        t.ss[ply + 1].dext = ss->dext + (ext == 2);
        make_child(t, ply, m);
        TT.prefetch(t.pos[ply + 1].key);
        uint64_t n0 = t.nodes;
        int newDepth = depth - 1 + ext;
        int score = -INF, searched = newDepth, alpha0 = alpha;
        bool givesCheck = t.pos[ply + 1].checkers != 0, full = false;
        if (depth >= 2 && moveCount > 1 + root) {
            int r = LMR[std::min(depth, 63)][std::min(moveCount, 63)];
            r += !pvNode;
            r -= tp::TtPvLmr * (ttPv && !pvNode);
            if (root) r -= tp::RootLmrSub;  // measured: root moves reduced 2-4 plies hide better moves
            else if (pvNode) r -= tp::PvLmrSub;
            r += cutNode;
            r -= improving;
            r -= givesCheck;
            if (noisy) r = r / 2;
            else r -= hist / tp::LmrHist;
            if (ttMove && b.is_noisy(ttMove)) r++;
            if (OPT.uncMode && !inCheck) r += (uscale < 850) - (uscale > 1300);
            if (OPT.policyMode && m == pk.polBest && pk.polBest) r--;
            int d = std::clamp(newDepth - r, 1, newDepth + 1);
            score = -search(t, ply + 1, -alpha - 1, -alpha, d, true);
            searched = d;
            // re-search at full depth after a fail high, and at the root also after a reduced fail low
            // that came close (dfpscal: such moves often beat the best when searched fully)
            if (d < newDepth && (score > alpha || (root && tp::RootVerify && score > alpha - tp::RootVerify))) {
                score = -search(t, ply + 1, -alpha - 1, -alpha, newDepth, !cutNode); searched = newDepth;
            }
        } else if (!pvNode || moveCount > 1)
            score = -search(t, ply + 1, -alpha - 1, -alpha, newDepth, !cutNode);
        if (pvNode && (moveCount == 1 || score > alpha)) {
            score = -search(t, ply + 1, -beta, -alpha, newDepth, false);
            searched = newDepth; full = true;
        }
        if (STOP) return 0;
        if (root) t.rootNodes[m & 4095] += t.nodes - n0;
        if (root && t.rootRec) {
            int bnd = score <= alpha0 ? B_UPPER : score >= beta || !full ? B_LOWER : B_EXACT;
            t.rootRec->push_back({m, score, searched, bnd, alpha0, t.nodes - n0});
        }
        if (score > best) {
            best = score;
            if (score > alpha) {
                bestMove = m;
                if (pvNode) {
                    t.pv[ply][ply] = m;
                    for (int i = ply + 1; i < t.pvLen[ply + 1]; i++) t.pv[ply][i] = t.pv[ply + 1][i];
                    t.pvLen[ply] = std::max(ply + 1, t.pvLen[ply + 1]);
                }
                if (score >= beta) break;
                alpha = score;
            }
        }
        if (m != bestMove) {
            if (!noisy && nq < 64) quiets[nq++] = m;
            else if (noisy && nc < 32) caps[nc++] = m;
        }
    }
    if (moveCount == 0) return excluded ? alpha : inCheck ? -MATE + ply : 0;

    if (best >= beta) {
        int bonus = std::min(tp::HistMax, tp::HistMul * depth - tp::HistSub);
        auto capIdx = [&](Move m) { return mtype(m) == MF_EP ? PAWN : b.sq[to_sq(m)] == NO_PIECE ? 5 : ptype(b.sq[to_sq(m)]); };
        if (!b.is_noisy(bestMove)) {
            auto upd = [&](Move m, int bn) {
                int pc = b.sq[from_sq(m)], to = to_sq(m);
                grav(t.mainHist[b.stm][from_sq(m) * 64 + to], bn);
                if (t.ss[ply - 1].cont) grav(t.ss[ply - 1].cont[pc * 64 + to], bn);
                if (t.ss[ply - 2].cont) grav(t.ss[ply - 2].cont[pc * 64 + to], bn);
            };
            upd(bestMove, bonus);
            for (int i = 0; i < nq; i++) upd(quiets[i], -bonus);
            ss->killer = bestMove;
            Move pm = t.ss[ply - 1].move;
            if (ply > 0 && pm && pm != NULLMOVE && b.sq[to_sq(pm)] != NO_PIECE) t.counter[b.sq[to_sq(pm)]][to_sq(pm)] = bestMove;
        } else grav(t.capHist[b.sq[from_sq(bestMove)]][to_sq(bestMove)][capIdx(bestMove)], bonus);
        for (int i = 0; i < nc; i++) grav(t.capHist[b.sq[from_sq(caps[i])]][to_sq(caps[i])][capIdx(caps[i])], -bonus);
    }
    if (tbMax < INF) best = std::min(best, tbMax);
    if (!excluded) {
        int bound = best >= beta ? B_LOWER : (pvNode && bestMove) ? B_EXACT : B_UPPER;
        TT.store(e, b.key, bestMove, to_tt(best, ply), raw, depth, bound, ttPv);
        if (!inCheck && (!bestMove || !b.is_noisy(bestMove)) && !(bound == B_LOWER && best <= ss->staticEval) &&
            !(bound == B_UPPER && best >= ss->staticEval) && std::abs(best) < TB_BOUND)
            corr_update(t, b, best - ss->staticEval, depth, ply);
    }
    return best;
}

// ---------------------------------------------------------------- iterative deepening
std::string score_str(int s) {
    if (s >= MATE_BOUND) return "mate " + std::to_string((MATE - s + 1) / 2);
    if (s <= -MATE_BOUND) return "mate " + std::to_string(-(MATE + s) / 2);
    if (s >= TB_BOUND) return "cp " + std::to_string(20000 - (TB_WIN_SCORE - s));
    if (s <= -TB_BOUND) return "cp " + std::to_string(-20000 + (TB_WIN_SCORE + s));
    return "cp " + std::to_string(s * 3 / 5);
}

void iterate(Thread& t) {
    t.nodes = 0; t.tbHits = 0; t.seldepth = 0; t.completedDepth = 0; t.bestMove = 0;
    std::memset(t.rootNodes, 0, sizeof t.rootNodes);
    for (auto& s : t.ssBuf) s = Stack();
    t.ss = t.ssBuf + 4;
    t.pos[0] = ROOT;
    t.acc[0].computed = false;
    int n = int(std::min<size_t>(GAME_KEYS.size(), 1000));
    for (int i = 0; i < n; i++) t.hist[i] = GAME_KEYS[GAME_KEYS.size() - n + i];
    t.base = n - 1;
    t.repBase = t.base;
    int score = 0, prevScore = 0, stable = 0; Move prevBest = 0;
    for (int d = 1; d <= LIM.depth; d++) {
        t.rootDepth = d; t.seldepth = 0;
        int delta = tp::AspDelta, alpha = -INF, beta = INF;
        if (d >= 4) { alpha = std::max(-INF, score - delta); beta = std::min(INF, score + delta); }
        while (true) {
            int v = search(t, 0, alpha, beta, d, false);
            if (STOP) break;
            if (v <= alpha) { beta = (alpha + beta) / 2; alpha = std::max(-INF, v - delta); }
            else if (v >= beta) { beta = std::min(INF, v + delta); if (t.pvLen[0] > 0) t.bestMove = t.pv[0][0]; }
            else { score = v; break; }
            delta += delta / 2;
            if (delta > 1500) { alpha = -INF; beta = INF; }
        }
        if (STOP && d > 1) break;
        t.completedDepth = d; t.bestMove = t.pv[0][0]; t.bestScore = score;
        if (t.id == 0) {
            uint64_t tot = 0, tbh = 0; for (auto& th : THREADS) { tot += th->nodes; tbh += th->tbHits; }
            int64_t el = std::max<int64_t>(1, now_ms() - LIM.start);
            // root is in the tablebases: show the certain verdict unless the search has found a mate
            int shown = score;
            if (ROOT_TB && std::abs(score) < MATE_BOUND) shown = ROOT_TB == 1 ? TB_WIN_SCORE : ROOT_TB == -1 ? -TB_WIN_SCORE : 0;
            std::string pv;
            for (int i = 0; i < t.pvLen[0]; i++) pv += move_str(t.pv[0][i]) + " ";
            std::cout << "info depth " << d << " seldepth " << t.seldepth << " multipv 1 score " << score_str(shown)
                      << " nodes " << tot << " nps " << tot * 1000 / el << " hashfull " << TT.hashfull() << " tbhits " << tbh << " time " << el
                      << " pv " << pv << std::endl;
            if (LIM.timed) {
                stable = (t.bestMove == prevBest) ? std::min(stable + 1, 10) : 0;
                prevBest = t.bestMove;
                double frac = t.nodes ? double(t.rootNodes[t.bestMove & 4095]) / double(t.nodes) : 0.5;
                double scale = (1.6 - frac) * 1.3 * (stable >= 4 ? 0.85 : stable == 0 ? 1.3 : 1.0);
                if (d > 1 && score < prevScore - 30) scale *= 1.2;
                prevScore = score;
                if (now_ms() - LIM.start >= LIM.soft * scale) break;
            }
        }
        if (STOP) break;
    }
}

// Rank the root moves with the DTZ tables (Fathom port of Stockfish's root ranking) and keep the
// top-ranked group: winning moves stay winning, and once a position has repeated only the
// DTZ-optimal moves remain, which guarantees progress under the 50-move rule.
void root_tb_probe() {
    ROOT_ALLOWED.clear(); ROOT_TB = 0;
    const Board& b = ROOT;
    if (!TB_LARGEST || b.castle || popcnt(b.occ()) > std::min<int>(OPT.tbProbeLimit, TB_LARGEST)) return;
    bool rep = false;
    int n = int(GAME_KEYS.size());
    for (int i = std::max(0, n - 1 - b.hm); i < n && !rep; i++)
        for (int j = i + 1; j < n; j++) if (GAME_KEYS[i] == GAME_KEYS[j]) { rep = true; break; }
    static TbRootMoves rm;
    unsigned ep = b.ep == NO_SQ ? 0 : b.ep;
    int ok = tb_probe_root_dtz(b.byC[WHITE], b.byC[BLACK], b.byPt[KING], b.byPt[QUEEN], b.byPt[ROOK], b.byPt[BISHOP], b.byPt[KNIGHT],
                               b.byPt[PAWN], b.hm, 0, ep, b.stm == WHITE, rep, OPT.tb50, &rm);
    if (!ok)
        ok = tb_probe_root_wdl(b.byC[WHITE], b.byC[BLACK], b.byPt[KING], b.byPt[QUEEN], b.byPt[ROOK], b.byPt[BISHOP], b.byPt[KNIGHT],
                               b.byPt[PAWN], b.hm, 0, ep, b.stm == WHITE, OPT.tb50, &rm);
    if (!ok || rm.size == 0) return;
    int bestRank = INT_MIN, bestScore = 0;
    for (unsigned i = 0; i < rm.size; i++) bestRank = std::max(bestRank, rm.moves[i].tbRank);
    MoveList ml; gen_moves(b, ml, true, true);
    static const int promo[5] = {0, QUEEN, ROOK, BISHOP, KNIGHT};
    for (unsigned i = 0; i < rm.size; i++) {
        if (rm.moves[i].tbRank != bestRank) continue;
        int f = TB_MOVE_FROM(rm.moves[i].move), to = TB_MOVE_TO(rm.moves[i].move), pr = TB_MOVE_PROMOTES(rm.moves[i].move);
        for (int k = 0; k < ml.n; k++) {
            Move m = ml.m[k];
            if (from_sq(m) != f || to_sq(m) != to || !b.legal(m)) continue;
            if ((mtype(m) == MF_PROMO) != (pr != 0)) continue;
            if (pr && promo_pt(m) != promo[pr]) continue;
            ROOT_ALLOWED.push_back(m); bestScore = rm.moves[i].tbScore;
        }
    }
    if (ROOT_ALLOWED.empty()) return;
    ROOT_TB = bestScore > 20000 ? 1 : bestScore < -20000 ? -1 : 2;
}

void init_tb() {
    if (OPT.syzygyPath.empty() || OPT.syzygyPath == "<empty>") { tb_free(); return; }
    tb_init(OPT.syzygyPath.c_str());
    std::cout << "info string Syzygy: tables up to " << TB_LARGEST << " pieces at " << OPT.syzygyPath << std::endl;
}

#include "dfps.h"
#include "selfplay.h"

std::thread MAIN_THREAD;

void start_search() {
    STOP = false;
    TT.gen = (TT.gen + 1) & 31;
    root_tb_probe();
    Move m = 0;
    if (dfps::P.mode == 1) m = dfps::run();
    else {
        std::vector<std::thread> helpers;
        for (size_t i = 1; i < THREADS.size(); i++) helpers.emplace_back([i]() { iterate(*THREADS[i]); });
        iterate(*THREADS[0]);
        if (LIM.infinite) while (!STOP) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        STOP = true;
        for (auto& h : helpers) h.join();
        Thread* bt = THREADS[0].get();
        if (tp::SmpVote && THREADS.size() > 1) {
            // vote: every thread backs its best move with (score above the worst + 14) x depth
            int minScore = INF;
            for (auto& th : THREADS) if (th->completedDepth > 0) minScore = std::min(minScore, th->bestScore);
            std::vector<std::pair<Move, int64_t>> votes;
            for (auto& th : THREADS) {
                if (th->completedDepth <= 0 || !th->bestMove) continue;
                int64_t v = int64_t(th->bestScore - minScore + 14) * th->completedDepth;
                auto it = std::find_if(votes.begin(), votes.end(), [&](auto& p) { return p.first == th->bestMove; });
                if (it == votes.end()) votes.push_back({th->bestMove, v}); else it->second += v;
            }
            auto vote = [&](Move mv) { for (auto& p : votes) if (p.first == mv) return p.second; return int64_t(0); };
            for (auto& th : THREADS) {
                if (th->completedDepth <= 0 || !th->bestMove) continue;
                if (std::abs(bt->bestScore) >= MATE_BOUND) {
                    if (th->bestScore > bt->bestScore) bt = th.get();  // a shorter win or a longer loss
                } else if (th->bestScore >= MATE_BOUND || vote(th->bestMove) > vote(bt->bestMove)) bt = th.get();
            }
        } else
            for (auto& th : THREADS)
                if (th->completedDepth > bt->completedDepth + 1 && th->bestMove && std::abs(th->bestScore) < MATE_BOUND) bt = th.get();
        m = bt->bestMove;
    }
    if (!m) { MoveList ml; gen_moves(ROOT, ml, true, true); for (int i = 0; i < ml.n; i++) if (ROOT.legal(ml.m[i])) { m = ml.m[i]; break; } }
    std::cout << "bestmove " << move_str(m) << std::endl;
}

void set_threads(int n) {
    THREADS.clear();
    for (int i = 0; i < n; i++) { auto t = std::make_unique<Thread>(); t->id = i; t->clear(); THREADS.push_back(std::move(t)); }
}

std::string exe_dir() {
    char buf[MAX_PATH]; GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string s(buf); size_t p = s.find_last_of("\\/");
    return p == std::string::npos ? "." : s.substr(0, p);
}

void bench(int depth) {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "6k1/5pp1/1q2p2p/4P3/P3P3/3b2P1/1p1B2BP/1Q5K w - - 0 40",
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 1 9",
        "2r3k1/pp3ppp/4p3/3pP3/3P4/P3K3/1P3PPP/2R5 b - - 0 25",
        "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1"};
    uint64_t total = 0; int64_t t0 = now_ms();
    for (auto f : fens) {
        ROOT.set_fen(f); GAME_KEYS = {ROOT.key};
        LIM = Limits(); LIM.start = now_ms(); LIM.depth = depth;
        for (auto& th : THREADS) th->clear();
        TT.clear();
        std::cout.setstate(std::ios_base::failbit);
        start_search();
        std::cout.clear();
        for (auto& th : THREADS) total += th->nodes;
    }
    int64_t el = std::max<int64_t>(1, now_ms() - t0);
    std::cout << "bench nodes " << total << " time " << el << " nps " << total * 1000 / el << std::endl;
}

int main(int argc, char** argv) {
    init_tables();
    init_lmr();
    OPT.evalFile = exe_dir() + "/vyuha.nnue";
    nnue::load(OPT.evalFile);
    nnue::load_policy(exe_dir() + "/vyuha.pol");
    dfps::P.modelFile = exe_dir() + "/vyuha.dfps";
    dfps::MODEL.load(dfps::P.modelFile);
    ROOT.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    GAME_KEYS = {ROOT.key};
    bool inited = false;
    auto lazy_init = [&]() { if (!inited) { set_threads(OPT.threads); TT.resize(OPT.hashMB); init_tb(); inited = true; } };
    if (argc > 1 && std::string(argv[1]) == "bench") {
        OPT.threads = 1; OPT.hashMB = 64; OPT.syzygyPath = ""; lazy_init();
        bench(argc > 2 ? std::atoi(argv[2]) : 12);
        return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd; is >> cmd;
        if (cmd == "uci") {
            std::cout << "id name Vyuha 0.4-dev\nid author Zero22-Cap\n"
                      << "option name Threads type spin default 16 min 1 max 256\n"
                      << "option name Hash type spin default 8192 min 1 max 65536\n"
                      << "option name UncMode type spin default 0 min 0 max 1\n"
                      << "option name Move Overhead type spin default 100 min 0 max 5000\n"
                      << "option name PolicyMode type spin default 0 min 0 max 1\n"
                      << "option name PolicyWeight type spin default 2000 min 0 max 20000\n"
                      << "option name PolicyDepth type spin default 5 min 1 max 30\n"
                      << "option name PolicyFile type string default <exe>/vyuha.pol\n"
                      << "option name SyzygyPath type string default H:/Chess/Tablebase/Common\n"
                      << "option name SyzygyProbeLimit type spin default 7 min 0 max 7\n"
                      << "option name SyzygyProbeDepth type spin default 1 min 1 max 100\n"
                      << "option name Syzygy50MoveRule type check default true\n"
                      << "option name EvalFile type string default <exe>/vyuha.nnue\n"
                      << "option name NoNet type check default false\n"
                      << "option name SearchMode type combo default AB var AB var DFPS\n"
                      << "option name DfpsInitDepth type spin default 6 min 1 max 40\n"
                      << "option name DfpsStep type spin default 1 min 1 max 8\n"
                      << "option name DfpsExpandDepth type spin default 8 min 2 max 60\n"
                      << "option name DfpsMinChildDepth type spin default 4 min 1 max 40\n"
                      << "option name DfpsWindow type spin default 150 min 10 max 3000\n"
                      << "option name DfpsMaxTree type spin default 100000 min 2000 max 4000000\n"
                      << "option name DfpsRho type spin default 50 min 0 max 95\n"
                      << "option name DfpsModel type string default <exe>/vyuha.dfps\n"
                      << "option name DfpsMaxTreePly type spin default 1 min 1 max 100\n"
                      << "option name DfpsBoundShift type spin default 30 min 0 max 500\n"
                      << "option name DfpsQScale type spin default 150 min 0 max 1000\n"
                      << "uciok" << std::endl;
        } else if (cmd == "isready") { lazy_init(); std::cout << "readyok" << std::endl; }
        else if (cmd == "setoption") {
            std::string tok, name, value; is >> tok;
            while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
            std::getline(is, value); if (!value.empty() && value[0] == ' ') value.erase(0, 1);
            if (name == "Threads") { OPT.threads = std::max(1, std::stoi(value)); if (inited) set_threads(OPT.threads); }
            else if (name == "Hash") { OPT.hashMB = std::max(1, std::stoi(value)); if (inited) TT.resize(OPT.hashMB); }
            else if (name == "UncMode") OPT.uncMode = std::stoi(value);
            else if (name == "Move Overhead") OPT.moveOverhead = std::stoi(value);
            else if (name == "PolicyMode") OPT.policyMode = std::stoi(value);
            else if (name == "SyzygyPath") { OPT.syzygyPath = value; if (inited) init_tb(); }
            else if (name == "SyzygyProbeLimit") OPT.tbProbeLimit = std::stoi(value);
            else if (name == "SyzygyProbeDepth") OPT.tbProbeDepth = std::stoi(value);
            else if (name == "Syzygy50MoveRule") OPT.tb50 = (value == "true");
            else if (name == "PolicyWeight") OPT.policyWeight = std::stoi(value);
            else if (name == "PolicyDepth") OPT.policyDepth = std::stoi(value);
            else if (name == "PolicyFile" && value.find("<exe>") == std::string::npos) {
                if (nnue::load_policy(value)) std::cout << "info string loaded policy " << value << std::endl;
                else std::cout << "info string FAILED to load policy " << value << std::endl;
            }
            else if (name == "EvalFile" && value.find("<exe>") == std::string::npos) {
                if (nnue::load(value)) std::cout << "info string loaded net " << value << std::endl;
                else std::cout << "info string FAILED to load net " << value << std::endl;
            }
            else if (name == "SearchMode") dfps::P.mode = (value == "DFPS" || value == "dfps") ? 1 : 0;
            else if (name == "DfpsInitDepth") dfps::P.initDepth = std::clamp(std::stoi(value), 1, 40);
            else if (name == "DfpsStep") dfps::P.step = std::clamp(std::stoi(value), 1, 8);
            else if (name == "DfpsExpandDepth") dfps::P.expandDepth = std::clamp(std::stoi(value), 2, 60);
            else if (name == "DfpsMinChildDepth") dfps::P.minChildDepth = std::clamp(std::stoi(value), 1, 40);
            else if (name == "DfpsWindow") dfps::P.window = std::clamp(std::stoi(value), 10, 3000);
            else if (name == "DfpsMaxTree") dfps::P.maxTree = std::clamp(std::stoi(value), 2000, 4000000);
            else if (name == "DfpsRho") dfps::P.rho = std::clamp(std::stoi(value), 0, 95);
            else if (name == "DfpsMaxTreePly") dfps::P.maxTreePly = std::clamp(std::stoi(value), 1, 100);
            else if (name == "DfpsBoundShift") dfps::P.boundShift = std::clamp(std::stoi(value), 0, 500);
            else if (name == "DfpsQScale") dfps::P.qscale = std::clamp(std::stoi(value), 0, 1000);
            else if (name == "DfpsDebug") dfps::P.debug = std::stoi(value);
            else if (name == "DfpsBias") dfps::P.useBias = std::stoi(value);
            else if (name == "NoNet") { if (value == "true") nnue::NET = nullptr; }  // material only (generation 0 baseline)
            else if (name == "DfpsRetest") dfps::P.retest = std::clamp(std::stoi(value), 0, 1000);
            else if (name == "DfpsTestCost") dfps::P.testCost = std::clamp(std::stoi(value), 1, 64);
            else if (name == "DfpsSeed") dfps::P.seed = value == "true" || value == "1";
            else if (name == "DfpsLmr") dfps::P.lmr = value == "true" || value == "1";
            else if (name == "DfpsStrict") dfps::P.strict = value == "true" || value == "1";
            else if (name == "DfpsPly1") dfps::P.ply1 = value == "true" || value == "1";
            else if (name == "DfpsVerify") dfps::P.verify = std::clamp(std::stoi(value), 0, 2000);
            else if (name == "DfpsTm") dfps::P.tm = std::clamp(std::stoi(value), 0, 1);
            else if (name == "DfpsSkipP") dfps::P.skipP = std::clamp(std::stoi(value), 0, 5000);
            else if (name == "DfpsSkipMax") dfps::P.skipMax = std::clamp(std::stoi(value), 0, 10);
            else if (name == "DfpsSafe") dfps::P.safe = std::clamp(std::stoi(value), 0, 2);
            else if (const TuneParam* tq = find_tune(name)) {
                *tq->v = std::clamp(std::stoi(value), tq->lo, tq->hi);
                if (name == "LmrBase" || name == "LmrDiv") init_lmr();
            }
            else if (name != "PolicyFile" && name != "EvalFile" && name != "DfpsModel")
                std::cout << "info string unknown option " << name << std::endl;
            else if (name == "DfpsModel" && value.find("<exe>") == std::string::npos) {
                dfps::P.modelFile = value;
                if (dfps::MODEL.load(value)) std::cout << "info string loaded DFPS model " << value << std::endl;
                else std::cout << "info string FAILED to load DFPS model " << value << ", using the built-in formula" << std::endl;
            }
        } else if (cmd == "ucinewgame") { lazy_init(); TT.clear(); for (auto& t : THREADS) t->clear(); }
        else if (cmd == "position") {
            std::string tok; is >> tok;
            std::string fen;
            if (tok == "startpos") { fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; is >> tok; }
            else if (tok == "fen") { while (is >> tok && tok != "moves") fen += tok + " "; }
            ROOT.set_fen(fen);
            GAME_KEYS = {ROOT.key};
            if (tok == "moves")
                while (is >> tok) {
                    Move m = parse_move(ROOT, tok);
                    if (!m) break;
                    ROOT.make(m);
                    GAME_KEYS.push_back(ROOT.key);
                }
        } else if (cmd == "go") {
            lazy_init();
            if (MAIN_THREAD.joinable()) { STOP = true; MAIN_THREAD.join(); }
            LIM = Limits(); LIM.start = now_ms();
            int64_t wt = -1, bt = -1, wi = 0, bi = 0, mtg = 0, mt = -1;
            std::string tok;
            while (is >> tok) {
                if (tok == "wtime") is >> wt; else if (tok == "btime") is >> bt;
                else if (tok == "winc") is >> wi; else if (tok == "binc") is >> bi;
                else if (tok == "movestogo") is >> mtg; else if (tok == "movetime") is >> mt;
                else if (tok == "depth") { is >> LIM.depth; LIM.depth = std::clamp(LIM.depth, 1, MAXPLY - 1); }
                else if (tok == "nodes") is >> LIM.nodes;
                else if (tok == "infinite") LIM.infinite = true;
            }
            int64_t tm = ROOT.stm == WHITE ? wt : bt, inc = ROOT.stm == WHITE ? wi : bi;
            if (mt >= 0) { LIM.timed = true; LIM.hard = std::max<int64_t>(1, mt - OPT.moveOverhead); LIM.soft = LIM.hard * 100; }
            else if (tm >= 0) {
                LIM.timed = true;
                int64_t avail = std::max<int64_t>(1, tm - OPT.moveOverhead);
                int64_t soft = mtg > 0 ? avail / std::min<int64_t>(mtg + 1, 40) : avail / 22 + inc * 3 / 4;
                soft = std::min(soft, avail * 2 / 5);
                LIM.hard = std::min(avail * 3 / 4, soft * 4);
                if (mtg <= 0 && inc == 0) {
                    soft     = avail / 30;
                    LIM.hard = std::max<int64_t>(1, std::min(avail / 16, soft * 3));
                }
                LIM.soft = soft * 6 / 10;
            }
            MAIN_THREAD = std::thread(start_search);
        } else if (cmd == "stop") { STOP = true; if (MAIN_THREAD.joinable()) MAIN_THREAD.join(); }
        else if (cmd == "quit") { STOP = true; if (MAIN_THREAD.joinable()) MAIN_THREAD.join(); break; }
        else if (cmd == "perft") {
            int d = 1; is >> d; int64_t t0 = now_ms();
            U64 n = perft(ROOT, d);
            std::cout << "perft " << d << " " << n << " time " << now_ms() - t0 << std::endl;
        } else if (cmd == "eval") {
            lazy_init();
            Thread& t = *THREADS[0]; t.pos[0] = ROOT; t.acc[0].computed = false;
            int u = 0; int v = raw_eval(t, 0, &u);
            std::cout << "eval " << v << " (stm) unc " << u << std::endl;
        } else if (cmd == "d") std::cout << ROOT.fen() << std::endl;
        else if (cmd == "bench") { lazy_init(); int d = 12; is >> d; bench(d); }
        else if (cmd == "tunelist") {  // name default min max, one per line
            for (const TuneParam& tq : TUNES) std::cout << tq.name << " " << tq.def << " " << tq.lo << " " << tq.hi << '\n';
            std::cout << "tunelist done" << std::endl;
        }
        else if (cmd == "dfpsgen") {  // dfpsgen <in.epd> <out.csv> <maxDepth>
            lazy_init();
            std::string in, out; int d = 16; is >> in >> out >> d;
            dfps::gen(in, out, std::clamp(d, 2, 60));
        }
        else if (cmd == "dfpscal") {  // dfpscal <in.epd> <out.csv> <depth>
            lazy_init();
            std::string in, out; int d = 10; is >> in >> out >> d;
            dfps::cal(in, out, std::clamp(d, 2, 40));
        }
        else if (cmd == "selfplay") {  // selfplay <out.bin> <games> <softNodes> <maxDepth> [randomPlies] [nonet]
            lazy_init();
            std::string out, tok; long long games = 1000; uint64_t sn = 5000; int md = 12, rp = 8;
            is >> out >> games >> sn >> md;
            while (is >> tok) {
                if (tok == "nonet") { nnue::NET = nullptr; std::cout << "info string selfplay without a net (material only)" << std::endl; }
                else rp = std::clamp(std::atoi(tok.c_str()), 0, 40);
            }
            selfplay::run(out, games, sn, std::clamp(md, 1, 60), rp);
        }
    }
    STOP = true;  // input closed without "quit": finish a running search before leaving
    if (MAIN_THREAD.joinable()) MAIN_THREAD.join();
    return 0;
}
