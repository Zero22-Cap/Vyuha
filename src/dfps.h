// Vyuha - Decision-Focused Probe Search (DFPS). Experimental, selected with "SearchMode DFPS".
//
// A small best-first tree is kept near the root. Its leaves are not evaluated by the net directly
// but by alpha-beta probes of a recorded depth, so traps a few plies deep are still seen. Each leaf
// holds a belief (mean, sigma) about its true value; sigma comes from an error model of how much a
// depth-d score still moves with deeper search. Beliefs are backed up with Clark's approximation
// for the maximum of Gaussians. A knowledge-gradient controller picks the leaf whose refinement is
// most likely to change the decision at the root and either deepens its probe or expands it into
// children. Proven intervals (checkmate, stalemate, draws by rule) propagate exactly; mate and
// tablebase scores from probes are treated as certain values.
//
// Included by engine.cpp after the alpha-beta search; uses its Thread, search(), TT and limits.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>

namespace dfps {

struct Params {
    int mode = 0;               // 0 = alpha-beta (iterate), 1 = DFPS
    int initDepth = 6;          // probe depth for the root's children
    int step = 1;               // plies added when a probe is deepened (1 measured best)
    int expandDepth = 8;        // a leaf probed at least this deep (and refined before) is expanded
    int minChildDepth = 4;      // smallest probe depth for a new child
    int window = 150;           // half width of probe windows, internal score units
    int maxTree = 100000;       // tree nodes
    int rho = 50;               // assumed correlation between siblings, percent (Clark backup)
    int maxTreePly = 1;         // tree depth limit (1 = root moves only; deeper measured worse)
    int boundShift = 30;        // a one-sided probe result is placed this far beyond its bound
    int qscale = 150;           // beliefs in expected-score space q = 1/(1+exp(-s/qscale)); 0 = centipawns
    int debug = 0;              // print the root children after every search
    int useBias = 0;            // apply the model's depth bias to probe scores (measured worse, off)
    int retest = 8;             // a refuted move is tested again once the best value moved this much
    int testCost = 4;           // a null-window test is costed at the best move's last probe / testCost
    int seed = 1;               // root children start from the root probe's own results (no probes)
    int ply1 = 1;               // probes run one ply below their parent (see run_probe)
    int verify = 0;             // re-test reduced refutations within this margin at full depth (0 = off)
    int tm = 1;                 // time rule: 1 = alpha-beta's (per round), 0 = p_wrong thresholds
    int skipP = 0;              // skip re-testing a move below this chance (1/10000) of beating the best (0 = off)
    int skipMax = 2;            // ... for at most this many plies of the best move's depth
    int strict = 0;             // 1: the best move is deepened only after every sibling is known below it (complete rounds)
    int lmr = 1;                // siblings of the best move are tested at reduced depth by their rank (as LMR)
    int safe = 1;               // when stopped mid-round, play the last completed round's best move
    std::string modelFile;
} P;

// ---------------------------------------------------------------- error model
// sigma(d, v, instab): spread of (deep score - depth-d score). A fitted table is loaded from a file
// written by tools/fit_dfps.py; without it a placeholder formula is used (not yet measured).
struct Model {
    bool loaded = false, q = false;  // q: the table holds sigmas of the expected score (scale K)
    int K = 150;
    float tab[64][4][7] = {};
    float bias[64][4][7] = {};  // mean of (deep - depth-d): shallow scores are too extreme
    static int ib(int instab) { return instab < 0 ? 1 : instab < 10 ? 0 : instab < 30 ? 1 : instab < 80 ? 2 : 3; }
    // signed score buckets: <-800, <-300, <-100, <100, <300, <800, more
    static int vb(int v) { return v < -800 ? 0 : v < -300 ? 1 : v < -100 ? 2 : v < 100 ? 3 : v < 300 ? 4 : v < 800 ? 5 : 6; }
    // sigma and bias from the table (in its own space, isq = q) or the placeholder (centipawns, no bias)
    double raw(int d, int v, int instab, bool& isq, double* b = nullptr) const {
        d = std::clamp(d, 1, 63);
        if (loaded) {
            float s = tab[d][ib(instab)][vb(v)];
            if (s > 0) { isq = q; if (b) *b = bias[d][ib(instab)][vb(v)]; return s; }
        }
        isq = false;
        if (b) *b = 0;
        double s = 400.0 / (d + 2) + 8.0 + 0.5 * (instab < 0 ? 20 : instab);
        return s * (1.0 + std::abs(v) / 600.0);
    }
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::memset(tab, 0, sizeof tab);
        std::memset(bias, 0, sizeof bias);
        std::string line; int n = 0;
        q = false;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.rfind("space", 0) == 0) {  // "space q <K>" or "space cp"
                std::istringstream hs(line); std::string w, sp; hs >> w >> sp;
                q = sp == "q";
                if (q && !(hs >> K)) K = 150;
                continue;
            }
            std::istringstream is(line);
            int d, i, v; double s, cnt = 0, b = 0;
            if (!(is >> d >> i >> v >> s) || d < 1 || d > 63 || i < 0 || i > 3 || v < 0 || v > 6 || s <= 0) continue;
            if (is >> cnt) is >> b;  // optional count and bias columns
            tab[d][i][v] = float(s); bias[d][i][v] = float(b); n++;
        }
        loaded = n > 0;
        if (loaded && q) P.qscale = K;  // run beliefs in the model's own scale
        return loaded;
    }
} MODEL;

inline double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
inline double phi(double z) { return 0.3989422804014327 * std::exp(-0.5 * z * z); }
// E[max(X - t, 0)] for X ~ N(m, s^2)
inline double ei(double m, double s, double t) {
    if (s <= 1e-9) return std::max(0.0, m - t);
    double z = (m - t) / s;
    return s * phi(z) + (m - t) * Phi(z);
}

// ---------------------------------------------------------------- probes
struct Job {
    int node = -1;
    Board pos;
    std::vector<U64> game, path;  // game history up to the root, then the tree path down to pos
    int depth = 1, alpha = -INF, beta = INF;
    int startDepth = 1;              // first iteration (a node probed before has its earlier depths in the TT)
    int guess = 0; bool haveGuess = false;  // earlier score: aspiration from the first iteration
    int score = 0, bound = B_NONE, pvLen = 0;
    Move pv[16];
    uint64_t nodes = 0;
    uint64_t softNodes = 0;  // optional: stop deepening once a completed depth has used this many
    bool aborted = false;
    bool wantRoot = false;            // keep every root move's result of the last completed depth
    Move fhMove = 0;                  // a move that failed high in an unfinished iteration
    std::vector<RootRec> rootRec;
    int cat = 0;                      // debug: kind of probe (see Search::catNodes)
    bool hasParent = false;           // search pos one ply below its parent (as alpha-beta searches a root move)
    Board parent; Move move = 0;
};

// Alpha-beta probe of j.pos to j.depth inside [j.alpha, j.beta], with iterative deepening and
// aspiration. onDepth (optional) is called after every completed depth. With a parent, pos is
// searched at ply 1 below it, so it gets the treatment of a root move's child in alpha-beta
// (pruning, continuation history and counter move from the move into it, singular extensions).
void run_probe(Thread& t, Job& j, const std::function<void(int, int, int, Move)>& onDepth = nullptr) {
    if (STOP) { j.aborted = true; return; }
    t.probe = true;
    for (auto& s : t.ssBuf) s = Stack();
    t.ss = t.ssBuf + 4;
    const int P1 = j.hasParent ? 1 : 0;
    if (P1) {
        t.pos[0] = j.parent; t.pos[1] = j.pos;
        t.acc[1].computed = false;
        t.ss[0].move = j.move;
        t.ss[0].cont = t.cont[j.parent.sq[from_sq(j.move)] * 64 + to_sq(j.move)];
    } else t.pos[0] = j.pos;
    t.acc[0].computed = false;
    int np = int(j.path.size());
    int ng = int(std::min<size_t>(j.game.size(), size_t(std::max(0, 1000 - np))));
    int k = 0;
    for (int i = 0; i < ng; i++) t.hist[k++] = j.game[j.game.size() - ng + i];
    for (int i = 0; i < np; i++) t.hist[k++] = j.path[i];
    if (k == 0) t.hist[k++] = j.pos.key;
    if (P1 && k < 2) { t.hist[1] = t.hist[0]; t.hist[0] = j.parent.key; k = 2; }
    t.base = k - 1 - P1;  // hist[base + ply] is the position at ply
    t.repBase = ng > 0 ? ng - 1 : t.base;  // repetitions after the game root count at once
    t.bestMove = 0;
    uint64_t n0 = t.nodes;
    const int A = j.alpha, B = j.beta;
    int score = j.guess; bool have = j.haveGuess, done = false;
    std::vector<RootRec> rec;
    t.rootRec = j.wantRoot ? &rec : nullptr;
    const int target = j.depth;
    // a ply-1 search returns mate scores one ply further from the parent: shift them back
    auto rel = [&](int v) { return !P1 ? v : v >= TB_BOUND ? v + 1 : v <= -TB_BOUND ? v - 1 : v; };
    const bool cut = B - A == 1;
    for (int d = std::clamp(j.startDepth, 1, target); d <= target; d++) {
        t.rootDepth = d + P1; t.seldepth = 0;
        int delta = tp::AspDelta, alpha = A, beta = B, v = 0;
        // aspiration only inside the outer window, and never for a null-window test
        if (d >= 4 && have && B - A > 1) {
            alpha = std::max(A, score - delta); beta = std::min(B, score + delta);
            if (alpha >= beta) { alpha = A; beta = B; }
        }
        while (true) {
            rec.clear();
            v = rel(search(t, P1, alpha, beta, d, P1 && cut));
            if (STOP) break;
            if (v <= alpha && alpha > A) { beta = (alpha + beta) / 2; alpha = std::max(A, v - delta); }
            else if (v >= beta && beta < B) {
                beta = std::min(B, v + delta);
                if (t.pvLen[P1] > P1) j.fhMove = t.bestMove = t.pv[P1][P1];  // played if the re-search is stopped
            }
            else break;
            delta += delta / 2;
            if (delta > 1500) { alpha = A; beta = B; }
        }
        if (STOP) break;
        score = v; have = done = true; j.fhMove = 0;
        j.score = v;
        j.depth = d;  // the depth actually completed (a soft node limit may end the probe early)
        j.bound = v <= A ? B_UPPER : v >= B ? B_LOWER : B_EXACT;
        if (t.pvLen[P1] > P1) {
            t.bestMove = t.pv[P1][P1];
            j.pvLen = std::min(16, t.pvLen[P1] - P1);
            for (int i = 0; i < j.pvLen; i++) j.pv[i] = t.pv[P1][P1 + i];
        }
        if (j.wantRoot) j.rootRec = rec;
        if (onDepth) onDepth(d, v, j.bound, t.bestMove);
        if (j.softNodes && t.nodes - n0 >= j.softNodes) break;
    }
    t.probe = false;
    t.rootRec = nullptr;
    j.nodes = t.nodes - n0;
    if (!done) j.aborted = true;  // a stop keeps the depths completed before it
}

// Worker pool: every worker owns one search Thread; jobs and results go through two queues.
struct Pool {
    std::mutex mx;
    std::condition_variable cvJob, cvDone;
    std::deque<Job*> todo, done;
    std::vector<std::thread> workers;
    bool quit = false;
    int inFlight = 0;
    void start(int n) {
        quit = false;
        for (int i = 0; i < n; i++) workers.emplace_back([this, i]() { loop(*THREADS[i]); });
    }
    void loop(Thread& t) {
        while (true) {
            Job* j;
            {
                std::unique_lock<std::mutex> lk(mx);
                cvJob.wait(lk, [&]() { return quit || !todo.empty(); });
                if (todo.empty()) return;
                j = todo.front(); todo.pop_front();
            }
            run_probe(t, *j);
            { std::lock_guard<std::mutex> lk(mx); done.push_back(j); }
            cvDone.notify_one();
        }
    }
    void push(Job* j) {
        { std::lock_guard<std::mutex> lk(mx); todo.push_back(j); inFlight++; }
        cvJob.notify_one();
    }
    std::vector<Job*> wait(int ms) {
        std::unique_lock<std::mutex> lk(mx);
        cvDone.wait_for(lk, std::chrono::milliseconds(ms), [&]() { return !done.empty(); });
        std::vector<Job*> r(done.begin(), done.end());
        done.clear();
        inFlight -= int(r.size());
        return r;
    }
    // stop the workers; returns the jobs finished since the last wait (a stopped probe keeps the
    // depths it completed)
    std::vector<Job*> finish() {
        { std::lock_guard<std::mutex> lk(mx); quit = true; }
        cvJob.notify_all();
        for (auto& w : workers) w.join();
        workers.clear();
        for (Job* j : todo) delete j;
        todo.clear();
        std::vector<Job*> r(done.begin(), done.end());
        done.clear();
        inFlight = 0;
        return r;
    }
};

// ---------------------------------------------------------------- tree
struct Node {
    Board pos;
    int parent = -1, first = -1, n = 0, pending = 0;
    Move move = 0;
    int ply = 0;
    int depth = 0, score = 0, prev = NONE, bound = B_NONE, refines = 0, fails = 0;  // last probe (node's side to move)
    int testD = -1, testT = 0;  // last null-window test: depth and threshold (parent's view)
    Move pv[16]; int pvLen = 0;
    uint64_t lastNodes = 0, prevNodes = 0;  // nodes of the last two probes (growth estimate)
    uint64_t subNodes = 0;                  // nodes spent in this subtree so far
    int lo = -INF, hi = INF;  // proven interval for the side to move
    double mu = 0, sd = 0;  // belief: mean, spread used to choose refinements
    double bsd = 0;         // spread used when backing up into the parent (tighter for one-sided results)
    bool busy = false, expanded = false, terminal = false;
    Move fhMove = 0;        // root: the move that failed high in the probe's unfinished iteration
};

struct Search {
    std::vector<Node> T;
    Pool pool;
    int nWorkers = 1;
    std::vector<RootRec> rootRec;  // every root move's result from the root probe's last depth
    Move safeMove = 0; int safeDepth = 0;  // best move of the last completed round and its depth
    uint64_t nProbes = 0, rootProbeNodes = 0;  // debug statistics
    // nodes and probes by kind: 0 root, 1 best deepened, 2 best re-searched, 3 sibling test, 4 reduced
    // sibling test, 5 sibling window search, 6 child probe at expansion
    uint64_t catNodes[8] = {}, catCount[8] = {};
    int tmStable = 0, tmPrevScore = NONE, tmDepth = 0; Move tmPrev = 0;  // time rule state (P.tm == 1)

    // mate and tablebase scores from a probe are relative to the probe root; make them root-relative
    static int adj(int s, int ply) { return s >= TB_BOUND ? s - ply : s <= -TB_BOUND ? s + ply : s; }

    std::vector<U64> path_keys(int n) const {
        std::vector<U64> k;
        for (int x = n; x > 0; x = T[x].parent) k.push_back(T[x].pos.key);
        std::reverse(k.begin(), k.end());
        return k;
    }

    bool settled(const Node& x) const { return x.lo == x.hi; }

    // ---- value space: beliefs live in expected score (qscale > 0) or in centipawns
    bool qs() const { return P.qscale > 0; }
    double toV(double s) const {
        if (!qs()) return s;
        if (s >= TB_BOUND) return 1;
        if (s <= -TB_BOUND) return 0;
        return 1 / (1 + std::exp(-s / P.qscale));
    }
    double toS(double v) const {
        if (!qs()) return v;
        v = std::clamp(v, 1e-6, 1 - 1e-6);
        return P.qscale * std::log(v / (1 - v));
    }
    double neg(double v) const { return qs() ? 1 - v : -v; }  // the parent's view of a child's value
    double sigmaV(int d, int s, int instab, double* biasOut = nullptr) const {
        if (biasOut) *biasOut = 0;
        if (std::abs(s) >= TB_BOUND) return 0;
        bool isq; double b = 0, r = MODEL.raw(d, s, instab, isq, &b);
        if (isq == qs()) { if (biasOut) *biasOut = P.useBias ? b : 0; return r; }
        double K = isq ? MODEL.K : P.qscale, p = 1 / (1 + std::exp(-s / K)), slope = p * (1 - p) / K;  // dq/ds
        double f = isq ? 1 / std::max(slope, 1e-6) : slope;
        if (biasOut) *biasOut = P.useBias ? b * f : 0;
        return r * f;
    }
    // probe window half width in score units for a belief (mu, sd)
    int wcp(double mu, double sd) const {
        double w = sd;
        if (qs()) { double p = std::clamp(mu, 0.02, 0.98); w = sd * P.qscale / (p * (1 - p)); }
        return int(std::clamp(2 * w, double(P.window), 3000.0));
    }

    // draws by rule at a tree node (not the root): 50 moves, bare material, repetition
    bool rule_draw(int n) const {
        const Board& b = T[n].pos;
        if (b.hm >= 100) return true;
        if (!(b.byPt[PAWN] | b.byPt[ROOK] | b.byPt[QUEEN]) && popcnt(b.occ() ^ b.byPt[KING]) <= 1) return true;
        std::vector<U64> seq(GAME_KEYS.begin(), GAME_KEYS.end());
        int gameRoot = int(seq.size()) - 1;
        std::vector<U64> p = path_keys(n);
        seq.insert(seq.end(), p.begin(), p.end());
        int idx = int(seq.size()) - 1, lo = std::max(0, idx - b.hm), earlier = 0;
        for (int j = idx - 4; j >= lo; j -= 2)
            if (seq[j] == b.key) {
                if (j > gameRoot) return true;
                if (++earlier >= 2) return true;
            }
        return false;
    }

    void set_exact(Node& x, int v) { x.terminal = true; x.lo = x.hi = v; x.mu = toV(v); x.sd = x.bsd = 0; x.score = v; x.bound = B_EXACT; }

    // A probe that failed outside its window only bounds the value (fail-soft); the belief is
    // placed boundShift beyond the bound. For choosing refinements it keeps the full model spread:
    // a bound measured against an old threshold says little once the threshold has moved. For the
    // backup into the parent a lower bound (the parent's move is capped) uses a tight spread, so
    // many refuted siblings do not inflate the parent's maximum.
    void leaf_belief(Node& x) {
        int instab = x.prev == NONE ? -1 : std::abs(x.score - x.prev);
        double bias = 0, sd = sigmaV(x.depth, x.score, instab, &bias), sh = P.boundShift;
        x.mu = toV(x.score); x.sd = x.bsd = sd;
        if (x.bound == B_EXACT) { if (std::abs(x.score) >= TB_BOUND) x.sd = x.bsd = 0; }
        else if (std::abs(x.score) >= TB_BOUND) x.sd = x.bsd = qs() ? 0.25 : std::max(sd, double(P.window));
        else if (x.bound == B_LOWER) {
            x.mu = toV(x.score + sh);
            x.bsd = std::abs(x.mu - toV(x.score));
            x.sd = std::max(sd, x.bsd);
        } else { x.mu = toV(x.score - sh); x.sd = x.bsd = std::max(sd, std::abs(toV(x.score) - x.mu)); }
        if (std::abs(x.score) < TB_BOUND) x.mu += bias;  // correct the depth bias of the probe score
        x.mu = qs() ? std::clamp(x.mu, 0.0, 1.0) : std::clamp(x.mu, double(-MATE), double(MATE));
        clamp_interval(x);
    }

    void clamp_interval(Node& x) {
        if (x.lo > -INF) x.mu = std::max(x.mu, toV(x.lo));
        if (x.hi < INF) x.mu = std::min(x.mu, toV(x.hi));
        if (settled(x)) { x.mu = toV(x.lo); x.sd = 0; }
    }

    // belief of an expanded node from its children (Clark 1961, pairwise maximum of Gaussians)
    void backup(int n) {
        Node& x = T[n];
        std::vector<std::pair<double, double>> c;
        int lo = -INF, hi = -INF;
        for (int i = x.first; i < x.first + x.n; i++) {
            const Node& y = T[i];
            c.push_back({neg(y.mu), y.bsd});
            lo = std::max(lo, y.hi >= INF ? -INF : -y.hi);
            hi = std::max(hi, y.lo <= -INF ? INF : -y.lo);
        }
        if (c.empty()) return;
        std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first; });
        double M = c[0].first, V = c[0].second * c[0].second, r = P.rho / 100.0;
        for (size_t i = 1; i < c.size(); i++) {
            double m = c[i].first, s = c[i].second;
            double a2 = V + s * s - 2 * r * std::sqrt(V) * s;
            if (a2 <= 1e-9) { if (m > M) { M = m; V = s * s; } continue; }
            double a = std::sqrt(a2), al = (M - m) / a, F = Phi(al), f = phi(al);
            double M2 = M * F + m * (1 - F) + a * f;
            double E2 = (M * M + V) * F + (m * m + s * s) * (1 - F) + (M + m) * a * f;
            V = std::max(0.0, E2 - M2 * M2);
            M = M2;
        }
        x.mu = qs() ? std::clamp(M, 0.0, 1.0) : M; x.sd = x.bsd = std::sqrt(V);
        x.lo = lo; x.hi = hi;
        clamp_interval(x);
    }

    void propagate(int n) {
        for (int p = T[n].parent; p >= 0; p = T[p].parent) {
            if (!T[p].expanded || T[p].pending) break;
            backup(p);
        }
    }

    int cap_depth() const { return LIM.depth; }

    // can this leaf be refined (deepened or expanded) within the depth limit?
    int child_depth(const Node& x) const {
        int d = std::max(P.minChildDepth, x.depth - 1);
        return std::min(d, cap_depth() - x.ply - 1);
    }
    bool can_deepen(const Node& x) const {
        if (x.bound != B_EXACT && x.depth > 0) return x.ply + x.depth <= cap_depth();  // re-search
        return x.ply + x.depth + P.step <= cap_depth();
    }
    bool can_expand(const Node& x) const {
        return x.ply < P.maxTreePly && int(T.size()) + 256 < P.maxTree && child_depth(x) >= 1;
    }
    bool want_expand(const Node& x) const {
        if (x.ply == 0) return true;  // the root is expanded right after its first probe
        if (!can_expand(x)) return false;
        if (!can_deepen(x)) return true;
        return x.depth >= P.expandDepth && x.refines >= 2;
    }

    // expected nodes of the next refinement of this subtree (deepen: growth of the last two probes;
    // re-search: about the last probe; internal node: a fraction of what it has used so far)
    double est_cost(const Node& y) const {
        if (y.expanded) return std::max<double>(1000, y.subNodes / 4.0);
        double last = std::max<double>(1000, y.lastNodes);
        if (y.bound != B_EXACT) return last;
        double g = y.prevNodes > 0 ? std::clamp(double(y.lastNodes) / y.prevNodes, 1.5, 8.0) : 3.0;
        return last * g;
    }

    // expected sigma after one refinement step
    double next_sd(const Node& x) const {
        if (!x.expanded) {
            int instab = x.prev == NONE ? -1 : std::abs(x.score - x.prev);
            return std::min(x.sd, sigmaV(x.depth + P.step, x.score, instab));
        }
        return x.sd * 0.8;
    }

    // The best child of p: highest value among children with an exact (or terminal) value; only
    // if none has one, among all probed children. second: the best of the others.
    int best_child(int p, int* second = nullptr) const {
        int a = -1, b = -1; bool aEx = false;
        for (int i = T[p].first; i < T[p].first + T[p].n; i++) {
            if (!T[i].terminal && T[i].depth == 0) continue;  // not probed yet
            bool ex = T[i].terminal || T[i].bound == B_EXACT;
            double v = neg(T[i].mu);
            if (a < 0 || (ex && !aEx) || (ex == aEx && v > neg(T[a].mu))) { if (a >= 0) b = (b < 0 || neg(T[a].mu) > neg(T[b].mu)) ? a : b; a = i; aEx = ex; }
            else if (b < 0 || v > neg(T[b].mu)) b = i;
        }
        if (second) *second = b;
        return a;
    }

    // A sibling of the best move needs no test at depth D against the best's value t (parent's
    // view) when it already has an exact value at that depth, or an upper bound at most t (a
    // fail-soft bound from a lower threshold still refutes it). testD is the depth a reduced test
    // stood for.
    // A refuted move whose test was reduced and whose bound is within verify of the best value is
    // tested again at full depth (measured with dfpscal: such moves often beat the best when
    // searched fully; the chance falls with the margin).
    bool verify_due(const Node& y, int D, int t) const {
        return P.verify > 0 && y.depth > 0 && y.depth < D && y.bound == B_LOWER && t + y.score < P.verify;
    }
    bool known_below(const Node& y, int D, int t) const {
        if (y.depth <= 0 || verify_due(y, D, t)) return false;
        int d = std::max(y.depth, y.testD);
        if (d >= D) {
            if (y.bound == B_EXACT) return true;
            if (y.bound == B_LOWER && -y.score <= t + P.retest) return true;
        }
        // decision-focused skip: a move whose value or upper bound from a recent depth leaves it a
        // negligible chance (error model) of beating t is not tested again yet
        if (P.skipP > 0 && d >= D - P.skipMax && (y.bound == B_EXACT || y.bound == B_LOWER)) {
            double up = toV(-y.score), sd = std::max(y.sd, 1e-9);
            if (1 - Phi((toV(t) - up) / sd) < P.skipP / 10000.0) return true;
        }
        return false;
    }

    // pick a leaf to refine: descend by the knowledge gradient, trying children in order.
    // Only the best child is deepened; another child is tested at the best child's depth against
    // its value, and not again until that depth or that value moves (alpha-beta's root iteration,
    // with the knowledge gradient deciding which tests are worth their nodes).
    int pick(int n) {
        Node& x = T[n];
        if (x.terminal || settled(x) || x.busy || x.pending) return -1;
        if (!x.expanded) return (can_deepen(x) || can_expand(x)) ? n : -1;
        int b = -1, a = best_child(n, &b);
        if (a < 0) return -1;
        double va = neg(T[a].mu);
        int D = T[a].depth, t = int(std::lround(toS(va)));
        std::vector<std::pair<double, int>> order;
        for (int i = x.first; i < x.first + x.n; i++) {
            const Node& y = T[i];
            if (y.terminal || settled(y)) continue;
            double kg, cost;
            if (i == a) {
                if (P.strict && !round_complete_at(n, a, D, t)) continue;
                double ns = next_sd(y), red = std::sqrt(std::max(0.0, y.sd * y.sd - ns * ns));
                kg = b < 0 ? 0 : ei(-va, red, -neg(T[b].mu));  // E[max(v_b - X_a, 0)] in x's view
                cost = est_cost(y);
            } else {
                if (known_below(y, D, t)) continue;
                kg = ei(neg(y.mu), y.sd, va);  // chance-weighted gain if this move turns out better
                cost = std::max<double>(1000, T[a].lastNodes / double(P.testCost));
            }
            if (n == 0) kg /= cost;  // value per node at the root
            order.push_back({kg, i});
        }
        // ties (e.g. every gain ~0) go to the current best move, then to higher values
        for (auto& o : order) if (o.first < 1e-15) o.first = 0;
        std::sort(order.begin(), order.end(), [&](auto& p, auto& q) {
            if (p.first != q.first) return p.first > q.first;
            if ((p.second == a) != (q.second == a)) return p.second == a;
            return neg(T[p.second].mu) > neg(T[q.second].mu);
        });
        for (auto& o : order) {
            int r = pick(o.second);
            if (r >= 0) return r;
        }
        return -1;
    }

    void push_probe(int n, int depth, int alpha, int beta, int cat = 0) {
        Node& x = T[n];
        Job* j = new Job;
        j->cat = cat;
        j->node = n; j->pos = x.pos; j->game = GAME_KEYS; j->path = path_keys(n);
        j->depth = depth; j->alpha = std::max(-INF, alpha); j->beta = std::min(INF, beta);
        // a node searched before continues from its last depth: the TT holds the earlier ones
        if (x.depth > 0) {
            j->startDepth = depth;  // no shallower iterations: the TT has them (as a ply-1 search in alpha-beta)
            if (x.bound == B_EXACT && std::abs(x.score) < TB_BOUND) { j->guess = x.score; j->haveGuess = true; }
        }
        j->wantRoot = n == 0 && !x.expanded;
        if (P.ply1 && x.parent >= 0) { j->hasParent = true; j->parent = T[x.parent].pos; j->move = x.move; }
        x.busy = true;
        nProbes++;
        pool.push(j);
    }

    // fill a root child from the root probe's own search of that move (no probe of its own)
    bool seed(int i) {
        Node& c = T[i];
        for (const RootRec& r : rootRec) {
            if (r.move != c.move || r.depth < 1) continue;
            c.depth = r.depth;
            c.score = -r.score;  // root-relative already
            c.bound = r.bound == B_EXACT ? B_EXACT : r.bound == B_UPPER ? B_LOWER : B_UPPER;
            c.lastNodes = c.subNodes = r.nodes;
            c.refines = 1;
            c.testD = T[0].depth - 1; c.testT = r.alpha;
            if (T[0].pvLen > 0 && T[0].pv[0] == c.move) {
                c.pvLen = T[0].pvLen - 1;
                for (int k = 0; k < c.pvLen; k++) c.pv[k] = T[0].pv[k + 1];
            }
            leaf_belief(c);
            return true;
        }
        return false;
    }

    // Refine a leaf: a one-sided result is first re-searched at the same depth (as PVS re-searches
    // a null-window fail high); an exact result is deepened by P.step.
    void deepen(int n) {
        Node& x = T[n];
        // A move that is not its parent's current best is only tested, as in PVS: a null window
        // at the best sibling's value, one step deeper (or at the same depth after a fail high).
        // It gets a real window only once it is the best.
        // Siblings are tested at a depth reduced by their rank below the best (alpha-beta's late move
        // reductions at the root); a reduced test that fails high is repeated at full depth before
        // it gets a real window.
        int par = x.parent;
        if (par >= 0 && std::abs(x.score) < TB_BOUND - 1000) {
            int a = best_child(par);
            bool failHigh = x.bound == B_UPPER && x.depth > 0;
            if (a >= 0 && a != n && (!failHigh || x.depth < T[a].depth)) {
                double va = neg(T[a].mu);
                if (std::abs(toS(va)) < TB_BOUND - 1000) {
                    int t = int(std::lround(toS(va)));  // parent's view
                    int d = std::max(1, T[a].depth), r = 0;
                    if (P.lmr && !failHigh && x.depth > 0 && d >= 2) {
                        int mc = 2;  // move number as in alpha-beta: the best is 1, then by value
                        for (int i = T[par].first; i < T[par].first + T[par].n; i++)
                            if (i != a && i != n && T[i].depth > 0 && neg(T[i].mu) > neg(x.mu)) mc++;
                        if (mc > 2) {
                            r = LMR[std::min(d + 1, 63)][std::min(mc, 63)];
                            if (T[par].pos.is_noisy(x.move)) r /= 2;
                            if (x.pos.checkers) r--;
                            r = std::clamp(r, 0, d - 1);
                        }
                    }
                    if (verify_due(x, d, t)) r = 0;
                    x.testD = d; x.testT = t;
                    push_probe(n, std::max(1, d - r), -t - 1, -t, r > 0 ? 4 : 3);
                    return;
                }
            }
        }
        int w = wcp(x.mu, x.sd);
        bool full = std::abs(x.score) >= TB_BOUND - 1000 || x.fails >= 2;
        int c = int(std::lround(toS(x.mu)));
        int lo = full ? -INF : c - w, hi = full ? INF : c + w;
        bool research = x.bound != B_EXACT && x.depth > 0;
        int cat = x.parent >= 0 && best_child(x.parent) != n ? 5 : research ? 2 : 1;
        push_probe(n, research ? x.depth : x.depth + P.step, lo, hi, cat);
    }

    void expand(int n) {
        MoveList ml; gen_moves(T[n].pos, ml, true, true);
        std::vector<Move> moves;
        for (int i = 0; i < ml.n; i++) {
            Move m = ml.m[i];
            if (!T[n].pos.legal(m)) continue;
            if (n == 0 && !ROOT_ALLOWED.empty() && std::find(ROOT_ALLOWED.begin(), ROOT_ALLOWED.end(), m) == ROOT_ALLOWED.end()) continue;
            moves.push_back(m);
        }
        if (moves.empty()) {  // only reachable for a leaf that was probed before; mate or stalemate
            set_exact(T[n], T[n].pos.checkers ? -MATE + T[n].ply : 0);
            propagate(n);
            return;
        }
        // the previous best move goes first: it gets an aspiration window, its siblings null windows
        bool havePv = false;
        if (T[n].pvLen > 0) {
            auto it = std::find(moves.begin(), moves.end(), T[n].pv[0]);
            if (it != moves.end()) { std::rotate(moves.begin(), it, it + 1); havePv = true; }
        }
        int first = int(T.size());
        int cd = std::max(1, child_depth(T[n]));
        bool fullWindow = T[n].depth == 0 || std::abs(T[n].score) >= TB_BOUND - 1000;
        double pm = T[n].mu;
        int w = wcp(pm, T[n].sd);
        for (Move m : moves) {
            Node c;
            c.pos = T[n].pos; c.pos.make(m);
            c.parent = n; c.move = m; c.ply = T[n].ply + 1;
            T.push_back(c);
        }
        T[n].first = first; T[n].n = int(moves.size()); T[n].expanded = true; T[n].pending = 0;
        for (int i = first; i < first + T[n].n; i++) {
            MoveList cl; gen_moves(T[i].pos, cl, true, true);
            bool any = false;
            for (int k = 0; k < cl.n && !any; k++) any = T[i].pos.legal(cl.m[k]);
            if (!any) { set_exact(T[i], T[i].pos.checkers ? -MATE + T[i].ply : 0); continue; }
            if (rule_draw(i)) { set_exact(T[i], 0); continue; }
            if (n == 0 && P.seed && seed(i)) continue;
            T[n].pending++;
            int t = int(std::lround(toS(pm)));  // parent's current value; children test against it
            if (fullWindow) push_probe(i, cd, -INF, INF, 6);
            else if (i == first && havePv) push_probe(i, cd, -t - w, -t + w, 6);
            else push_probe(i, cd, -t - 1, -t, 6);  // "is this move better than t?" (Scout test)
        }
        if (!T[n].pending) { backup(n); propagate(n); }
    }

    void integrate(Job* j) {
        Node& x = T[j->node];
        x.busy = false;
        bool fresh = x.depth == 0;
        catNodes[j->cat] += j->nodes; catCount[j->cat]++;
        if (j->aborted) return;
        if (j->wantRoot) { rootRec = std::move(j->rootRec); rootProbeNodes = j->nodes; }
        if (!fresh && j->depth > x.depth) x.prev = x.bound == B_EXACT ? x.score : NONE;
        x.fails = j->bound == B_EXACT ? 0 : (!fresh && j->depth == x.depth ? x.fails + 1 : 0);
        x.depth = j->depth;
        x.score = adj(j->score, x.ply);
        x.bound = j->bound;
        x.pvLen = j->pvLen;
        x.fhMove = j->fhMove;
        for (int i = 0; i < j->pvLen; i++) x.pv[i] = j->pv[i];
        x.prevNodes = x.lastNodes;
        x.lastNodes = j->nodes;
        for (int a = j->node; a >= 0; a = T[a].parent) T[a].subNodes += j->nodes;
        x.refines++;
        if (!x.expanded) leaf_belief(x);
        if (fresh && x.parent >= 0 && T[x.parent].pending > 0) {
            if (--T[x.parent].pending == 0) backup(x.parent);
            else return;
            propagate(x.parent);
            return;
        }
        propagate(j->node);
    }

    // ---------------------------------------------------------------- root reporting
    // the move to play: exact values first (a fail-high not yet re-searched does not count), then
    // the higher value, then the shorter mate / higher score
    int best_root_child() const {
        if (T.empty() || T[0].n == 0) return -1;
        int a = best_child(0);
        if (a < 0) return -1;
        bool aEx = T[a].terminal || T[a].bound == B_EXACT;
        for (int i = T[0].first; i < T[0].first + T[0].n; i++) {
            const Node& y = T[i];
            if (i == a || (!y.terminal && y.depth == 0)) continue;
            bool ex = y.terminal || y.bound == B_EXACT;
            if (ex == aEx && neg(y.mu) == neg(T[a].mu) && -y.score > -T[a].score) a = i;
        }
        return a;
    }

    // probability that some other root move beats the current best (pairwise, independent Gaussians)
    double p_wrong() const {
        int a = best_root_child();
        if (a < 0) return 1;
        double keep = 1;
        const Node& r = T[0];
        for (int i = r.first; i < r.first + r.n; i++) {
            if (i == a || (!T[i].terminal && T[i].depth == 0)) continue;
            double s = std::sqrt(T[i].sd * T[i].sd + T[a].sd * T[a].sd);
            double d = neg(T[i].mu) - neg(T[a].mu);
            keep *= 1 - (s <= 1e-9 ? double(d > 0) : Phi(d / s));
        }
        return 1 - keep;
    }

    bool round_complete_at(int p, int a, int D, int t) const {
        for (int i = T[p].first; i < T[p].first + T[p].n; i++) {
            const Node& y = T[i];
            if (i == a || y.terminal || settled(y)) continue;
            if (!known_below(y, D, t)) return false;
        }
        return true;
    }

    // alpha-beta's completed iteration: the best root move has an exact value and every other move
    // is known to be no better at the best move's depth
    bool round_complete() const {
        int a = best_child(0);
        if (a < 0 || T[0].pending || !(T[a].terminal || T[a].bound == B_EXACT)) return false;
        int t = int(std::lround(toS(neg(T[a].mu)))), D = T[a].depth;
        for (int i = T[0].first; i < T[0].first + T[0].n; i++) {
            const Node& y = T[i];
            if (i == a || y.terminal || settled(y)) continue;
            if (!known_below(y, D, t)) return false;
        }
        return true;
    }

    // the move to play: the current best if the round is complete or it has an exact value from
    // a deeper search than the last completed round; otherwise that round's best move (a move
    // whose new search is unfinished, or that only beat a stale threshold, is not trusted)
    int final_child() const {
        int a = best_root_child();
        if (a < 0) return a;
        // alpha-beta's rule: a move that beat the best move's value at its depth (a fail high not yet
        // re-searched) is played; the highest lower bound if there are several
        if (P.safe >= 2 && (T[a].terminal || T[a].bound == B_EXACT)) {
            int f = -1;
            for (int i = T[0].first; i < T[0].first + T[0].n; i++) {
                const Node& y = T[i];
                if (i == a || y.terminal || y.bound != B_UPPER || y.depth < T[a].depth) continue;
                if (-y.score <= -T[a].score) continue;  // not above the best's value
                if (f < 0 || -y.score > -T[f].score) f = i;
            }
            if (f >= 0) return f;
        }
        if (!P.safe || !safeMove || round_complete()) return a;
        if ((T[a].terminal || T[a].bound == B_EXACT) && T[a].depth > safeDepth) return a;
        for (int i = T[0].first; i < T[0].first + T[0].n; i++)
            if (T[i].move == safeMove) return i;
        return a;
    }

    // the proven intervals alone already decide the root move
    bool root_proven() const {
        int a = best_root_child();
        if (a < 0 || T[0].pending) return false;
        double low = T[a].hi >= INF ? -INF : -T[a].hi;
        const Node& r = T[0];
        for (int i = r.first; i < r.first + r.n; i++) {
            if (i == a) continue;
            double up = std::min(T[i].lo <= -INF ? double(INF) : double(-T[i].lo), double(MATE - 1));  // nothing beats mate in 1
            if (up > low) return false;
        }
        return true;
    }

    void info(int64_t el) {
        int a = best_root_child();
        if (a < 0) return;
        uint64_t tot = 0, tbh = 0; int seld = 0;
        for (auto& th : THREADS) { tot += th->nodes; tbh += th->tbHits; seld = std::max(seld, th->seldepth); }
        std::string pv; int n = a, depth = 0;
        while (true) {
            pv += move_str(T[n].move) + " ";
            if (!T[n].expanded || T[n].pending) break;
            int b = -1;
            for (int i = T[n].first; i < T[n].first + T[n].n; i++)
                if (b < 0 || neg(T[i].mu) > neg(T[b].mu)) b = i;
            if (b < 0) break;
            n = b;
        }
        depth = T[n].ply + T[n].depth;
        for (int i = 0; i < T[n].pvLen; i++) pv += move_str(T[n].pv[i]) + " ";
        // decisive values have no centipawn size in q space: show the PV leaf's own score then
        double va = neg(T[a].mu);
        int score = int(std::lround(toS(va)));
        if (!qs()) score = int(std::lround(va));
        else if (va > 0.9995 || va < 0.0005) score = T[n].ply % 2 ? -T[n].score : T[n].score;
        if (ROOT_TB && std::abs(score) < MATE_BOUND) score = ROOT_TB == 1 ? TB_WIN_SCORE : ROOT_TB == -1 ? -TB_WIN_SCORE : 0;
        std::cout << "info depth " << depth << " seldepth " << T[n].ply + seld << " multipv 1 score " << score_str(score)
                  << " nodes " << tot << " nps " << tot * 1000 / std::max<int64_t>(1, el) << " hashfull " << TT.hashfull()
                  << " tbhits " << tbh << " time " << el << " pv " << pv << std::endl;
        std::cout << "info string dfps tree " << T.size() << " pwrong " << int(p_wrong() * 1000) << "/1000" << std::endl;
    }

    Move run() {
        nWorkers = int(THREADS.size());
        for (auto& th : THREADS) { th->nodes = 0; th->tbHits = 0; th->seldepth = 0; th->rootDepth = 0; }
        T.clear();
        T.reserve(size_t(P.maxTree) + 512);
        rootRec.clear();
        safeMove = 0; safeDepth = 0; nProbes = rootProbeNodes = 0;
        std::fill(std::begin(catNodes), std::end(catNodes), 0); std::fill(std::begin(catCount), std::end(catCount), 0);
        tmStable = 0; tmPrev = 0; tmPrevScore = NONE; tmDepth = 0;
        Node root; root.pos = ROOT; T.push_back(root);
        pool.start(nWorkers);
        // probe the root first: its value and best move set the windows for the root's children
        push_probe(0, std::max(1, std::min(P.initDepth, cap_depth())), -INF, INF);
        int64_t lastInfo = now_ms();
        while (true) {
            while (pool.inFlight < nWorkers && !STOP) {
                int n = pick(0);
                if (n < 0) break;
                if (want_expand(T[n])) expand(n); else deepen(n);
            }
            if (pool.inFlight == 0) {
                if (!LIM.infinite || STOP) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            for (Job* j : pool.wait(5)) { integrate(j); delete j; }
            bool timeUp = false;
            if (round_complete()) {
                int a = best_child(0);
                bool newRound = T[a].depth > safeDepth;
                if (P.debug && newRound) {
                    uint64_t tot = 0; for (auto& th : THREADS) tot += th->nodes;
                    std::cout << "info string round " << T[a].depth + 1 << " nodes " << tot << std::endl;
                }
                safeMove = T[a].move; safeDepth = T[a].depth;
            }
            // alpha-beta's time rule each time the best move has an exact value one ply deeper (rounds
            // need not complete: the best move is often deepened before every sibling is re-tested):
            // its share of the nodes, its stability and a falling score set the soft limit
            if (P.tm == 1 && LIM.timed && !LIM.infinite) {
                int a = best_child(0);
                if (a >= 0 && (T[a].terminal || T[a].bound == B_EXACT) && T[a].depth > tmDepth) {
                    tmDepth = T[a].depth;
                    tmStable = T[a].move == tmPrev ? std::min(tmStable + 1, 10) : 0;
                    tmPrev = T[a].move;
                    int sc = -T[a].score;
                    double frac = T[0].subNodes ? double(T[a].subNodes) / double(T[0].subNodes) : 0.5;
                    double scale = (1.6 - frac) * 1.3 * (tmStable >= 4 ? 0.85 : tmStable == 0 ? 1.3 : 1.0);
                    if (tmPrevScore != NONE && sc < tmPrevScore - 30) scale *= 1.2;
                    tmPrevScore = sc;
                    timeUp = now_ms() - LIM.start >= LIM.soft * scale;
                }
            }
            if (STOP || timeUp) break;
            if (!LIM.infinite && root_proven()) break;
            int64_t el = now_ms() - LIM.start;
            if (LIM.timed && el >= LIM.hard) { STOP = true; break; }
            if (P.tm == 0 && LIM.timed && !LIM.infinite && best_root_child() >= 0) {
                double pw = p_wrong();
                double scale = pw < 0.02 ? 0.6 : pw > 0.3 ? 1.6 : 1.0;
                if (el >= LIM.soft * scale) break;
            }
            if (LIM.nodes) {
                uint64_t tot = 0; for (auto& th : THREADS) tot += th->nodes;
                if (tot >= LIM.nodes) break;
            }
            if (now_ms() - lastInfo >= 500) { info(el); lastInfo = now_ms(); }
        }
        STOP = true;
        for (Job* j : pool.finish()) { integrate(j); delete j; }
        info(now_ms() - LIM.start);
        if (P.debug) {
            const Node& r = T[0];
            std::cout << "info string stats probes " << nProbes << " rootprobe " << rootProbeNodes << " safeDepth " << safeDepth
                      << " complete " << round_complete() << " bykind";
            for (int k = 0; k < 7; k++) std::cout << " " << k << ":" << catCount[k] << "/" << catNodes[k];
            std::cout << std::endl;
            std::cout << "info string root depth " << r.depth << " score " << r.score << " bound " << r.bound << " mu " << r.mu << " sd " << r.sd << std::endl;
            for (int i = r.first; i < r.first + r.n; i++) {
                const Node& y = T[i];
                std::cout << "info string child " << move_str(y.move) << " d " << y.depth << " score " << y.score << " bound " << y.bound
                          << " view " << neg(y.mu) << " sd " << y.sd << " nodes " << y.lastNodes << " sub " << y.subNodes << " ref " << y.refines
                          << " fails " << y.fails << " exp " << y.expanded << " pend " << y.pending << " busy " << y.busy << " lo " << y.lo << " hi " << y.hi << std::endl;
            }
        }
        int a = final_child();
        if (a >= 0) return T[a].move;
        // stopped before the root was expanded: alpha-beta's rule (a fail-high move counts)
        return T[0].fhMove ? T[0].fhMove : T[0].pvLen > 0 ? T[0].pv[0] : 0;
    }
};

// One DFPS search from ROOT under LIM; returns the best move (0 if nothing was probed).
Move run() {
    static Search s;
    return s.run();
}

// ---------------------------------------------------------------- data for the error model
// dfpsgen <in.epd> <out.csv> <maxDepth>: probe every position at depths 1..maxDepth (one thread
// per position) and write one row per completed depth:
//   index,pieces,depth,score,bound,bestmove,nodes
// tools/fit_dfps.py turns this into the sigma table read by Model::load.
void gen(const std::string& in, const std::string& out, int maxDepth) {
    std::ifstream f(in);
    if (!f) { std::cout << "info string cannot read " << in << std::endl; return; }
    std::vector<std::string> fens;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string a, b, c, d;
        if (!(is >> a >> b >> c >> d)) continue;
        fens.push_back(a + " " + b + " " + c + " " + d + " 0 1");
    }
    std::ofstream o(out);
    if (!o) { std::cout << "info string cannot write " << out << std::endl; return; }
    o << "index,pieces,depth,score,bound,bestmove,nodes\n";
    std::mutex om;
    std::atomic<size_t> next{0};
    STOP = false;
    LIM = Limits();
    LIM.start = now_ms();
    std::vector<std::thread> th;
    for (size_t w = 0; w < THREADS.size(); w++)
        th.emplace_back([&, w]() {
            Thread& t = *THREADS[w];
            while (true) {
                size_t i = next++;
                if (i >= fens.size()) break;
                Job j;
                if (!j.pos.set_fen(fens[i])) continue;
                j.path = {j.pos.key};
                j.depth = maxDepth;
                t.clear();
                std::vector<std::string> rows;
                uint64_t n0 = t.nodes;
                run_probe(t, j, [&](int d, int v, int bound, Move m) {
                    rows.push_back(std::to_string(i) + "," + std::to_string(popcnt(j.pos.occ())) + "," + std::to_string(d) + "," +
                                   std::to_string(v) + "," + std::to_string(bound) + "," + move_str(m) + "," +
                                   std::to_string(t.nodes - n0));
                });
                std::lock_guard<std::mutex> lk(om);
                for (auto& r : rows) o << r << "\n";
            }
        });
    for (auto& x : th) x.join();
    std::cout << "info string dfpsgen wrote " << fens.size() << " positions to " << out << std::endl;
}

// dfpscal <in.epd> <out.csv> <depth>: calibration data for beliefs about refuted root moves. Each
// position gets an alpha-beta root search to <depth> that records every root move; every move that
// only got an upper bound is then searched with a full window at depth-1 (one ply below the root,
// as in the root search). One row per such move:
//   index,move,searched,bound,alpha,best,exact
// (root view: "bound" is the fail-soft upper bound, "searched" its depth, "best" the root value).
void cal(const std::string& in, const std::string& out, int depth) {
    std::ifstream f(in);
    if (!f) { std::cout << "info string cannot read " << in << std::endl; return; }
    std::vector<std::string> fens;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string a, b, c, d;
        if (!(is >> a >> b >> c >> d)) continue;
        fens.push_back(a + " " + b + " " + c + " " + d + " 0 1");
    }
    std::ofstream o(out);
    if (!o) { std::cout << "info string cannot write " << out << std::endl; return; }
    o << "index,move,searched,bound,alpha,best,exact\n";
    std::mutex om;
    std::atomic<size_t> next{0};
    STOP = false;
    LIM = Limits();
    LIM.start = now_ms();
    std::vector<std::thread> th;
    for (size_t w = 0; w < THREADS.size(); w++)
        th.emplace_back([&, w]() {
            Thread& t = *THREADS[w];
            while (true) {
                size_t i = next++;
                if (i >= fens.size()) break;
                Job r;
                if (!r.pos.set_fen(fens[i])) continue;
                r.path = {r.pos.key}; r.depth = depth; r.wantRoot = true;
                t.clear();
                run_probe(t, r);
                if (r.aborted || r.bound != B_EXACT) continue;
                std::vector<std::string> rows;
                for (const RootRec& x : r.rootRec) {
                    if (x.bound != B_UPPER || std::abs(x.score) >= TB_BOUND) continue;
                    Job c;
                    c.pos = r.pos; c.pos.make(x.move);
                    c.hasParent = true; c.parent = r.pos; c.move = x.move;
                    c.path = {r.pos.key, c.pos.key}; c.depth = std::max(1, depth - 1);
                    run_probe(t, c);
                    if (c.aborted || c.bound != B_EXACT) continue;
                    rows.push_back(std::to_string(i) + "," + move_str(x.move) + "," + std::to_string(x.depth) + "," +
                                   std::to_string(x.score) + "," + std::to_string(x.alpha) + "," + std::to_string(r.score) + "," +
                                   std::to_string(-c.score));
                }
                std::lock_guard<std::mutex> lk(om);
                for (auto& row : rows) o << row << "\n";
            }
        });
    for (auto& x : th) x.join();
    std::cout << "info string dfpscal done " << fens.size() << " positions" << std::endl;
}

}  // namespace dfps
