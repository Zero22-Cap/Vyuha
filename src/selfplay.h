// Vyuha - self-play data for the self-learned net (layer 2 of the architecture).
//
// selfplay <out.bin> <games> <softNodes> <maxDepth> [randomPlies] [nonet]
// Every worker thread plays whole games: a few random opening plies, then every move from a
// search with a soft node limit. Games end by mate, stalemate, the draw rules, or adjudication on
// the search score; tablebases are never used. Each quiet position (not in check, best move not a
// capture or promotion) becomes a 72-byte record in the format of tools/convert.cpp, which
// train/train.py reads: the search score and the game result, both for the side to move.
// Positions with a proven mate are kept (proof labels): score clamped to PROOF_SCORE, a sure win/loss.
// "nonet" plays with material only, so generation 0 starts from zero knowledge.
#pragma once
#include <filesystem>
#include <random>

namespace selfplay {

constexpr int PROOF_SCORE = 4000;  // recorded score of a position with a proven mate

#pragma pack(push, 1)
struct Rec {
    uint8_t sq[32];
    uint8_t pc[32];   // type + 6*color, 255 = empty slot
    int16_t score;    // side to move, internal units
    int8_t result;    // side to move: -1/0/1
    uint8_t stm;      // 0 white, 1 black
    uint16_t move;    // from | to<<6 (castling as the king's two-square move)
    uint16_t ply;
};
#pragma pack(pop)
static_assert(sizeof(Rec) == 72, "rec size");

inline bool insufficient(const Board& b) {
    return !(b.byPt[PAWN] | b.byPt[ROOK] | b.byPt[QUEEN]) && popcnt(b.occ() ^ b.byPt[KING]) <= 1;
}

inline bool threefold(const std::vector<U64>& keys, int hm) {
    int n = int(keys.size()), c = 1;
    for (int j = n - 3; j >= std::max(0, n - 1 - hm); j -= 2)
        if (keys[j] == keys[n - 1] && ++c >= 3) return true;
    return false;
}

inline void legal_moves(const Board& b, std::vector<Move>& out) {
    MoveList ml; gen_moves(b, ml, true, true);
    out.clear();
    for (int i = 0; i < ml.n; i++) if (b.legal(ml.m[i])) out.push_back(ml.m[i]);
}

struct Stats { std::atomic<long long> games{0}, recs{0}, w{0}, d{0}, l{0}, skipped{0}; };

// one game; returns false if the opening was rejected or the search was stopped
bool play(Thread& t, std::mt19937_64& rng, uint64_t softNodes, int maxDepth, int randomPlies,
          std::vector<Rec>& recs, int& result) {
    recs.clear(); result = 0;
    Board b; b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::vector<U64> keys{b.key};
    std::vector<Move> lm;
    int plies = randomPlies + int(rng() % 2);  // both colours start the scored part
    for (int i = 0; i < plies; i++) {
        legal_moves(b, lm);
        if (lm.empty()) return false;
        b.make(lm[rng() % lm.size()]);
        keys.push_back(b.key);
    }
    t.clear();
    {   // reject openings that are already decided
        dfps::Job j; j.pos = b; j.game = keys; j.depth = 6;
        dfps::run_probe(t, j);
        if (j.aborted || std::abs(j.score) > 400) return false;
    }
    int winRun = 0, winSign = 0, drawRun = 0;
    for (int ply = 0;; ply++) {
        legal_moves(b, lm);
        if (lm.empty()) { result = b.checkers ? (b.stm == WHITE ? -1 : 1) : 0; break; }
        if (b.hm >= 100 || insufficient(b) || threefold(keys, b.hm) || ply >= 400) { result = 0; break; }
        dfps::Job j; j.pos = b; j.game = keys; j.depth = maxDepth; j.softNodes = softNodes;
        dfps::run_probe(t, j);
        if (j.aborted) return false;
        int s = j.score;
        Move m = j.pvLen > 0 ? j.pv[0] : lm[0];
        if (std::find(lm.begin(), lm.end(), m) == lm.end()) m = lm[0];
        // a proven mate (exact mate score) is kept as a sure win or loss: its score is clamped to
        // PROOF_SCORE, which the trainer's sigmoid maps to an expected score of about 1 or 0
        bool proven = std::abs(s) >= MATE_BOUND && j.bound == B_EXACT;
        if (!b.checkers && !b.is_noisy(m) && mtype(m) != MF_PROMO && (std::abs(s) < 10000 || proven)) {
            Rec r; std::memset(r.sq, 0, 32); std::memset(r.pc, 255, 32);
            int n = 0;
            for (int sq = 0; sq < 64 && n < 32; sq++)
                if (b.sq[sq] != NO_PIECE) { r.sq[n] = uint8_t(sq); r.pc[n] = b.sq[sq]; n++; }
            r.score = int16_t(proven ? (s > 0 ? PROOF_SCORE : -PROOF_SCORE) : s); r.result = 0; r.stm = uint8_t(b.stm);
            r.move = uint16_t(from_sq(m) | (to_sq(m) << 6)); r.ply = uint16_t(std::min(ply + plies, 65535));
            recs.push_back(r);
        }
        // adjudication on the search score (white's view), no tablebases
        int ws = b.stm == WHITE ? s : -s;
        if (std::abs(ws) >= 2500) {
            int sg = ws > 0 ? 1 : -1;
            winRun = sg == winSign ? winRun + 1 : 1; winSign = sg;
            if (winRun >= 4) { result = sg; break; }
        } else winRun = 0;
        drawRun = (ply + plies >= 80 && std::abs(s) <= 8) ? drawRun + 1 : 0;
        if (drawRun >= 12) { result = 0; break; }
        b.make(m);
        keys.push_back(b.key);
    }
    for (auto& r : recs) r.result = int8_t(r.stm == WHITE ? result : -result);
    return true;
}

void run(const std::string& out, long long games, uint64_t softNodes, int maxDepth, int randomPlies) {
    // a file cut mid-record (process stopped while writing) must be trimmed to whole records
    // before appending, or every new record would be shifted
    std::error_code ec;
    auto size = std::filesystem::file_size(out, ec);
    if (!ec && size % sizeof(Rec)) {
        std::filesystem::resize_file(out, size - size % sizeof(Rec), ec);
        std::cout << "info string trimmed " << size % sizeof(Rec) << " bytes of a partial record from " << out << std::endl;
    }
    FILE* f = std::fopen(out.c_str(), "ab");
    if (!f) { std::cout << "info string cannot write " << out << std::endl; return; }
    STOP = false;
    LIM = Limits();
    LIM.start = now_ms();
    Stats st;
    std::atomic<long long> next{0};
    std::mutex fm;
    uint64_t seed = uint64_t(now_ms()) * 0x9E3779B97F4A7C15ULL;
    std::vector<std::thread> th;
    for (size_t w = 0; w < THREADS.size(); w++)
        th.emplace_back([&, w]() {
            Thread& t = *THREADS[w];
            std::mt19937_64 rng(seed ^ (w * 0xD1B54A32D192ED03ULL));
            std::vector<Rec> recs;
            int result;
            while (!STOP) {
                if (next++ >= games) break;
                if (!play(t, rng, softNodes, maxDepth, randomPlies, recs, result)) { st.skipped++; next--; continue; }
                {
                    std::lock_guard<std::mutex> lk(fm);
                    std::fwrite(recs.data(), sizeof(Rec), recs.size(), f);
                }
                st.recs += long(recs.size());
                (result > 0 ? st.w : result < 0 ? st.l : st.d)++;
                long long g = ++st.games;
                if (g % 1000 == 0) {
                    std::lock_guard<std::mutex> lk(fm);
                    std::fflush(f);
                    int64_t el = std::max<int64_t>(1, now_ms() - LIM.start);
                    std::cout << "info string selfplay games " << g << " records " << st.recs << " white/draw/black " << st.w << "/"
                              << st.d << "/" << st.l << " skipped " << st.skipped << " rec/s " << st.recs * 1000 / el << std::endl;
                }
            }
        });
    for (auto& x : th) x.join();
    std::fclose(f);
    std::cout << "info string selfplay done games " << st.games << " records " << st.recs << " white/draw/black " << st.w << "/"
              << st.d << "/" << st.l << std::endl;
}

}  // namespace selfplay
