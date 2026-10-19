// Build policy-training records: 72-byte rec + legal quiet move list (with moving piece type) + label index.
#include "../src/chess.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

#pragma pack(push, 1)
struct Rec { uint8_t sq[32]; uint8_t pc[32]; int16_t score; int8_t result; uint8_t stm; uint16_t move; uint16_t ply; };
struct Pol { Rec r; uint8_t n; uint8_t label; uint16_t mv[64]; };  // mv = from | to<<6 | pt<<12
#pragma pack(pop)
static_assert(sizeof(Pol) == 202, "pol");

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "polgen in.bin out.bin maxOut\n"); return 1; }
    init_tables();
    FILE* fi = std::fopen(argv[1], "rb"); FILE* fo = std::fopen(argv[2], "wb");
    long long maxOut = std::atoll(argv[3]), out = 0, seen = 0, miss = 0;
    std::vector<Rec> buf(1 << 16);
    std::vector<Pol> ob; ob.reserve(1 << 16);
    size_t got;
    while (out < maxOut && (got = std::fread(buf.data(), sizeof(Rec), buf.size(), fi)) > 0) {
        for (size_t i = 0; i < got && out < maxOut; i++) {
            const Rec& r = buf[i]; ++seen;
            Board b; std::memset(&b, 0, sizeof b);
            for (int s = 0; s < 64; s++) b.sq[s] = NO_PIECE;
            for (int k = 0; k < 32; k++) if (r.pc[k] != 255) b.put(r.pc[k], r.sq[k]);
            b.stm = r.stm; b.ep = NO_SQ;
            b.castle = 0;
            if (b.sq[4] == make_piece(WHITE, KING)) {
                if (b.sq[7] == make_piece(WHITE, ROOK)) b.castle |= 1;
                if (b.sq[0] == make_piece(WHITE, ROOK)) b.castle |= 2;
            }
            if (b.sq[60] == make_piece(BLACK, KING)) {
                if (b.sq[63] == make_piece(BLACK, ROOK)) b.castle |= 4;
                if (b.sq[56] == make_piece(BLACK, ROOK)) b.castle |= 8;
            }
            if (!b.pieces(WHITE, KING) || !b.pieces(BLACK, KING)) continue;
            b.update_checks();
            if (b.checkers) continue;
            MoveList ml; gen_moves(b, ml, false, true);
            Pol p; p.r = r; p.n = 0; p.label = 255;
            for (int j = 0; j < ml.n && p.n < 64; j++) {
                Move m = ml.m[j];
                if (mtype(m) == MF_PROMO || !b.legal(m)) continue;
                int f = from_sq(m), t = to_sq(m);
                if ((r.move & 4095) == (f | (t << 6))) p.label = p.n;
                p.mv[p.n++] = uint16_t(f | (t << 6) | (ptype(b.sq[f]) << 12));
            }
            if (p.label == 255 || p.n < 2) { ++miss; continue; }
            for (int j = p.n; j < 64; j++) p.mv[j] = 0;
            ob.push_back(p); ++out;
        }
        if (ob.size() >= (1u << 15)) { std::fwrite(ob.data(), sizeof(Pol), ob.size(), fo); ob.clear(); }
    }
    if (!ob.empty()) std::fwrite(ob.data(), sizeof(Pol), ob.size(), fo);
    std::fclose(fo); std::fclose(fi);
    std::fprintf(stderr, "seen %lld out %lld miss %lld\n", seen, out, miss);
    return 0;
}
