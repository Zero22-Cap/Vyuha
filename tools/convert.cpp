// Binpack -> Vyuha training records (72 bytes each).
// Uses the nnue-pytorch binpack reader (ext/). Output: shards of fixed-size records.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "../ext/binpack.h"

#pragma pack(push, 1)
struct Rec {
    uint8_t sq[32];
    uint8_t pc[32];   // type + 6*color, 255 = empty slot
    int16_t score;    // side-to-move perspective, SF internal units
    int8_t result;    // side-to-move perspective: -1/0/1
    uint8_t stm;      // 0 white, 1 black
    uint16_t move;    // from | to<<6 | promo<<12 (1..4 = N,B,R,Q)
    uint16_t ply;
};
#pragma pack(pop)
static_assert(sizeof(Rec) == 72, "rec size");

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: convert in.binpack outprefix maxRecords shardSize\n"); return 1; }
    std::string in = argv[1], outp = argv[2];
    long long maxRec = std::atoll(argv[3]), shard = std::atoll(argv[4]);
    binpack::CompressedTrainingDataEntryReader reader(in);
    std::vector<Rec> buf; buf.reserve(1 << 20);
    long long seen = 0, kept = 0, inShard = 0; int shardNo = 0;
    FILE* out = nullptr;
    auto openShard = [&]() {
        char name[512]; std::snprintf(name, sizeof name, "%s_%03d.bin", outp.c_str(), shardNo++);
        out = std::fopen(name, "wb"); inShard = 0;
    };
    auto flush = [&]() { if (!buf.empty()) { std::fwrite(buf.data(), sizeof(Rec), buf.size(), out); buf.clear(); } };
    openShard();
    long long skCap = 0, skChk = 0, skScore = 0;
    while (reader.hasNext() && kept < maxRec) {
        auto e = reader.next();
        ++seen;
        if (e.score == 32002 || e.score > 10000 || e.score < -10000) { ++skScore; continue; }
        if (e.isCapturingMove() || e.move.type == chess::MoveType::Promotion || e.move.type == chess::MoveType::EnPassant) { ++skCap; continue; }
        if (e.pos.isCheck()) { ++skChk; continue; }
        Rec r; std::memset(r.sq, 0, 32); std::memset(r.pc, 255, 32);
        int n = 0;
        for (int s = 0; s < 64 && n < 32; ++s) {
            auto p = e.pos.pieceAt(chess::Square(s));
            if (p == chess::Piece::none()) continue;
            r.sq[n] = (uint8_t)s;
            r.pc[n] = (uint8_t)((int)chess::ordinal(p.type()) + 6 * (int)chess::ordinal(p.color()));
            ++n;
        }
        r.score = e.score;
        r.result = (int8_t)e.result;
        r.stm = (uint8_t)chess::ordinal(e.pos.sideToMove());
        int from = (int)chess::ordinal(e.move.from), to = (int)chess::ordinal(e.move.to);
        if (e.move.type == chess::MoveType::Castle) { // chess.h encodes castling as king->rook
            to = (to > from) ? from + 2 : from - 2;
        }
        r.move = (uint16_t)(from | (to << 6));
        r.ply = e.ply;
        buf.push_back(r);
        ++kept; ++inShard;
        if (buf.size() >= (1u << 20)) flush();
        if (inShard >= shard) { flush(); std::fclose(out); openShard(); }
        if ((kept & ((1 << 24) - 1)) == 0)
            std::fprintf(stderr, "seen %lld kept %lld (skip cap %lld chk %lld score %lld)\n", seen, kept, skCap, skChk, skScore);
    }
    flush(); std::fclose(out);
    std::fprintf(stderr, "DONE seen %lld kept %lld shards %d\n", seen, kept, shardNo);
    return 0;
}
