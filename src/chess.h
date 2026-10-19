// Vyuha - board representation, attacks, move generation, SEE.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>
#include <immintrin.h>

using U64 = uint64_t;
using Move = uint16_t;
enum { WHITE, BLACK };
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr int NO_PIECE = 12;
constexpr int NO_SQ = 64;
constexpr Move NULLMOVE = 65;  // a1->b1 pattern never generated as a real move with this exact code
inline int make_piece(int c, int pt) { return pt + 6 * c; }
inline int ptype(int pc) { return pc % 6; }
inline int pcolor(int pc) { return pc / 6; }

constexpr int MF_PROMO = 1 << 14, MF_EP = 2 << 14, MF_CASTLE = 3 << 14;
inline int from_sq(Move m) { return m & 63; }
inline int to_sq(Move m) { return (m >> 6) & 63; }
inline int mtype(Move m) { return m & (3 << 14); }
inline int promo_pt(Move m) { return ((m >> 12) & 3) + KNIGHT; }
inline Move mk(int f, int t, int flags = 0) { return Move(f | (t << 6) | flags); }
inline Move mkpromo(int f, int t, int pt) { return Move(f | (t << 6) | MF_PROMO | ((pt - KNIGHT) << 12)); }

inline int lsb(U64 b) { return __builtin_ctzll(b); }
inline int popcnt(U64 b) { return __builtin_popcountll(b); }
inline int poplsb(U64& b) { int s = lsb(b); b &= b - 1; return s; }
constexpr U64 bb(int s) { return 1ULL << s; }
constexpr U64 FileA = 0x0101010101010101ULL, FileH = FileA << 7;
constexpr U64 Rank1 = 0xFFULL, Rank2 = Rank1 << 8, Rank3 = Rank1 << 16, Rank6 = Rank1 << 40, Rank7 = Rank1 << 48, Rank8 = Rank1 << 56;

inline U64 PawnAtt[2][64], KnightAtt[64], KingAtt[64];
inline U64 RookMask[64], BishopMask[64];
inline U64* RookPtr[64];
inline U64* BishopPtr[64];
inline U64 SlideTable[102400 + 5248];
inline U64 Between[64][64], Line[64][64];
inline U64 ZPiece[12][64], ZSide, ZCastle[16], ZEp[8];
inline int CastleKeep[64];
constexpr int SEEVal[7] = {100, 320, 330, 500, 950, 0, 0};

inline U64 rook_att(int s, U64 occ) { return RookPtr[s][_pext_u64(occ, RookMask[s])]; }
inline U64 bishop_att(int s, U64 occ) { return BishopPtr[s][_pext_u64(occ, BishopMask[s])]; }
inline U64 queen_att(int s, U64 occ) { return rook_att(s, occ) | bishop_att(s, occ); }

inline U64 slide_slow(int s, U64 occ, bool rook) {
    static const int dr[2][4][2] = {{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}, {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
    U64 a = 0; int r = s / 8, f = s % 8;
    for (int d = 0; d < 4; d++) {
        int rr = r + dr[rook][d][0], ff = f + dr[rook][d][1];
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
            int t = rr * 8 + ff; a |= bb(t);
            if (occ & bb(t)) break;
            rr += dr[rook][d][0]; ff += dr[rook][d][1];
        }
    }
    return a;
}

inline void init_tables() {
    for (int s = 0; s < 64; s++) {
        int r = s / 8, f = s % 8;
        auto on = [](int rr, int ff) { return rr >= 0 && rr < 8 && ff >= 0 && ff < 8; };
        PawnAtt[WHITE][s] = PawnAtt[BLACK][s] = KnightAtt[s] = KingAtt[s] = 0;
        if (on(r + 1, f - 1)) PawnAtt[WHITE][s] |= bb((r + 1) * 8 + f - 1);
        if (on(r + 1, f + 1)) PawnAtt[WHITE][s] |= bb((r + 1) * 8 + f + 1);
        if (on(r - 1, f - 1)) PawnAtt[BLACK][s] |= bb((r - 1) * 8 + f - 1);
        if (on(r - 1, f + 1)) PawnAtt[BLACK][s] |= bb((r - 1) * 8 + f + 1);
        const int kn[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
        for (auto& d : kn) if (on(r + d[0], f + d[1])) KnightAtt[s] |= bb((r + d[0]) * 8 + f + d[1]);
        for (int dr = -1; dr <= 1; dr++) for (int df = -1; df <= 1; df++)
            if ((dr || df) && on(r + dr, f + df)) KingAtt[s] |= bb((r + dr) * 8 + f + df);
    }
    U64* p = SlideTable;
    for (int rook = 0; rook < 2; rook++)
        for (int s = 0; s < 64; s++) {
            U64 edges = ((Rank1 | Rank8) & ~(Rank1 << (8 * (s / 8)))) | ((FileA | FileH) & ~(FileA << (s % 8)));
            U64 mask = slide_slow(s, 0, rook) & ~edges;
            if (rook) { RookMask[s] = mask; RookPtr[s] = p; } else { BishopMask[s] = mask; BishopPtr[s] = p; }
            int n = 1 << popcnt(mask);
            for (int i = 0; i < n; i++) p[i] = slide_slow(s, _pdep_u64(i, mask), rook);
            p += n;
        }
    for (int a = 0; a < 64; a++)
        for (int b = 0; b < 64; b++) {
            Between[a][b] = Line[a][b] = 0;
            if (a == b) continue;
            if (bishop_att(a, 0) & bb(b)) {
                Line[a][b] = (bishop_att(a, 0) & bishop_att(b, 0)) | bb(a) | bb(b);
                Between[a][b] = bishop_att(a, bb(b)) & bishop_att(b, bb(a));
            } else if (rook_att(a, 0) & bb(b)) {
                Line[a][b] = (rook_att(a, 0) & rook_att(b, 0)) | bb(a) | bb(b);
                Between[a][b] = rook_att(a, bb(b)) & rook_att(b, bb(a));
            }
        }
    U64 seed = 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() { U64 z = (seed += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); };
    for (auto& a : ZPiece) for (auto& x : a) x = rnd();
    ZSide = rnd();
    for (auto& x : ZCastle) x = rnd();
    for (auto& x : ZEp) x = rnd();
    for (int s = 0; s < 64; s++) CastleKeep[s] = 15;
    CastleKeep[4] = ~3 & 15; CastleKeep[7] = ~1 & 15; CastleKeep[0] = ~2 & 15;
    CastleKeep[60] = ~12 & 15; CastleKeep[63] = ~4 & 15; CastleKeep[56] = ~8 & 15;
}

struct MoveList {
    Move m[256]; int s[256]; int n = 0;
    void add(Move x) { m[n++] = x; }
};

struct Board {
    U64 byPt[6], byC[2];
    uint8_t sq[64];
    int stm, castle, ep, hm, fm;
    U64 key, pawnKey, npKey[2], minorKey;  // minorKey: knights, bishops and kings
    U64 checkers, pinned;
    // NNUE dirty info for the move that produced this position
    int nAdd, nSub, addPc[2], addSq[2], subPc[2], subSq[2];

    U64 pieces(int c, int pt) const { return byPt[pt] & byC[c]; }
    U64 occ() const { return byC[0] | byC[1]; }
    int king(int c) const { return lsb(pieces(c, KING)); }
    bool non_pawn(int c) const { return byC[c] & ~(byPt[PAWN] | byPt[KING]); }
    bool is_capture(Move m) const { return sq[to_sq(m)] != NO_PIECE || mtype(m) == MF_EP; }
    bool is_noisy(Move m) const { return is_capture(m) || (mtype(m) == MF_PROMO && promo_pt(m) == QUEEN); }

    void put(int pc, int s) {
        byPt[ptype(pc)] |= bb(s); byC[pcolor(pc)] |= bb(s); sq[s] = pc;
        key ^= ZPiece[pc][s];
        if (ptype(pc) == PAWN) pawnKey ^= ZPiece[pc][s]; else npKey[pcolor(pc)] ^= ZPiece[pc][s];
        if (ptype(pc) == KNIGHT || ptype(pc) == BISHOP || ptype(pc) == KING) minorKey ^= ZPiece[pc][s];
    }
    void remove(int s) {
        int pc = sq[s];
        byPt[ptype(pc)] ^= bb(s); byC[pcolor(pc)] ^= bb(s); sq[s] = NO_PIECE;
        key ^= ZPiece[pc][s];
        if (ptype(pc) == PAWN) pawnKey ^= ZPiece[pc][s]; else npKey[pcolor(pc)] ^= ZPiece[pc][s];
        if (ptype(pc) == KNIGHT || ptype(pc) == BISHOP || ptype(pc) == KING) minorKey ^= ZPiece[pc][s];
    }
    void d_add(int pc, int s) { addPc[nAdd] = pc; addSq[nAdd++] = s; }
    void d_sub(int pc, int s) { subPc[nSub] = pc; subSq[nSub++] = s; }

    U64 attackers_to(int s, U64 o) const {
        return (PawnAtt[BLACK][s] & pieces(WHITE, PAWN)) | (PawnAtt[WHITE][s] & pieces(BLACK, PAWN)) |
               (KnightAtt[s] & byPt[KNIGHT]) | (KingAtt[s] & byPt[KING]) |
               (bishop_att(s, o) & (byPt[BISHOP] | byPt[QUEEN])) | (rook_att(s, o) & (byPt[ROOK] | byPt[QUEEN]));
    }
    bool attacked(int s, int by) const { return attackers_to(s, occ()) & byC[by]; }

    void update_checks() {
        int us = stm, them = us ^ 1, k = king(us);
        checkers = attackers_to(k, occ()) & byC[them];
        pinned = 0;
        U64 snipers = (rook_att(k, 0) & (pieces(them, ROOK) | pieces(them, QUEEN))) |
                      (bishop_att(k, 0) & (pieces(them, BISHOP) | pieces(them, QUEEN)));
        U64 o = occ();
        while (snipers) {
            int s = poplsb(snipers);
            U64 b = Between[k][s] & o;
            if (b && !(b & (b - 1)) && (b & byC[us])) pinned |= b;
        }
    }

    bool set_fen(const std::string& fen) {
        std::memset(this, 0, sizeof(Board));
        for (int s = 0; s < 64; s++) sq[s] = NO_PIECE;
        std::istringstream is(fen);
        std::string pl, side, cas, eps; int h = 0, f = 1;
        is >> pl >> side >> cas >> eps >> h >> f;
        int r = 7, fl = 0;
        for (char c : pl) {
            if (c == '/') { r--; fl = 0; continue; }
            if (c >= '1' && c <= '8') { fl += c - '0'; continue; }
            const char* P = "PNBRQKpnbrqk";
            const char* q = std::strchr(P, c);
            if (!q) return false;
            put(int(q - P), r * 8 + fl); fl++;
        }
        stm = side == "b" ? BLACK : WHITE;
        castle = 0;
        for (char c : cas) castle |= c == 'K' ? 1 : c == 'Q' ? 2 : c == 'k' ? 4 : c == 'q' ? 8 : 0;
        ep = NO_SQ;
        if (eps.size() == 2) {
            int e = (eps[1] - '1') * 8 + (eps[0] - 'a');
            if (PawnAtt[stm ^ 1][e] & pieces(stm, PAWN)) { ep = e; key ^= ZEp[e & 7]; }
        }
        hm = h; fm = f;
        key ^= ZCastle[castle];
        if (stm == BLACK) key ^= ZSide;
        update_checks();
        return true;
    }

    std::string fen() const {
        std::string s;
        for (int r = 7; r >= 0; r--) {
            int e = 0;
            for (int f = 0; f < 8; f++) {
                int p = sq[r * 8 + f];
                if (p == NO_PIECE) { e++; continue; }
                if (e) { s += char('0' + e); e = 0; }
                s += "PNBRQKpnbrqk"[p];
            }
            if (e) s += char('0' + e);
            if (r) s += '/';
        }
        s += stm == WHITE ? " w " : " b ";
        std::string c;
        if (castle & 1) c += 'K';
        if (castle & 2) c += 'Q';
        if (castle & 4) c += 'k';
        if (castle & 8) c += 'q';
        s += c.empty() ? "-" : c;
        s += ' ';
        if (ep == NO_SQ) s += '-'; else { s += char('a' + ep % 8); s += char('1' + ep / 8); }
        s += " " + std::to_string(hm) + " " + std::to_string(fm);
        return s;
    }

    bool legal(Move m) const {
        int us = stm, them = us ^ 1, f = from_sq(m), t = to_sq(m), k = king(us);
        if (mtype(m) == MF_EP) {
            int cs = t + (us == WHITE ? -8 : 8);
            U64 o = (occ() ^ bb(f) ^ bb(cs)) | bb(t);
            return !(attackers_to(k, o) & byC[them] & ~bb(cs));
        }
        if (ptype(sq[f]) == KING) return !(attackers_to(t, occ() ^ bb(f)) & byC[them]);
        if (checkers) {
            if (checkers & (checkers - 1)) return false;
            int c = lsb(checkers);
            if (!((Between[k][c] | checkers) & bb(t))) return false;
        }
        if (pinned & bb(f)) return Line[f][k] & bb(t);
        return true;
    }

    void make(Move m) {
        nAdd = nSub = 0;
        int f = from_sq(m), t = to_sq(m), pc = sq[f], us = stm, them = us ^ 1, type = mtype(m);
        int cap = type == MF_EP ? make_piece(them, PAWN) : sq[t];
        if (ep != NO_SQ) { key ^= ZEp[ep & 7]; ep = NO_SQ; }
        key ^= ZCastle[castle];
        hm++;
        if (cap != NO_PIECE) {
            int cs = type == MF_EP ? t + (us == WHITE ? -8 : 8) : t;
            remove(cs); d_sub(cap, cs); hm = 0;
        }
        if (type == MF_CASTLE) {
            int rf, rt;
            if (t > f) { rf = f + 3; rt = f + 1; } else { rf = f - 4; rt = f - 1; }
            int rook = sq[rf];
            remove(f); remove(rf); put(pc, t); put(rook, rt);
            d_sub(pc, f); d_add(pc, t); d_sub(rook, rf); d_add(rook, rt);
        } else {
            remove(f);
            int np = type == MF_PROMO ? make_piece(us, promo_pt(m)) : pc;
            put(np, t); d_sub(pc, f); d_add(np, t);
            if (ptype(pc) == PAWN) {
                hm = 0;
                if ((t ^ f) == 16) {
                    int e = (f + t) / 2;
                    if (PawnAtt[us][e] & pieces(them, PAWN)) { ep = e; key ^= ZEp[e & 7]; }
                }
            }
        }
        castle &= CastleKeep[f] & CastleKeep[t];
        key ^= ZCastle[castle];
        stm = them; key ^= ZSide;
        if (us == BLACK) fm++;
        update_checks();
    }

    void make_null() {
        nAdd = nSub = 0;
        if (ep != NO_SQ) { key ^= ZEp[ep & 7]; ep = NO_SQ; }
        stm ^= 1; key ^= ZSide; hm++;
        update_checks();
    }

    bool see_ge(Move m, int thr) const {
        if (mtype(m) != 0) return 0 >= thr;
        int f = from_sq(m), t = to_sq(m);
        int swap = (sq[t] == NO_PIECE ? 0 : SEEVal[ptype(sq[t])]) - thr;
        if (swap < 0) return false;
        swap = SEEVal[ptype(sq[f])] - swap;
        if (swap <= 0) return true;
        U64 o = occ() ^ bb(f) ^ bb(t);
        int s = stm;
        U64 att = attackers_to(t, o);
        U64 bq = byPt[BISHOP] | byPt[QUEEN], rq = byPt[ROOK] | byPt[QUEEN];
        int res = 1;
        while (true) {
            s ^= 1; att &= o;
            U64 my = att & byC[s];
            if (!my) break;
            res ^= 1;
            U64 b;
            if ((b = my & byPt[PAWN])) { if ((swap = SEEVal[PAWN] - swap) < res) break; o ^= b & -b; att |= bishop_att(t, o) & bq; }
            else if ((b = my & byPt[KNIGHT])) { if ((swap = SEEVal[KNIGHT] - swap) < res) break; o ^= b & -b; }
            else if ((b = my & byPt[BISHOP])) { if ((swap = SEEVal[BISHOP] - swap) < res) break; o ^= b & -b; att |= bishop_att(t, o) & bq; }
            else if ((b = my & byPt[ROOK])) { if ((swap = SEEVal[ROOK] - swap) < res) break; o ^= b & -b; att |= rook_att(t, o) & rq; }
            else if ((b = my & byPt[QUEEN])) { if ((swap = SEEVal[QUEEN] - swap) < res) break; o ^= b & -b; att |= (bishop_att(t, o) & bq) | (rook_att(t, o) & rq); }
            else return (att & ~byC[s]) ? res ^ 1 : res;
        }
        return bool(res);
    }
};

// noisy = captures, ep, all promotions that capture, queen push-promotions. quiet = the rest.
inline void gen_moves(const Board& b, MoveList& ml, bool noisy, bool quiet) {
    int us = b.stm, them = us ^ 1;
    U64 o = b.occ(), enemy = b.byC[them], empty = ~o;
    U64 pawns = b.pieces(us, PAWN);
    U64 r7 = us == WHITE ? Rank7 : Rank2, r3 = us == WHITE ? Rank3 : Rank6;
    int up = us == WHITE ? 8 : -8;
    auto shift = [&](U64 x) { return us == WHITE ? x << 8 : x >> 8; };
    U64 p7 = pawns & r7, pn = pawns & ~r7;
    if (quiet) {
        U64 s1 = shift(pn) & empty, s2 = shift(s1 & r3) & empty;
        while (s1) { int t = poplsb(s1); ml.add(mk(t - up, t)); }
        while (s2) { int t = poplsb(s2); ml.add(mk(t - 2 * up, t)); }
    }
    if (noisy) {
        U64 x = pn;
        while (x) { int f = poplsb(x); U64 a = PawnAtt[us][f] & enemy; while (a) ml.add(mk(f, poplsb(a))); }
        if (b.ep != NO_SQ) { U64 a = PawnAtt[them][b.ep] & pn; while (a) ml.add(mk(poplsb(a), b.ep, MF_EP)); }
    }
    U64 x = p7;
    while (x) {
        int f = poplsb(x);
        int t = f + up;
        if (empty & bb(t)) {
            if (noisy) ml.add(mkpromo(f, t, QUEEN));
            if (quiet) { ml.add(mkpromo(f, t, KNIGHT)); ml.add(mkpromo(f, t, ROOK)); ml.add(mkpromo(f, t, BISHOP)); }
        }
        if (noisy) {
            U64 a = PawnAtt[us][f] & enemy;
            while (a) { int c = poplsb(a); for (int pt : {QUEEN, KNIGHT, ROOK, BISHOP}) ml.add(mkpromo(f, c, pt)); }
        }
    }
    U64 targets = (noisy ? enemy : 0) | (quiet ? empty : 0);
    for (int pt = KNIGHT; pt <= KING; pt++) {
        U64 ps = b.pieces(us, pt);
        while (ps) {
            int f = poplsb(ps);
            U64 a = pt == KNIGHT ? KnightAtt[f] : pt == BISHOP ? bishop_att(f, o) : pt == ROOK ? rook_att(f, o)
                  : pt == QUEEN ? queen_att(f, o) : KingAtt[f];
            a &= targets;
            while (a) ml.add(mk(f, poplsb(a)));
        }
    }
    if (quiet && !b.checkers) {
        if (us == WHITE) {
            if ((b.castle & 1) && !(o & (bb(5) | bb(6))) && !b.attacked(5, BLACK) && !b.attacked(6, BLACK)) ml.add(mk(4, 6, MF_CASTLE));
            if ((b.castle & 2) && !(o & (bb(1) | bb(2) | bb(3))) && !b.attacked(3, BLACK) && !b.attacked(2, BLACK)) ml.add(mk(4, 2, MF_CASTLE));
        } else {
            if ((b.castle & 4) && !(o & (bb(61) | bb(62))) && !b.attacked(61, WHITE) && !b.attacked(62, WHITE)) ml.add(mk(60, 62, MF_CASTLE));
            if ((b.castle & 8) && !(o & (bb(57) | bb(58) | bb(59))) && !b.attacked(59, WHITE) && !b.attacked(58, WHITE)) ml.add(mk(60, 58, MF_CASTLE));
        }
    }
}

inline std::string move_str(Move m) {
    if (!m) return "0000";
    std::string s;
    s += char('a' + from_sq(m) % 8); s += char('1' + from_sq(m) / 8);
    s += char('a' + to_sq(m) % 8); s += char('1' + to_sq(m) / 8);
    if (mtype(m) == MF_PROMO) s += "nbrq"[promo_pt(m) - KNIGHT];
    return s;
}

inline Move parse_move(const Board& b, const std::string& s) {
    MoveList ml; gen_moves(b, ml, true, true);
    for (int i = 0; i < ml.n; i++) if (move_str(ml.m[i]) == s && b.legal(ml.m[i])) return ml.m[i];
    return 0;
}

inline U64 perft(const Board& b, int d) {
    MoveList ml; gen_moves(b, ml, true, true);
    U64 n = 0;
    for (int i = 0; i < ml.n; i++) {
        if (!b.legal(ml.m[i])) continue;
        if (d == 1) { n++; continue; }
        Board c = b; c.make(ml.m[i]);
        n += perft(c, d - 1);
    }
    return n;
}
