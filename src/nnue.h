// Vyuha NNUE inference: (10 king buckets x 768 -> HL) x 2 -> SCReLU -> 8 output buckets.
// Two heads over one accumulator: value and uncertainty.
#pragma once
#include "chess.h"
#include <cstdio>
#include <vector>
#include <algorithm>

namespace nnue {
constexpr int HL = 1024, KB = 10, OB = 8, QA = 255, QB = 64, SCALE = 180;
constexpr int Layout[64] = {0, 1, 2, 3, 3, 2, 1, 0, 4, 4, 5, 5, 5, 5, 4, 4, 6, 6, 7, 7, 7, 7, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8,
                            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

struct Net {
    alignas(64) int16_t ftW[KB * 768][HL];
    alignas(64) int16_t ftB[HL];
    alignas(64) int16_t outW[OB][2 * HL];
    alignas(64) int16_t uncW[OB][2 * HL];
    int32_t outB[OB], uncB[OB];
};
inline Net* NET = nullptr;
inline int NET_GEN = 0;  // incremented by every load (invalidates the refresh cache)

inline bool load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4]; int hdr[4];
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "SCN1", 4) == 0 &&
              std::fread(hdr, 4, 4, f) == 4 && hdr[0] == HL && hdr[1] == KB && hdr[2] == OB;
    Net* n = ok ? new Net : nullptr;
    if (ok) {
        ok = std::fread(n->ftW, sizeof(n->ftW), 1, f) == 1 && std::fread(n->ftB, sizeof(n->ftB), 1, f) == 1 &&
             std::fread(n->outW, sizeof(n->outW), 1, f) == 1 && std::fread(n->outB, sizeof(n->outB), 1, f) == 1 &&
             std::fread(n->uncW, sizeof(n->uncW), 1, f) == 1 && std::fread(n->uncB, sizeof(n->uncB), 1, f) == 1;
    }
    std::fclose(f);
    if (!ok) { delete n; return false; }
    Net* old = NET; NET = n; delete old;
    NET_GEN++;
    return true;
}

inline int kcode(int p, int ksq) {
    int k = ksq ^ (p ? 56 : 0);
    bool m = (k & 7) >= 4;
    if (m) k ^= 7;
    return Layout[k] * 2 + m;
}
inline int feat(int p, int ksq, int pc, int s) {
    int k = ksq ^ (p ? 56 : 0), x = s ^ (p ? 56 : 0);
    if ((k & 7) >= 4) { k ^= 7; x ^= 7; }
    return Layout[k] * 768 + ((pcolor(pc) == p ? 0 : 6) + ptype(pc)) * 64 + x;
}

struct Acc {
    alignas(64) int16_t v[2][HL];
    int kc[2];
    bool computed;
};

// Refresh cache ("Finny table"): per thread, perspective and king code, the accumulator of the
// position last refreshed there and its pieces; a refresh applies only the differences.
// NET_GEN invalidates the entries when another net is loaded.
struct Finny { alignas(64) int16_t v[HL]; U64 bb[12]; int gen = -1; };
inline thread_local Finny FINNY[2][2 * KB];

inline void refresh(Acc& a, const Board& b, int p) {
    int k = b.king(p), kc = kcode(p, k);
    Finny& e = FINNY[p][kc];
    if (e.gen != NET_GEN) {
        std::memcpy(e.v, NET->ftB, sizeof(int16_t) * HL);
        std::memset(e.bb, 0, sizeof e.bb);
        e.gen = NET_GEN;
    }
    int16_t* d = e.v;
    for (int pc = 0; pc < 12; pc++) {
        U64 cur = b.pieces(pc / 6, pc % 6);
        U64 add = cur & ~e.bb[pc], sub = e.bb[pc] & ~cur;
        e.bb[pc] = cur;
        while (add) { const int16_t* w = NET->ftW[feat(p, k, pc, poplsb(add))]; for (int i = 0; i < HL; i++) d[i] += w[i]; }
        while (sub) { const int16_t* w = NET->ftW[feat(p, k, pc, poplsb(sub))]; for (int i = 0; i < HL; i++) d[i] -= w[i]; }
    }
    std::memcpy(a.v[p], d, sizeof(int16_t) * HL);
    a.kc[p] = kc;
}

inline void update(const Acc& par, Acc& ch, const Board& b) {
    for (int p = 0; p < 2; p++) {
        int k = b.king(p);
        int kc = kcode(p, k);
        if (kc != par.kc[p]) { refresh(ch, b, p); continue; }
        ch.kc[p] = kc;
        int16_t* d = ch.v[p];
        const int16_t* s = par.v[p];
        if (b.nAdd == 1 && b.nSub == 1) {
            const int16_t* a0 = NET->ftW[feat(p, k, b.addPc[0], b.addSq[0])];
            const int16_t* s0 = NET->ftW[feat(p, k, b.subPc[0], b.subSq[0])];
            for (int i = 0; i < HL; i++) d[i] = s[i] + a0[i] - s0[i];
        } else if (b.nAdd == 1 && b.nSub == 2) {
            const int16_t* a0 = NET->ftW[feat(p, k, b.addPc[0], b.addSq[0])];
            const int16_t* s0 = NET->ftW[feat(p, k, b.subPc[0], b.subSq[0])];
            const int16_t* s1 = NET->ftW[feat(p, k, b.subPc[1], b.subSq[1])];
            for (int i = 0; i < HL; i++) d[i] = s[i] + a0[i] - s0[i] - s1[i];
        } else if (b.nAdd == 2 && b.nSub == 2) {
            const int16_t* a0 = NET->ftW[feat(p, k, b.addPc[0], b.addSq[0])];
            const int16_t* a1 = NET->ftW[feat(p, k, b.addPc[1], b.addSq[1])];
            const int16_t* s0 = NET->ftW[feat(p, k, b.subPc[0], b.subSq[0])];
            const int16_t* s1 = NET->ftW[feat(p, k, b.subPc[1], b.subSq[1])];
            for (int i = 0; i < HL; i++) d[i] = s[i] + a0[i] + a1[i] - s0[i] - s1[i];
        } else {
            std::memcpy(d, s, sizeof(int16_t) * HL);
        }
    }
}

inline int32_t screlu_dot(const int16_t* acc, const int16_t* w) {
#if defined(__AVX512BW__)
    __m512i sum = _mm512_setzero_si512(), z = _mm512_setzero_si512(), q = _mm512_set1_epi16(QA);
    for (int i = 0; i < HL; i += 32) {
        __m512i v = _mm512_min_epi16(_mm512_max_epi16(_mm512_load_si512((const void*)(acc + i)), z), q);
        __m512i ww = _mm512_load_si512((const void*)(w + i));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_mullo_epi16(v, ww), v));
    }
    return _mm512_reduce_add_epi32(sum);
#elif defined(__AVX2__)
    __m256i sum = _mm256_setzero_si256(), z = _mm256_setzero_si256(), q = _mm256_set1_epi16(QA);
    for (int i = 0; i < HL; i += 16) {
        __m256i v = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i*)(acc + i)), z), q);
        __m256i ww = _mm256_load_si256((const __m256i*)(w + i));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(v, ww), v));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#else
    int32_t sum = 0;
    for (int i = 0; i < HL; i++) { int v = std::clamp<int>(acc[i], 0, QA); sum += v * v * w[i]; }
    return sum;
#endif
}

inline int bucket(const Board& b) { return std::min(7, (popcnt(b.occ()) - 2) / 4); }

// value in cp (~SF internal / 2)
inline int evaluate(const Acc& a, const Board& b) {
    int ob = bucket(b), us = b.stm;
    int64_t s = screlu_dot(a.v[us], NET->outW[ob]) + screlu_dot(a.v[us ^ 1], NET->outW[ob] + HL);
    return int((s / QA + NET->outB[ob]) * SCALE / (QA * QB));
}
inline int uncertainty(const Acc& a, const Board& b) {
    int ob = bucket(b), us = b.stm;
    int64_t s = screlu_dot(a.v[us], NET->uncW[ob]) + screlu_dot(a.v[us ^ 1], NET->uncW[ob] + HL);
    return std::max(0, int((s / QA + NET->uncB[ob]) * SCALE / (QA * QB)));
}

// ---- policy head on the same accumulator (float; used only at deeper nodes)
constexpr int PD = 32;
struct Policy {
    std::vector<float> Wp, bp, Ef, Et, bf, bt;  // Wp: [2HL][PD]
};
inline Policy* POLICY = nullptr;
inline bool load_policy(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4]; int hdr[2];
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "SCP1", 4) == 0 && std::fread(hdr, 4, 2, f) == 2 && hdr[0] == HL && hdr[1] == PD;
    Policy* p = ok ? new Policy : nullptr;
    if (ok) {
        p->Wp.resize(2 * HL * PD); p->bp.resize(PD); p->Ef.resize(384 * PD); p->Et.resize(384 * PD); p->bf.resize(384); p->bt.resize(384);
        for (auto* v : {&p->Wp, &p->bp, &p->Ef, &p->Et, &p->bf, &p->bt})
            ok = ok && std::fread(v->data(), 4, v->size(), f) == v->size();
    }
    std::fclose(f);
    if (!ok) { delete p; return false; }
    POLICY = p;
    return true;
}
inline void policy_query(const Acc& a, int stm, float* q) {
    const Policy& P = *POLICY;
    for (int j = 0; j < PD; j++) q[j] = P.bp[j];
    constexpr float inv = 1.0f / (QA * QA);
    for (int h = 0; h < 2; h++) {
        const int16_t* v = a.v[h == 0 ? stm : stm ^ 1];
        const float* W = P.Wp.data() + h * HL * PD;
        for (int i = 0; i < HL; i++) {
            int x = v[i];
            if (x <= 0) continue;
            if (x > QA) x = QA;
            float fx = float(x * x) * inv;
            const float* w = W + i * PD;
            for (int j = 0; j < PD; j++) q[j] += fx * w[j];
        }
    }
}
inline float policy_logit(const float* q, int stm, bool kmir, int pt, int from, int to) {
    const Policy& P = *POLICY;
    int f = from ^ (stm ? 56 : 0), t = to ^ (stm ? 56 : 0);
    if (kmir) { f ^= 7; t ^= 7; }
    int fi = pt * 64 + f, ti = pt * 64 + t;
    const float* ef = P.Ef.data() + fi * PD; const float* et = P.Et.data() + ti * PD;
    float s = P.bf[fi] + P.bt[ti];
    for (int j = 0; j < PD; j++) s += q[j] * (ef[j] + et[j]);
    return s;
}
}  // namespace nnue
