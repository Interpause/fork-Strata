// A grammar's bitmap as the sampler's dense row: the conversion both decode paths share (grammar/mask_bias.hpp).
// Host-only, no GPU, no model, no llguidance - which is why it is here and not in a device test.
#include "strata/grammar/mask_bias.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

const float NINF = -std::numeric_limits<float>::infinity();

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "mask_bias regression: %s\n", what);
        std::exit(1);
    }
}

void allow(std::vector<uint32_t>& bits, int64_t id, uint32_t words) {
    if (id < (int64_t) words * 32) bits[(size_t) (id >> 5)] |= 1u << (id & 31);
}

}   // namespace

int main() {
    const int64_t nv = 100;   // not a multiple of 32: ids past the bitmap's last word are refused
    const uint32_t words = 2;  // 64 bits set-able, 36 ids of the vocabulary unbittable
    std::vector<float> out((size_t) nv);
    std::vector<float> base;
    std::vector<int32_t> banned;
    std::vector<uint32_t> bits;

    // 1. Two allowed ids, no request bias: they are zero, everything else is -inf.
    bits.assign(words, 0);
    allow(bits, 1, words);
    allow(bits, 50, words);
    check(strata::grammar::mask_to_bias(bits.data(), words, base, banned, out.data(), nv), "two allowed ids draw");
    check(out[1] == 0.0f && out[50] == 0.0f, "an allowed id with no request bias keeps zero");
    check(out[0] == NINF && out[2] == NINF && out[49] == NINF && out[51] == NINF, "a refused id is -inf");
    check(out[63] == NINF && out[64] == NINF && out[99] == NINF, "ids past the bitmap's words are refused");

    // 2. The request's own bias survives on the ids the grammar allows (the composition C3 specified).
    base.assign((size_t) nv, 0.0f);
    base[1] = 2.5f;
    base[50] = -1.0f;
    base[7] = 9.0f;   // refused by the mask, so its value must not appear
    check(strata::grammar::mask_to_bias(bits.data(), words, base, banned, out.data(), nv), "a biased row still draws");
    check(out[1] == 2.5f && out[50] == -1.0f, "an allowed id keeps the request's bias");
    check(out[7] == NINF, "a refused id is -inf whatever the request's bias says");

    // 3. A banned id (a draw the mask allowed and the matcher then refused) is -inf, and stops counting as drawable.
    banned = {50};
    check(strata::grammar::mask_to_bias(bits.data(), words, base, banned, out.data(), nv), "one of two allowed stays");
    check(out[50] == NINF && out[1] == 2.5f, "the banned id is -inf, the other allowed id is untouched");

    // 4. Only-allowed-id banned: nothing is drawable, so the caller must stop the slot rather than draw token 0.
    banned = {1, 50, 50, -1, 100000};   // duplicates and out-of-range entries are tolerated, not double counted
    check(!strata::grammar::mask_to_bias(bits.data(), words, base, banned, out.data(), nv), "a fully banned row stops");

    // 5. An allowed id whose own bias is a hard exclusion (-inf) is not drawable either: with 50 refused by the mask,
    //    the row has no drawable id left even though the bitmap sets one.
    banned.clear();
    bits.assign(words, 0);
    allow(bits, 1, words);
    std::vector<float> excluded = base;
    excluded[1] = NINF;
    check(strata::grammar::mask_to_bias(bits.data(), words, base, banned, out.data(), nv), "the same row draws plainly");
    check(!strata::grammar::mask_to_bias(bits.data(), words, excluded, banned, out.data(), nv),
          "an allowed id excluded by the request's own bias is not drawable");

    // 6. A mask that allows nothing at all, and a null bitmap: both are "stop", never a draw.
    bits.assign(words, 0);
    check(!strata::grammar::mask_to_bias(bits.data(), words, base, banned, out.data(), nv), "an empty mask stops");
    check(!strata::grammar::mask_to_bias(nullptr, 0, base, banned, out.data(), nv), "a null mask stops");
    check(out[0] == NINF && out[99] == NINF, "a stopping row is all -inf");

    // 7. Every id allowed (a free step): the row is the request's bias, and the conversion says it can draw.
    //    Four words cover the whole 100-id vocabulary; two words would leave ids 64..99 refused (case 1).
    const uint32_t full_words = 4;
    std::vector<uint32_t> full(full_words, 0);
    for (uint32_t w = 0; w < full_words; ++w) full[w] = 0xffffffffu;
    check(strata::grammar::mask_to_bias(full.data(), full_words, base, banned, out.data(), nv), "a free step draws");
    bool every = true;
    for (int64_t v = 0; v < nv; ++v) every = every && out[(size_t) v] == base[(size_t) v];
    check(every, "a free step leaves the request's bias exactly as it was");

    std::puts("mask_bias_test: passed, including a full ban and a short bitmap");
}
