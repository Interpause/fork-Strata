// A grammar's answer - "which of these ids may come next" - as the one thing the sampler reads: a dense
// float[n_vocab] per step.
//
// The matcher answers with a bitmap (`Match::mask`): one bit per id, set when the id is allowed.  The sampler takes
// `SamplerParams::logit_bias`, a dense row where an allowed id keeps its score and a refused one is -inf.  This is the
// conversion, and it is deliberately one function used by BOTH decode paths: the single-request step and a batch
// window's row.  Two copies would be two chances to compose the mask, the request's own `logit_bias` and the
// engine's banned ids in a different order, which is how a constrained reply starts differing from itself depending
// on how many other requests happened to be running.
//
// Host-only and header-only: no CUDA, no matcher, no library.  What it converts is a bitmap the caller owns, so it is
// testable without a GPU, a model or llguidance (src/core/mask_bias_test.cpp).
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace strata::grammar {

/// Fill `out[0..n_vocab)` from a matcher mask.
///
/// An allowed id keeps `base[v]` - the request's own `logit_bias`, empty meaning "no bias" - and a refused id is set
/// to -inf, so the sampler can still draw any allowed id at its own probability while a refused one becomes
/// impossible.  Ids past the bitmap's last word are refused: the matcher's bitmap is allowed to be shorter than the
/// vocabulary, and a missing bit means "no".
///
/// `banned` holds ids the mask allowed and the matcher then refused - a library bug - which the engine bans for the
/// slot's next step instead of drawing them again.  They are refused here too, and they do not count as drawable: an
/// id that is both allowed by the bitmap and banned must not make the row look non-empty, because an all -inf row
/// handed to the sampler makes the greedy kernel emit token 0 (a real token) rather than nothing.  The same holds for
/// an allowed id whose own `base` entry is -inf (a client's hard exclusion).
///
/// Returns false when nothing is drawable.  Such a row must NOT be handed to the sampler at all: the caller ends the
/// slot, because "no token is legal here" is an answer the protocol expresses as a stop, not as token 0.
///
/// `banned` is expected to be a handful of ids (it is only ever filled after a draw the matcher refused), and the
/// loop below touches each id once, so this stays O(n_vocab + banned) with no allocation.
inline bool mask_to_bias(const uint32_t* bits, uint32_t words, const std::vector<float>& base,
                         const std::vector<int32_t>& banned, float* out, int64_t n_vocab) {
    if (out == nullptr || n_vocab <= 0) return false;
    const float ninf = -std::numeric_limits<float>::infinity();
    const bool have_base = !base.empty() && (int64_t) base.size() == n_vocab;
    int64_t drawable = 0;
    for (int64_t v = 0; v < n_vocab; ++v) {
        const bool allowed = bits != nullptr && v < (int64_t) words * 32 &&
                             ((bits[(size_t) (v >> 5)] >> (v & 31)) & 1u) != 0;
        const float b = allowed ? (have_base ? base[(size_t) v] : 0.0f) : ninf;
        out[(size_t) v] = b;
        if (b != ninf) ++drawable;
    }
    for (const int32_t id : banned) {
        if (id < 0 || id >= n_vocab) continue;
        if (out[(size_t) id] != ninf) {   // still drawable: this ban is what removes it (a duplicate ban is not)
            out[(size_t) id] = ninf;
            --drawable;
        }
    }
    return drawable > 0;
}

}   // namespace strata::grammar
