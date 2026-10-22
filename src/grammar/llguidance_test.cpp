// src/grammar/llguidance_test.cpp - the vocabulary reader and the dlopen wrapper,
// without a GPU and without a model file.  Both halves skip, with a printed reason,
// when their input is missing: a runner without the pack still checks the failure
// messages, and a runner without libllguidance still checks the vocabulary.
//
// The numbers below were measured on the gyro_l pack by
// `serve/grammar/matcher.py`'s own build, so this is the test that keeps the engine's
// copy of the byte alphabet from drifting away from the server's.
#include "strata/grammar/llguidance.hpp"
#include "strata/grammar/vocabulary.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int fails = 0;
static int skips = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
#define SKIP(what, why) do { std::printf("SKIP %s: %s\n", (what), (why)); ++skips; } while (0)

namespace {

const char* pack_dir() {
    const char* named = std::getenv("STRATA_PACK_TOKENIZER_DIR");
    if (named != nullptr && named[0] != '\0') return named;
    static const char* kFallbacks[] = {"../Strata-data/packs/gyro_l/tokenizer", "../../Strata-data/packs/gyro_l/tokenizer",
                                       "packs/gyro_l/tokenizer"};
    for (const char* p : kFallbacks) {
        std::string vocab = std::string(p) + "/vocab.json";
        if (std::FILE* f = std::fopen(vocab.c_str(), "rb")) {
            std::fclose(f);
            return p;
        }
    }
    return nullptr;
}

// The id whose byte string is exactly `text`, or -1.  Only ASCII, which is all this
// test needs; the byte alphabet maps ASCII printable characters to themselves.
int id_of(const strata::grammar::Vocabulary& v, const char* text) {
    const size_t n = std::strlen(text);
    for (size_t i = 0; i < v.lens.size(); ++i) {
        if (v.lens[i] != n) continue;
        size_t at = 0;
        for (size_t j = 0; j < i; ++j) at += v.lens[j];
        if (std::memcmp(v.bytes.data() + at, text, n) == 0) return (int) i;
    }
    return -1;
}

uint8_t first_byte(const strata::grammar::Vocabulary& v, size_t id) {
    size_t at = 0;
    for (size_t j = 0; j < id; ++j) at += v.lens[j];
    return v.lens[id] == 0 ? 0 : v.bytes[at];
}

}  // namespace

int main() {
    using namespace strata::grammar;

    // ---------------------------------------------------------------- vocabulary
    const char* dir = pack_dir();
    Vocabulary vocab;
    bool have_vocab = false;
    if (dir == nullptr) {
        SKIP("vocabulary", "no pack: set STRATA_PACK_TOKENIZER_DIR (or place the pack at ../Strata-data/packs)");
    } else {
        std::string err;
        have_vocab = load_vocabulary(dir, vocab, err);
        if (!have_vocab) {
            // A pack the runner named and that does not load is a failure; a pack
            // that is simply not on this machine is a skip.
            if (std::getenv("STRATA_PACK_TOKENIZER_DIR") != nullptr) {
                std::printf("FAIL line %d: STRATA_PACK_TOKENIZER_DIR did not load: %s\n", __LINE__, err.c_str());
                ++fails;
            } else {
                SKIP("vocabulary", err.c_str());
            }
        }
    }
    if (have_vocab) {
        // The numbers the Python build measured on gyro_l.  1,846,190 is the sum of
        // every token's byte length and is what proves the byte alphabet is right;
        // the 33 are the control/user-defined ids that get the 0xFF prefix.
        CHECK(vocab.lens.size() == 248320);
        CHECK(vocab.eos == 248046);
        CHECK(vocab.special_count == 33);
        size_t total = 0;
        for (uint32_t n : vocab.lens) total += n;
        CHECK(total == 1846190);
        CHECK(!vocab.bytes.empty() && vocab.bytes.size() == total);
        // A printed ASCII token stands for itself, so "!" is byte 0x21 at id 0,
        // "a" is 0x61 at id 64 and "ab" is two bytes at id 365.
        CHECK(vocab.lens[0] == 1 && vocab.bytes[0] == 0x21);
        CHECK(id_of(vocab, "!") == 0);
        CHECK(id_of(vocab, "a") == 64);
        CHECK(id_of(vocab, "ab") == 365);
        // A byte the alphabet had to move is one byte here too: the space token's
        // text is U+0120 and its byte is 0x20, at id 220.
        CHECK(vocab.lens[220] == 1);
        CHECK(id_of(vocab, " ") == 220);
        // `<|im_end|>` is special: its bytes are 0xFF and then its own text.
        size_t at = 0;
        for (size_t j = 0; j < vocab.eos; ++j) at += vocab.lens[j];
        CHECK(vocab.lens[vocab.eos] == 1 + 10);
        CHECK(vocab.bytes[at] == 0xFF);
        CHECK(std::memcmp(vocab.bytes.data() + at + 1, "<|im_end|>", 10) == 0);
    }

    // A directory without a tokenizer is a message naming the file, not a crash.
    {
        std::string err;
        Vocabulary none;
        CHECK(!load_vocabulary("/nonexistent/strata-tokenizer", none, err));
        CHECK(err.find("vocab.json") != std::string::npos);
    }

    // ------------------------------------------------------------------- session
    Session session;
    std::string err;
    if (!session.open(dir != nullptr ? dir : "/nonexistent/strata-tokenizer", err)) {
        // Either the library or the pack is missing; both are skips, and the message
        // says which.  The failure is still a message, not a crash.
        CHECK(!err.empty());
        // An attempted compile on a failed session repeats the reason.
        std::string again;
        Match m = session.compile("start: \"ab\"", again);
        CHECK(!m.valid());
        CHECK(!again.empty());
        SKIP("mask", err.c_str());
    } else {
        std::printf("     llguidance %s at %s\n", session.library_version().c_str(), session.library_path().c_str());
        CHECK(session.vocab_size() == vocab.lens.size());
        CHECK(session.eos() == vocab.eos);

        // A grammar of two characters: the mask must hold exactly the ids whose bytes
        // begin with 'a', so an id starting with anything else is refused.
        Match m = session.compile("start: \"ab\"", err);
        CHECK(m.valid());
        CHECK(err.empty());
        if (m.valid()) {
            uint32_t words = 0;
            const uint32_t* bits = m.mask(words);
            CHECK(bits != nullptr);
            CHECK(words == (vocab.lens.size() + 31) / 32);
            size_t allowed = 0;
            for (size_t id = 0; id < vocab.lens.size(); ++id) {
                const bool ok = ((bits[id / 32] >> (id % 32)) & 1u) != 0;
                if (!ok) continue;
                ++allowed;
                CHECK(vocab.lens[id] > 0 && first_byte(vocab, id) == 'a');
            }
            CHECK(allowed > 0);
            CHECK(m.allows(64));     // "a"
            CHECK(!m.allows(65));    // "b"
            CHECK(!m.allows(220));   // the space byte
            CHECK(!m.allows(vocab.eos));
            CHECK(!m.accepting());   // "ab" is not complete yet

            // A clone is a copy of the state, not a handle on it.
            Match copy = m.clone();
            CHECK(copy.valid());
            CHECK(m.consume(64));    // "a"
            // After "a" only ids whose remaining bytes begin with 'b' may follow.
            const uint32_t* next = m.mask(words);
            CHECK(next != nullptr);
            for (size_t id = 0; id < vocab.lens.size(); ++id) {
                if (((next[id / 32] >> (id % 32)) & 1u) == 0) continue;
                CHECK(first_byte(vocab, id) == 'b');
            }
            CHECK(m.allows(65));     // "b"
            CHECK(!m.allows(64));    // "a" again
            // The clone still expects the 'a' it has not seen.
            CHECK(copy.allows(64));
            CHECK(!copy.allows(65));
            CHECK(m.consume(65));    // "b"
            CHECK(m.accepting());
            CHECK(m.stopped());      // the whole grammar has been generated
            CHECK(m.allows(vocab.eos));  // ...so the turn may end here
            CHECK(copy.consume(64));
            CHECK(!copy.allows(365));  // "ab" is two characters, the grammar wants one more
        }

        // A token the grammar refuses: the library says so instead of moving on.
        Match r = session.compile("start: \"ab\"", err);
        CHECK(r.valid());
        CHECK(!r.consume(65));
        CHECK(!r.error().empty());

        // A grammar that does not compile carries the compiler's message.
        Match bad = session.compile("start: (this is not lark", err);
        CHECK(!bad.valid());
        CHECK(!err.empty());
        if (!err.empty()) std::printf("     bad grammar: %s\n", err.c_str());

        // Two states from one session are independent.
        Match one = session.compile("start: \"ab\"", err);
        Match two = session.compile("start: \"ab\"", err);
        CHECK(one.valid() && two.valid());
        CHECK(one.consume(64));
        CHECK(two.allows(64));
        CHECK(!one.allows(64));
    }

    std::printf("%s: %d failed, %d skipped\n", fails == 0 ? "PASS" : "FAIL", fails, skips);
    return fails == 0 ? 0 : 1;
}
