// The pack's tokenizer as llguidance wants it: one byte string per id.
//
// llguidance does not tokenize. It is handed every token as bytes and answers
// "which of these ids may come next", so the vocabulary has to be the pack's
// own: `tokenizer/vocab.json` for the text of each id, `tokenizer/token_type.json`
// for which ids are special. The text in `vocab.json` is written in the GPT-2
// byte alphabet (one printable character per byte), so each character is mapped
// back to the byte it stands for; a special token is its text with a 0xFF byte in
// front, which is how a grammar names it (`<|im_end|>` rather than its id).
//
// `serve/grammar/matcher.py` builds the same table in Python. This is the engine's
// copy, and the two are kept honest by a test that asserts the numbers the Python
// side measured on the gyro_l pack: 248,320 ids, 2,839,470 bytes over all of them,
// and `<|im_end|>` as id 248,046.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata {
namespace grammar {

struct Vocabulary {
    // One entry per id, in id order.
    std::vector<uint32_t> lens;
    // The concatenation of every token's bytes; `lens` splits it.
    std::vector<uint8_t> bytes;
    // Where each id starts in `bytes`: `n + 1` entries, so id `i`'s text is `bytes[offsets[i] .. offsets[i+1])`
    // and the last entry is `bytes.size()`.  `lens` already says how long each id is; this says where it is,
    // which is what a caller that has to look at one token's bytes (the fence tracker) needs in O(1).
    std::vector<uint32_t> offsets;
    // The id the grammar ends a turn on: `<|im_end|>`, which the pack marks special.
    uint32_t eos = 0;
    // Ids whose text is a name the grammar may use, not bytes to be matched.
    uint32_t special_count = 0;
};

// Reads `<dir>/vocab.json` and `<dir>/token_type.json` into `out`.
//
// Every failure is a message naming the file and what was wrong with it: an
// unreadable tokenizer must stop a session before its first token rather than
// silently constrain with the wrong ids.
bool load_vocabulary(const std::string& dir, Vocabulary& out, std::string& err);

}  // namespace grammar
}  // namespace strata
