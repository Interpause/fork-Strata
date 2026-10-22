// The engine's side of the grammar: llguidance loads at runtime, compiles a Lark
// grammar once, and answers "which ids may come next" for every decoding step.
//
// Why a runtime load and not a linked archive: `setup.py` compiles the engine on
// the user's PC, where there is no Rust toolchain, so a linked `libllguidance.a`
// would have to be shipped prebuilt - 72 MB in every archive. The shared object is
// 4.5 MB, needs nothing at link time, and the repository already delivers a
// companion `.so` this way (src/prefill/mmq_kq_loader.cpp, with an ABI handshake).
// The symbols are resolved once, before the engine prints READY, so a missing or
// mismatched library is still discovered before the first request: the engine then
// reports no `grammar=` capability and refuses a request that asks for one.
//
// The C calls below mirror `serve/grammar/matcher.py` exactly - same struct order,
// same functions, same flags - so a mask the server has accepted is a mask the
// engine has already been proven able to produce. `llg_*` is pinned with the
// vendored checkout; another version is not silently accepted.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/grammar/vocabulary.hpp"

namespace strata {
namespace grammar {

// The version this checkout is pinned to (third_party/llguidance, tools/build_llguidance.sh).
inline constexpr const char* kPinnedVersion = "1.9.1";
// The file the runtime search looks for, beside the engine first.
inline constexpr const char* kLibraryName = "libllguidance.so";
// The environment variable that names another copy, the same one the server reads.
inline constexpr const char* kLibraryEnv = "STRATA_LLGUIDANCE_LIB";

// One decoding state.  A request owns one; a draft is validated on a clone so that a
// refusal cannot move the state the accepted tokens produced.
class Match {
  public:
    Match() = default;
    ~Match();
    Match(Match&& other) noexcept;
    Match& operator=(Match&& other) noexcept;
    Match(const Match&) = delete;
    Match& operator=(const Match&) = delete;

    bool valid() const { return ptr_ != nullptr; }
    // The library's own message for a failed compile or a refused token; empty when
    // nothing has gone wrong.
    std::string error() const;

    // The ids allowed at this step, one bit per id, `words` words of 32.  The array
    // stays valid until the next call on this matcher.  Null when the mask could not
    // be computed, in which case `error()` says why and the step must not be sampled.
    const uint32_t* mask(uint32_t& words);
    // True when `id` is allowed at this step (a mask call is made if needed).
    bool allows(uint32_t id);

    // Advances the state by one committed token.  False when the grammar refuses it,
    // which is a bug in the mask or in the caller's ordering, never a legal draw.
    bool consume(uint32_t id);
    // A copy with its own state, for validating a draw without committing to it.
    Match clone() const;

    bool accepting() const;  // the grammar may end here
    bool stopped() const;    // the grammar has ended

  private:
    friend class Session;
    explicit Match(void* ptr) : ptr_(ptr) {}
    void* ptr_ = nullptr;
};

// The library, its tokenizer and its vocabulary: one per engine process, built once
// from the pack's tokenizer directory.
class Session {
  public:
    Session() = default;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // Loads the library and builds the vocabulary from `tokenizer_dir`.  On failure
    // `err` says what was missing or wrong, and the session stays unusable -
    // `compile()` then fails with the same message rather than constraining with
    // the wrong ids.
    bool open(const std::string& tokenizer_dir, std::string& err);

    bool ready() const { return tokenizer_ != nullptr && failure_.empty(); }
    const std::string& failure() const { return failure_; }
    const std::string& library_path() const { return library_path_; }
    std::string library_version() const { return version_; }

    uint32_t vocab_size() const { return vocab_.lens.empty() ? 0u : (uint32_t) vocab_.lens.size(); }
    uint32_t eos() const { return vocab_.eos; }

    // Compiles `grammar` (Lark text, as the server splices it) into a fresh state.
    // `kind` is llguidance's grammar format name; only "lark" is used today.
    Match compile(const std::string& grammar, std::string& err, const std::string& kind = "lark");

  private:
    Vocabulary vocab_;
    std::string library_path_;
    std::string version_;
    std::string failure_;
    void* tokenizer_ = nullptr;
};

}  // namespace grammar
}  // namespace strata
