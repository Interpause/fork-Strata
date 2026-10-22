// The dlopen wrapper over llguidance's C API.  See the header for why it is a
// runtime load and why the calls mirror serve/grammar/matcher.py.
//
// The struct declarations are a hand copy of `parser/llguidance.h` at the pinned
// version; the static_asserts below are what makes a copy that drifted a compile
// error instead of a misread field at the first mask.

#include "strata/grammar/llguidance.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace strata {
namespace grammar {
namespace {

// ------------------------------------------------------------------- the C ABI
struct LlgParserLimits {
    size_t max_items_in_row;
    uint64_t initial_lexer_fuel;
    uint64_t step_lexer_fuel;
    size_t step_max_items;
    size_t max_lexer_states;
    size_t max_grammar_size;
    bool precompute_large_lexemes;
    bool verbose_errors;
};

struct LlgTokenizer;

using LlgTokenizeFn = size_t (*)(const void*, const uint8_t*, size_t, uint32_t*, size_t);

struct LlgConstraintInit {
    const LlgTokenizer* tokenizer;
    uint32_t log_buffer_level;
    uint32_t log_stderr_level;
    bool ff_tokens_ok;
    bool backtrack_ok;
    LlgParserLimits limits;
};

struct LlgTokenizerInit {
    uint32_t vocab_size;
    uint32_t tok_eos;
    const uint32_t* token_lens;
    const uint8_t* token_bytes;
    const char* tokenizer_json;
    bool tokenize_assumes_string;
    LlgTokenizeFn tokenize_fn;
    bool use_approximate_greedy_tokenize_fn;
    const void* tokenize_user_data;
    const char* const* slices;
};

static_assert(sizeof(LlgParserLimits) == 8 * 6 + 2 + 6, "LlgParserLimits must match parser/llguidance.h");
static_assert(sizeof(LlgConstraintInit) == 8 + 4 + 4 + 1 + 1 + sizeof(LlgParserLimits) + 6,
              "LlgConstraintInit must match parser/llguidance.h");
static_assert(sizeof(LlgTokenizerInit) == 72, "LlgTokenizerInit must match parser/llguidance.h");

// The entry points, as parser/llguidance.h declares them.
using ConstraintInitSetDefaultsFn = void (*)(LlgConstraintInit*, const LlgTokenizer*);
using NewTokenizerFn = LlgTokenizer* (*)(const LlgTokenizerInit*, char*, size_t);
using FreeTokenizerFn = void (*)(LlgTokenizer*);
using NewMatcherFn = void* (*)(const LlgConstraintInit*, const char*, const char*);
using CloneMatcherFn = void* (*)(const void*);
using FreeMatcherFn = void (*)(void*);
using MatcherIsErrorFn = bool (*)(const void*);
using MatcherGetErrorFn = const char* (*)(void*);
using MatcherComputeMaskFn = int32_t (*)(void*);
using MatcherGetMaskFn = const uint32_t* (*)(void*);
using MatcherGetMaskByteSizeFn = size_t (*)(const void*);
using MatcherConsumeTokenFn = int32_t (*)(void*, uint32_t);
using MatcherIsAcceptingFn = bool (*)(void*);
using MatcherIsStoppedFn = bool (*)(const void*);
using GetVersionFn = const char* (*)();

struct Api {
    bool ok = false;
    std::string path;
    std::string version;
    std::string failure;

    ConstraintInitSetDefaultsFn set_defaults = nullptr;
    NewTokenizerFn new_tokenizer = nullptr;
    FreeTokenizerFn free_tokenizer = nullptr;
    NewMatcherFn new_matcher = nullptr;
    CloneMatcherFn clone_matcher = nullptr;
    FreeMatcherFn free_matcher = nullptr;
    MatcherIsErrorFn is_error = nullptr;
    MatcherGetErrorFn get_error = nullptr;
    MatcherComputeMaskFn compute_mask = nullptr;
    MatcherGetMaskFn get_mask = nullptr;
    MatcherGetMaskByteSizeFn mask_byte_size = nullptr;
    MatcherConsumeTokenFn consume_token = nullptr;
    MatcherIsAcceptingFn is_accepting = nullptr;
    MatcherIsStoppedFn is_stopped = nullptr;
};

// One engine process loads one library, once, before it prints READY.
void load_api(Api& a);

Api& api() {
    static Api a;
    static std::once_flag once;
    std::call_once(once, [&] { load_api(a); });
    return a;
}

// ----------------------------------------------------------------- search order
std::string own_folder() {
#if defined(_WIN32)
    char p[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, MAX_PATH);
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#endif
}

// The environment first, then beside the engine, then the checkout's build tree -
// the same three places the server looks, so one checkout needs one file.
std::vector<std::string> candidates() {
    std::vector<std::string> out;
    const char* named = std::getenv(kLibraryEnv);
    if (named != nullptr && named[0] != '\0') out.push_back(named);
    const std::string folder = own_folder();
    if (!folder.empty()) {
        out.push_back(folder + kLibraryName);
        out.push_back(folder + "third_party/llguidance/target/release/" + kLibraryName);
        out.push_back(folder + "../third_party/llguidance/target/release/" + kLibraryName);
    }
    out.push_back(std::string("third_party/llguidance/target/release/") + kLibraryName);
    return out;
}

void* open_library(const std::string& path) {
#if defined(_WIN32)
    return (void*) LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress((HMODULE) handle, name));
#else
    return dlsym(handle, name);
#endif
}

// Resolves every entry point into `a`, or records why it could not.  A missing symbol
// is a library from another version, so the message names both versions.
void load_api(Api& a) {
    const std::vector<std::string> paths = candidates();
    std::string tried;
    void* handle = nullptr;
    for (const std::string& p : paths) {
        handle = open_library(p);
        if (handle != nullptr) {
            a.path = p;
            break;
        }
        if (!tried.empty()) tried += ", ";
        tried += p;
    }
    if (handle == nullptr) {
        a.failure = std::string("cannot load ") + kLibraryName + " (tried " + tried +
                    "): run tools/build_llguidance.sh, or set " + kLibraryEnv;
        return;
    }
    const auto version = (GetVersionFn) symbol(handle, "llg_get_version");
    if (version == nullptr) {
        a.failure = a.path + ": has no llg_get_version - not an llguidance library";
        return;
    }
    const char* v = version();
    a.version = v != nullptr ? v : "";
    if (a.version.find(kPinnedVersion) == std::string::npos) {
        a.failure = a.path + ": is " + a.version + ", this engine is built for llguidance " + kPinnedVersion +
                    " (see tools/build_llguidance.sh)";
        return;
    }
    a.set_defaults = (ConstraintInitSetDefaultsFn) symbol(handle, "llg_constraint_init_set_defaults");
    a.new_tokenizer = (NewTokenizerFn) symbol(handle, "llg_new_tokenizer");
    a.free_tokenizer = (FreeTokenizerFn) symbol(handle, "llg_free_tokenizer");
    a.new_matcher = (NewMatcherFn) symbol(handle, "llg_new_matcher");
    a.clone_matcher = (CloneMatcherFn) symbol(handle, "llg_clone_matcher");
    a.free_matcher = (FreeMatcherFn) symbol(handle, "llg_free_matcher");
    a.is_error = (MatcherIsErrorFn) symbol(handle, "llg_matcher_is_error");
    a.get_error = (MatcherGetErrorFn) symbol(handle, "llg_matcher_get_error");
    a.compute_mask = (MatcherComputeMaskFn) symbol(handle, "llg_matcher_compute_mask");
    a.get_mask = (MatcherGetMaskFn) symbol(handle, "llg_matcher_get_mask");
    a.mask_byte_size = (MatcherGetMaskByteSizeFn) symbol(handle, "llg_matcher_get_mask_byte_size");
    a.consume_token = (MatcherConsumeTokenFn) symbol(handle, "llg_matcher_consume_token");
    a.is_accepting = (MatcherIsAcceptingFn) symbol(handle, "llg_matcher_is_accepting");
    a.is_stopped = (MatcherIsStoppedFn) symbol(handle, "llg_matcher_is_stopped");
    if (a.set_defaults == nullptr || a.new_tokenizer == nullptr || a.free_tokenizer == nullptr ||
        a.new_matcher == nullptr || a.clone_matcher == nullptr || a.free_matcher == nullptr ||
        a.is_error == nullptr || a.get_error == nullptr || a.compute_mask == nullptr || a.get_mask == nullptr ||
        a.mask_byte_size == nullptr || a.consume_token == nullptr || a.is_accepting == nullptr ||
        a.is_stopped == nullptr) {
        a.failure = a.path + ": is llguidance " + a.version + " but is missing entry points this engine calls - "
                    "rebuild it with tools/build_llguidance.sh";
        return;
    }
    a.ok = true;
}

}  // namespace

// ---------------------------------------------------------------------- Session
Session::~Session() {
    if (tokenizer_ != nullptr) api().free_tokenizer((LlgTokenizer*) tokenizer_);
    tokenizer_ = nullptr;
}

bool Session::open(const std::string& tokenizer_dir, std::string& err) {
    failure_.clear();
    if (!api().ok) {
        failure_ = api().failure;
        err = failure_;
        return false;
    }
    library_path_ = api().path;
    version_ = api().version;
    if (!load_vocabulary(tokenizer_dir, vocab_, err)) {
        failure_ = err;
        return false;
    }
    if (vocab_.lens.size() > 0xFFFFFFFFull) {
        err = tokenizer_dir + ": vocabulary too large";
        failure_ = err;
        return false;
    }
    LlgTokenizerInit ti{};
    ti.vocab_size = (uint32_t) vocab_.lens.size();
    ti.tok_eos = vocab_.eos;
    ti.token_lens = vocab_.lens.data();
    ti.token_bytes = vocab_.bytes.data();
    ti.tokenizer_json = nullptr;
    ti.tokenize_assumes_string = false;
    // No fast-forward yet, so no tokenizer callback: the C API accepts a null one
    // only together with the approximate greedy flag (HANDOVER.md section 6).
    ti.tokenize_fn = nullptr;
    ti.use_approximate_greedy_tokenize_fn = true;
    ti.tokenize_user_data = nullptr;
    ti.slices = nullptr;
    char message[4096] = {};
    tokenizer_ = api().new_tokenizer(&ti, message, sizeof(message));
    if (tokenizer_ == nullptr) {
        failure_ = std::string("llg_new_tokenizer failed: ") + message;
        err = failure_;
        return false;
    }
    return true;
}

Match Session::compile(const std::string& grammar, std::string& err, const std::string& kind) {
    if (!ready()) {
        err = failure_.empty() ? std::string("the grammar library is not loaded") : failure_;
        return Match();
    }
    LlgConstraintInit ci{};
    api().set_defaults(&ci, (const LlgTokenizer*) tokenizer_);
    ci.log_stderr_level = 0;         // a refused draft token is normal, not something to print
    ci.log_buffer_level = 1;         // ...but keep the message, so error() can say what failed
    ci.limits.verbose_errors = false;  // without the grammar and the token history in every message
    void* ptr = api().new_matcher(&ci, kind.c_str(), grammar.c_str());
    if (ptr == nullptr) {
        err = "llg_new_matcher returned null";
        return Match();
    }
    // A grammar that does not compile still comes back as a matcher, carrying the
    // compiler's message - that message is the useful part, so it is copied into
    // `err` and the matcher is dropped rather than handed on in an error state.
    Match m(ptr);
    if (api().is_error(ptr)) {
        err = m.error();
        if (err.empty()) err = "the grammar did not compile";
        return Match();
    }
    return m;
}

// ------------------------------------------------------------------------ Match
Match::~Match() {
    if (ptr_ != nullptr) api().free_matcher(ptr_);
    ptr_ = nullptr;
}

Match::Match(Match&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

Match& Match::operator=(Match&& other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) api().free_matcher(ptr_);
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
    }
    return *this;
}

std::string Match::error() const {
    if (ptr_ == nullptr) return {};
    const char* e = api().get_error(ptr_);
    return e != nullptr ? std::string(e) : std::string();
}

const uint32_t* Match::mask(uint32_t& words) {
    words = 0;
    if (ptr_ == nullptr) return nullptr;
    if (api().compute_mask(ptr_) != 0) return nullptr;
    const size_t bytes = api().mask_byte_size(ptr_);
    const uint32_t* m = api().get_mask(ptr_);
    if (m == nullptr || bytes == 0) return nullptr;
    words = (uint32_t) (bytes / sizeof(uint32_t));
    return m;
}

bool Match::allows(uint32_t id) {
    uint32_t words = 0;
    const uint32_t* m = mask(words);
    if (m == nullptr) return false;
    const uint32_t word = id / 32;
    if (word >= words) return false;
    return ((m[word] >> (id % 32)) & 1u) != 0;
}

bool Match::consume(uint32_t id) {
    if (ptr_ == nullptr) return false;
    return api().consume_token(ptr_, id) == 0;
}

Match Match::clone() const {
    if (ptr_ == nullptr) return Match();
    return Match(api().clone_matcher(ptr_));
}

bool Match::accepting() const { return ptr_ != nullptr && api().is_accepting(ptr_); }
bool Match::stopped() const { return ptr_ != nullptr && api().is_stopped(ptr_); }

}  // namespace grammar
}  // namespace strata
