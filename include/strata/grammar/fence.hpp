// Whether a Markdown fenced code block is open in a reply, from the bytes of the tokens committed so far.
//
// Why the engine needs this at all: inside a fence a structural delimiter such as `<tool_call>` is content, not
// structure, but a byte terminal cannot forbid a *token id* - llguidance's skeleton says special tokens "must never
// be named inside a terminal" - so the ban belongs in the sampler's mask, and the mask needs one fact the grammar
// does not carry: is a fence open right now (D-067, tmp/impl-plan/fence-ban.md).
//
// W1 in the plan: an FSM over bytes rather than a precomputed token-transition table, and the FSM carries all of its
// own state, so there is no byte tail to buffer.  A few bytes per slot and trivially copyable: a batch row that
// validates a draft copies the slot's tracker and advances the copy by the draft's first token.
//
// Recognition is the plan's, with two deliberate departures from CommonMark, both because the rule is "err toward
// re-allowing early": a closing run needs only three of the opening character, however long the opening run was
// (a model that opens with four and closes with three is doing what it thinks is right); and a backtick inside an
// info string still opens, so ` ```foo``` ` on one line is an opener and not inline code.
//
// Ordinary bytes: `\r` is "other", so a `\r\n` line ending works through its `\n`, which returns to a line start;
// a tab at line start is "other" too, so it does not count as fence indent.  Byte 0xFF - the prefix
// `load_vocabulary` puts on every special token - can never start or continue a run, so a special token's bytes can
// never look like fence text.
//
// Host-only and header-only: no CUDA, no matcher, no library (src/grammar/fence_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::grammar {

/// One reply's fenced code blocks, as its committed tokens' bytes arrive.
///
/// `open()` turns true on the third character of an opener (three or more backticks or tildes at the start of a
/// line, after at most three spaces) and false again on the third character of a run of the opening character at
/// the start of a later line.  Everything after an opener's run on its own line is its info string, so the line
/// that holds the language tag is inside the fence, not outside it.
class FenceTracker {
  public:
    enum class State : uint8_t {
        // Outside a fence: at a line start after 0..3 spaces of indent, somewhere else in a line, or holding a run
        // of one or two fence characters at line start that has not reached three.
        LineIndent0,
        LineIndent1,
        LineIndent2,
        LineIndent3,
        LineOther,
        RunTick1,
        RunTick2,
        RunTilde1,
        RunTilde2,
        // Inside a fence: the opener's own line (its info string), and then the same four line-start states and
        // three mid-line ones as outside.  These are declared last, so `open()` is a single comparison.
        Info,
        InsideLineIndent0,
        InsideLineIndent1,
        InsideLineIndent2,
        InsideLineIndent3,
        InsideLineOther,
        InsideRunTick1,
        InsideRunTick2,
        InsideRunTilde1,
        InsideRunTilde2,
    };

    void reset() noexcept {
        state_ = State::LineIndent0;
        opener_ = 0;
    }

    /// Appends one committed token's bytes, in order.
    void append(const uint8_t* bytes, size_t len) noexcept {
        for (size_t i = 0; i < len; ++i) advance(bytes[i]);
    }

    /// True from the third character of an opener until the third character of a closer.
    bool open() const noexcept { return state_ >= State::Info; }

    State state() const noexcept { return state_; }

  private:
    /// The character the open fence opened with.  A closer must be three of that character, and none of the
    /// line-start states carries it, so it lives here.  Nought while outside a fence.
    uint8_t opener_ = 0;
    State state_ = State::LineIndent0;

    /// A newline returns to a line start on whichever side of the fence we are.  A run of one or two fence
    /// characters is spoilt by it like by any other byte, and the next line starts clean.
    void newline() noexcept {
        state_ = open() ? State::InsideLineIndent0 : State::LineIndent0;
    }

    /// The closer's line is outside from its third character on, so a fence cannot reopen on the same line.
    void close() noexcept {
        state_ = State::LineOther;
        opener_ = 0;
    }

    /// One more space at a line start.  A fourth space is four spaces of indent, which is a code block rather than
    /// a fence line, so from the fourth indent the line simply becomes an ordinary one.
    State indent_after_space() const noexcept {
        switch (state_) {
            case State::LineIndent0: return State::LineIndent1;
            case State::LineIndent1: return State::LineIndent2;
            case State::LineIndent2: return State::LineIndent3;
            case State::InsideLineIndent0: return State::InsideLineIndent1;
            case State::InsideLineIndent1: return State::InsideLineIndent2;
            case State::InsideLineIndent2: return State::InsideLineIndent3;
            case State::LineIndent3: return State::LineOther;
            case State::InsideLineIndent3: return State::InsideLineOther;
            default: return state_;   // not at a line start: the caller never asks
        }
    }

    void advance(uint8_t b) noexcept {
        if (b == '\n') {
            newline();
            return;
        }
        switch (state_) {
            // ---- outside a fence
            case State::LineIndent0:
            case State::LineIndent1:
            case State::LineIndent2:
            case State::LineIndent3:
                if (b == ' ') {
                    state_ = indent_after_space();
                    return;
                }
                if (b == '`') {
                    state_ = State::RunTick1;
                    return;
                }
                if (b == '~') {
                    state_ = State::RunTilde1;
                    return;
                }
                // A tab is indented but not by spaces, so this line cannot start a fence.
                state_ = State::LineOther;
                return;
            case State::LineOther:
                return;
            case State::RunTick1:
                state_ = b == '`' ? State::RunTick2 : State::LineOther;
                return;
            case State::RunTick2:
                if (b == '`') {   // the third backtick opens: the rest of this line is the info string
                    opener_ = '`';
                    state_ = State::Info;
                    return;
                }
                state_ = State::LineOther;   // a run of two is not a fence line
                return;
            case State::RunTilde1:
                state_ = b == '~' ? State::RunTilde2 : State::LineOther;
                return;
            case State::RunTilde2:
                if (b == '~') {
                    opener_ = '~';
                    state_ = State::Info;
                    return;
                }
                state_ = State::LineOther;
                return;
            case State::Info:
                return;   // whatever the info string holds, it is still the opener's line
            // ---- inside a fence
            case State::InsideLineIndent0:
            case State::InsideLineIndent1:
            case State::InsideLineIndent2:
            case State::InsideLineIndent3:
                if (b == ' ') {
                    state_ = indent_after_space();
                    return;
                }
                if (b == '`') {
                    state_ = State::InsideRunTick1;
                    return;
                }
                if (b == '~') {
                    state_ = State::InsideRunTilde1;
                    return;
                }
                state_ = State::InsideLineOther;
                return;
            case State::InsideLineOther:
                return;
            case State::InsideRunTick1:
                state_ = b == '`' ? State::InsideRunTick2 : State::InsideLineOther;
                return;
            case State::InsideRunTick2:
                // Three of the opening character close, whatever the opening run's length was; three of the other
                // character are content and leave the fence open.
                if (b == '`' && opener_ == '`') {
                    close();
                    return;
                }
                state_ = State::InsideLineOther;
                return;
            case State::InsideRunTilde1:
                state_ = b == '~' ? State::InsideRunTilde2 : State::InsideLineOther;
                return;
            case State::InsideRunTilde2:
                if (b == '~' && opener_ == '~') {
                    close();
                    return;
                }
                state_ = State::InsideLineOther;
                return;
        }
    }
};

// A batch slot copies its tracker for a draft row and advances the copy by one token, so a copy must stay cheap
// (no allocation, no indirection) and must not be invalidated by the original moving on.
static_assert(std::is_trivially_copyable<FenceTracker>::value, "a slot's fence state is copied, not shared");

}   // namespace strata::grammar
