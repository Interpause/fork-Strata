// The Markdown fence tracker the grammar's delimiter ban reads (grammar/fence.hpp): where a fence opens, where it
// closes, and which bytes inside one are content.  Host-only and header-only, so no GPU, no model and no grammar
// library are needed - which is why this is here and not in src/grammar/llguidance_test.cpp.
#include "strata/grammar/fence.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using strata::grammar::FenceTracker;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "fence regression: %s\n", what);
        std::exit(1);
    }
}

// The tracker after one token's worth of bytes, so a case is the byte string and the answer it must give.
FenceTracker tracker(const char* bytes) {
    FenceTracker f;
    f.append(reinterpret_cast<const uint8_t*>(bytes), std::strlen(bytes));
    return f;
}

void expect(const char* bytes, bool want, const char* what) {
    check(tracker(bytes).open() == want, what);
}

}   // namespace

int main() {
    // 1. Plain prose never opens a fence, and reset() is a clean start.
    expect("", false, "an empty reply has no fence");
    expect("the answer is 42\nand that is all\n", false, "prose with newlines is prose");
    expect("a `single` backtick pair is inline code\n", false, "inline code is not a fence");
    {
        FenceTracker f = tracker("```\n");
        check(f.open(), "the fence was open before reset");
        f.reset();
        check(!f.open(), "reset() closes the fence");
        check(f.state() == FenceTracker::State::LineIndent0, "reset() is at a line start");
    }

    // 2. Three fence characters at a line start open; fewer, or three that start mid-line, do not.
    expect("`", false, "one backtick does not open");
    expect("``", false, "two backticks do not open");
    expect("`x", false, "one backtick and text does not open");
    expect("```", true, "three backticks open, with no newline needed");
    expect("```\n", true, "three backticks and a newline open");
    expect("~~~~\n", true, "a four-tilde opener opens");
    expect("~~\n", false, "two tildes do not open");
    expect("x ```\n", false, "three backticks after text do not open");
    expect("   ```\n", true, "three spaces of indent may start a fence");
    expect("    ```\n", false, "four spaces of indent is a code block, not a fence");
    expect("\t```\n", false, "a tab at line start is not fence indent");

    // 3. A short run only spoils its own line: the next line may still open a fence.
    expect("``\n```\n", true, "a short run on an earlier line does not stop the next line opening");
    expect("x``\n```\n", true, "a short run mid-line does not stop the next line opening");

    // 4. The opener's own line is inside: everything after its run is an info string, whatever it holds.
    check(tracker("```").state() == FenceTracker::State::Info, "the third backtick reaches the opener's line");
    expect("```python\n", true, "a language tag is on the opener's line");
    expect("```<tool_call>\n", true, "a structural tag in the info string opens nothing else");
    expect("```foo``` \n", true, "a backtick in the info string still opens (a deliberate departure)");
    check(tracker("```python\n").state() == FenceTracker::State::InsideLineIndent0,
          "the line after an opener starts inside the fence");

    // 5. A closer is three of the opening character at a line start, whatever the opening run's length is and
    //    whatever follows it on its line.  Three of the other character are content.
    expect("```\na\n```\n", false, "a matching closer at line start closes");
    expect("```\na\n```", false, "the closer needs no newline after it");
    expect("````\na\n```\n", false, "a four-backtick opener closes on three (a deliberate departure)");
    expect("~~~\na\n~~~\n", false, "a tilde fence closes on three tildes");
    expect("~~~\na\n```\n", true, "a tilde fence is not closed by backticks");
    expect("```\na\n~~~\n", true, "a backtick fence is not closed by tildes");
    expect("   ```\na\n   ```\n", false, "indented fences close on an indented line start");
    expect("```\na ``` b\n", true, "a mid-line triple inside a fence is content");
    expect("```\na\n``\n```\n", false, "a short run inside a fence does not close it");
    expect("```\na\n```x", false, "the closer's own line cannot start a new fence");
    expect("```\na\n```x\n```\nb\n```\n", false, "a later line opens a fresh fence after an earlier close");

    // 6. `\r` is an ordinary byte, so CRLF line endings behave like LF ones.
    expect("```\r\n", true, "a CRLF line ending opens like LF");
    expect("```\r\na\r\n```\r\n", false, "a CRLF closer closes");
    expect("```\na\r\n```\n", false, "a CRLF line inside the fence does not hide the closer");
    expect("```\r\n```\r", false, "a CRLF closer with no final newline closes");

    // 7. Byte 0xFF, the prefix on every special token, is "other": it can neither start nor continue a run.
    {
        const uint8_t after_special[] = {0xFF, '`', '`', '`', '\n'};
        FenceTracker f;
        f.append(after_special, sizeof after_special);
        check(!f.open(), "0xFF before three backticks does not open a fence");
    }
    {
        FenceTracker f = tracker("```\n");
        const uint8_t special_then_ticks[] = {0xFF, '`', '`', '`', '\n'};
        f.append(special_then_ticks, sizeof special_then_ticks);
        check(f.open(), "0xFF at a line start inside a fence does not let a closer through");
    }

    // 8. An unterminated fence stays open through arbitrary text, newlines included.
    expect("```\nno closer here\n\nstill open\njust text", true,
           "an unterminated fence stays open to the end of the reply");

    // 9. A token boundary is not a line boundary: the same bytes in two appends give the same answer.
    {
        FenceTracker f;
        f.append(reinterpret_cast<const uint8_t*>("``"), 2);
        check(!f.open(), "two of an opener's three characters do not open");
        f.append(reinterpret_cast<const uint8_t*>("`\n"), 2);
        check(f.open(), "the third character opens across an append boundary");
    }

    std::puts("fence_test: passed, from an opener's third character to a closer's third character");
}
