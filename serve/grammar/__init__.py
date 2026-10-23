"""The reply structure of the chat template, as a grammar the decoder cannot leave.

    skeleton.lark          the structure, checked in, byte-identical to what the exploration proved
    build.py               one request's grammar: the two splices and the entry rule
    tools_fragment.py      the declared tools -> per-tool rules
    entries.py             which entry rule a prompt tail calls for
    matcher.py             the library that turns a grammar into a mask
    pipeline.py            one request -> (entry, spliced text, 16-hex id), the splice cached by shape
    guard.py               the startup assertions that the grammar still matches the pack and the template

During a request: `pipeline` picks the entry rule and splices the text (it needs no library), and the server hands
the engine the text as a file (`grammar=<path>`), which the engine compiles and masks every step with.  So nothing
here compiles anything: `matcher` is the only file that loads the library, and outside the engine its only caller is
`serve/test_grammar_matcher.py`, which checks that the grammar accepts and refuses the right bytes (the engine's own
build of the same table is tested against it by `src/grammar/llguidance_test.cpp`).
"""
