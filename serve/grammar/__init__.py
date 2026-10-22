"""The reply structure of the chat template, as a grammar the decoder cannot leave.

    skeleton.lark          the structure, checked in, byte-identical to what the exploration proved
    build.py               one request's grammar: the two splices and the entry rule
    tools_fragment.py      the declared tools -> per-tool rules
    entries.py             which entry rule a prompt tail calls for
    matcher.py             the library that turns a grammar into a mask
    pipeline.py            one request -> (entry, spliced text, 16-hex id), the splice cached by shape
    pool.py                compiled prototypes by id, cloned for every request
    guard.py               the startup assertions that the grammar still matches the pack and the template

Nothing here runs during a request yet: the engine attachment is a later phase.
"""
