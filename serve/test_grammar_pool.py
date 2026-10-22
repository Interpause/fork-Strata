"""The prototype pool: one compile per grammar, a clone per request, a bound.

    python -m unittest serve.test_grammar_pool -v

The compile half needs `libllguidance.so` and a pack vocabulary and is skipped, with the reason, without them; the
"disabled" half runs anywhere and is the one that must not lie (a pool that silently pretends to work is worse than
no pool).
"""
import unittest
from pathlib import Path

from serve.grammar import build as bg
from serve.grammar import pool as pl

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = bg.for_entry("entry_answer_plain")


class Disabled(unittest.TestCase):
    def test_no_tokenizer_directory_disables_the_pool_and_says_why(self):
        said = []
        p = pl.Pool(None, None, log=said.append)
        self.assertIsNone(p.acquire("abc", GRAMMAR))
        self.assertTrue(p.disabled)
        self.assertEqual(len(said), 1)
        self.assertIn("build_llguidance.sh", said[0])
        self.assertEqual(p.stats()["prototypes"], 0)

    def test_a_missing_directory_disables_the_pool(self):
        p = pl.Pool(ROOT / "tmp" / "no-such-pack" / "tokenizer", None, log=lambda m: None)
        self.assertIsNone(p.acquire("abc", GRAMMAR))
        self.assertTrue(p.disabled)

    def test_release_of_nothing_is_safe(self):
        pl.Pool.release(None)


def vocabulary():
    from serve.test_grammar_matcher import pack_dir
    from serve.grammar import matcher as M
    directory = pack_dir()
    if directory is None:
        raise unittest.SkipTest("no pack tokenizer (set STRATA_PACK_TOKENIZER_DIR)")
    try:
        return directory, M.vocabulary(directory)
    except M.LibraryMissing as e:
        raise unittest.SkipTest(str(e))


class Prototypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir, cls.v = vocabulary()

    def pool(self, **kw):
        said = []
        p = pl.Pool(self.dir, None, log=said.append)
        p._vocab = self.v                      # the test's own vocabulary: building it is the slow part
        p.said = said
        return p

    def test_two_requests_share_one_compile(self):
        p = self.pool()
        a = p.acquire("g", GRAMMAR)
        b = p.acquire("g", GRAMMAR)
        try:
            self.assertIsNot(a, b)                     # never the same live matcher
            self.assertEqual((p.stats()["hits"], p.stats()["misses"], p.stats()["prototypes"]), (1, 1, 1))
        finally:
            pl.Pool.release(a)
            pl.Pool.release(b)
            p.close()

    def test_a_clone_advances_alone(self):
        p = self.pool()
        a = p.acquire("g", GRAMMAR)
        first, _ = a.mask()
        self.assertTrue(a.consume(self.v.ids("abcdef")))
        b = p.acquire("g", GRAMMAR)
        try:
            self.assertEqual(b.mask()[0], first)        # the prototype did not move
            self.assertNotEqual(a.mask()[0], first)
        finally:
            pl.Pool.release(a)
            pl.Pool.release(b)
            p.close()

    def test_the_bound_evicts_and_recompiles(self):
        p = self.pool()
        p.max = 1
        a = p.acquire("g1", GRAMMAR)
        pl.Pool.release(a)
        b = p.acquire("g2", GRAMMAR)
        pl.Pool.release(b)
        self.assertEqual(p.stats()["evictions"], 1)
        self.assertEqual(p.stats()["prototypes"], 1)
        c = p.acquire("g1", GRAMMAR)
        pl.Pool.release(c)
        self.assertEqual(p.stats()["misses"], 3)

    def test_release_frees_the_matcher(self):
        p = self.pool()
        a = p.acquire("g", GRAMMAR)
        pl.Pool.release(a)
        self.assertIsNone(a.ptr)
        p.close()

    def test_close_frees_every_prototype(self):
        p = self.pool()
        pl.Pool.release(p.acquire("g1", GRAMMAR))
        pl.Pool.release(p.acquire("g2", GRAMMAR))
        p.close()
        self.assertEqual(p.stats()["prototypes"], 0)


if __name__ == "__main__":
    unittest.main()
