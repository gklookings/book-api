import unittest
from unittest import mock

from app.langchain import alwaraq
from app.langchain.alwaraq_lib import retrieval


def _p(key, doc="b1", **extra):
    return {"key": key, "document_id": doc, "text": key, **extra}


class RrfFuseTest(unittest.TestCase):
    def test_items_in_multiple_lists_rank_higher(self):
        fused = retrieval.rrf_fuse([[_p("a"), _p("b"), _p("c")], [_p("c"), _p("d")]])
        keys = [p["key"] for p in fused]
        self.assertEqual(keys[0], "c")
        self.assertEqual(set(keys), {"a", "b", "c", "d"})
        self.assertTrue(all("rrf" in p for p in fused))

    def test_weights_let_keyword_list_compete(self):
        vector_lists = [[_p("v"), _p("x")] for _ in range(4)]
        keyword_list = [_p("kw")]
        unweighted = retrieval.rrf_fuse(vector_lists + [keyword_list])
        weighted = retrieval.rrf_fuse(vector_lists + [keyword_list], weights=[1, 1, 1, 1, 5])
        self.assertEqual([p["key"] for p in unweighted][-1], "kw")
        self.assertEqual([p["key"] for p in weighted][0], "kw")

    def test_empty(self):
        self.assertEqual(retrieval.rrf_fuse([]), [])


class ReserveSlotsTest(unittest.TestCase):
    def test_reserved_keyword_hits_replace_the_tail(self):
        fused = [_p(k) for k in ["a", "b", "c", "d", "kw1", "kw2"]]
        out = retrieval.reserve_slots(fused, ["kw1", "kw2", "a"], limit=4, max_reserved=2)
        self.assertEqual([p["key"] for p in out], ["a", "b", "kw1", "kw2"])

    def test_cap_on_reserved(self):
        fused = [_p(k) for k in ["a", "b", "c", "kw1", "kw2"]]
        out = retrieval.reserve_slots(fused, ["kw1", "kw2"], limit=3, max_reserved=1)
        self.assertEqual([p["key"] for p in out], ["a", "b", "kw1"])

    def test_nothing_missing(self):
        fused = [_p(k) for k in ["a", "kw1", "b"]]
        self.assertEqual(len(retrieval.reserve_slots(fused, ["kw1"], limit=5, max_reserved=2)), 3)


class PrepareKeywordsTest(unittest.TestCase):
    def test_normalizes_dedupes_and_limits(self):
        kws = retrieval.prepare_keywords(["الصِّين", "الصين", "أ", "", "بلاد الصين"] + [f"كلمة{i}" for i in range(20)])
        self.assertEqual(kws[:2], ["الصين", "بلاد الصين"])
        self.assertEqual(len(kws), 8)

    def test_like_pattern_escapes_wildcards(self):
        self.assertEqual(retrieval._like_pattern("50%_a"), "%50\\%\\_a%")


class SelectPassagesTest(unittest.TestCase):
    def test_keeps_relevant_and_applies_per_book_cap(self):
        cands = [_p(f"x{i}", doc="b1", relevance=3, rrf=1 - i / 100) for i in range(6)]
        cands += [_p("y1", doc="b2", relevance=2, rrf=0.5), _p("z", doc="b3", relevance=1, rrf=0.9)]
        selected = alwaraq.select_passages(cands, max_sources=10, per_book_cap=4)
        keys = [p["key"] for p in selected]
        self.assertEqual(keys, ["x0", "x1", "x2", "x3", "y1"])

    def test_no_cap_in_book_scope(self):
        cands = [_p(f"x{i}", relevance=3, rrf=1) for i in range(6)]
        self.assertEqual(len(alwaraq.select_passages(cands, 12, None)), 6)

    def test_rerank_failure_keeps_fusion_order(self):
        cands = [_p("a", relevance=None), _p("b", relevance=None)]
        self.assertEqual([p["key"] for p in alwaraq.select_passages(cands, 12, None)], ["a", "b"])


class RouteBooksTest(unittest.TestCase):
    def test_combines_named_keyword_and_vector_signals(self):
        catalogue = [{"legacy_bookid": "ib", "title_ar": "تحفة النظار في غرائب الأمصار", "author_ar": "ابن بطوطة"}]
        with mock.patch.object(retrieval, "get_catalogue", return_value=catalogue), \
             mock.patch.object(retrieval, "_safe_fetch_all", return_value=[]), \
             mock.patch.object(retrieval, "keyword_route", return_value=["67", "2194", "5"]) as kw, \
             mock.patch.object(retrieval, "vector_route", return_value=["91401", "67", "99"]):
            books = retrieval.route_books(["[0]"], ["ابن بطوطه"], ["مدينة الزيتون"], top_n=3, entities=["الصين"])
        self.assertEqual(books[0], "ib")      # named book always first
        self.assertEqual(books[1], "67")      # found by keyword AND vector
        self.assertEqual(len(books), 3)
        self.assertEqual(kw.call_args.args[0], ["الصين", "مدينة الزيتون"])  # titles not scanned

    def test_works_without_catalogue_or_profiles(self):
        with mock.patch.object(retrieval, "get_catalogue", return_value=[]), \
             mock.patch.object(retrieval, "_safe_fetch_all", return_value=[]), \
             mock.patch.object(retrieval, "keyword_route", return_value=[]), \
             mock.patch.object(retrieval, "vector_route", return_value=["276", "244"]):
            self.assertEqual(retrieval.route_books(["[0]"], [], [], top_n=8), ["276", "244"])

    def test_vector_route_filters_non_library_books(self):
        rows = [{"bookid": "67"}, {"bookid": "diaralaqool"}, {"bookid": "67"}, {"bookid": "IB-AwardsList"}, {"bookid": "5"}]
        with mock.patch.object(retrieval.db, "fetch_all", return_value=rows) as fa:
            self.assertEqual(retrieval.vector_route(["[0]"]), ["67", "5"])
        self.assertIn("<->", fa.call_args.args[0])  # L2: the operator the existing index supports
        self.assertEqual(fa.call_args.kwargs["settings"], {"ivfflat.probes": retrieval.IVFFLAT_PROBES})

    def test_keyword_scoring_prefers_rare_terms(self):
        # term 0 ("مدينة الزيتون") is rare, term 1 ("ابن بطوطة") appears in many books
        rows = [{"bookid": "67", "k0": 7, "k1": 3}, {"bookid": "2141", "k0": 0, "k1": 8}]
        rows += [{"bookid": str(i), "k0": 0, "k1": 1} for i in range(100, 130)]
        ranked = retrieval.score_keyword_hits(rows, 2, total_books=2000)
        self.assertEqual(ranked[0], "67")

    def test_pick_route_terms_prefers_specific_phrases(self):
        terms = retrieval.pick_route_terms(["الصين", "مدينة الزيتون", "ابن بطوطة", "في", "الصين"])
        self.assertEqual(terms, ["مدينة الزيتون", "ابن بطوطة"])

    def test_keyword_route_timeout_is_skipped(self):
        from psycopg2 import errors as pg_errors

        with mock.patch.object(retrieval.search_index, "is_ready", return_value=False), \
             mock.patch.object(retrieval.db, "fetch_all", side_effect=pg_errors.QueryCanceled()):
            self.assertEqual(retrieval.keyword_route(["مدينة الزيتون"]), [])

    def test_keyword_route_uses_index_when_ready(self):
        hits = {
            "%مدينه الزيتون%": [{"bookid": "67", "hits": 7}, {"bookid": "diaralaqool", "hits": 1}],
            "%ابن بطوطه%": [{"bookid": "67", "hits": 3}, {"bookid": "2141", "hits": 8}],
            "%الصين%": [{"bookid": str(i), "hits": 1000} for i in range(6)],  # too common
        }

        def fake_fetch(sql, params, settings=None):
            self.assertIn("FROM alwaraq_chunk_text", sql)
            return hits[params[0]]

        with mock.patch.object(retrieval.search_index, "is_ready", return_value=True), \
             mock.patch.object(retrieval.db, "fetch_all", side_effect=fake_fetch):
            ranked = retrieval.keyword_route(["مدينة الزيتون", "ابن بطوطة", "الصين"])
        self.assertEqual(ranked[0], "67")
        self.assertNotIn("diaralaqool", ranked)
        self.assertNotIn("0", ranked)  # books only matched by the over-common term are ignored

    def test_keyword_route_scans_books_when_index_not_ready(self):
        with mock.patch.object(retrieval.search_index, "is_ready", return_value=False), \
             mock.patch.object(retrieval.db, "fetch_all", return_value=[]) as fa:
            retrieval.keyword_route(["مدينة الزيتون", "ابن بطوطة", "الصين"])
        sql, params = fa.call_args.args
        self.assertIn("FROM books", sql)
        self.assertEqual(len(params), 4)  # 2 terms max without the index


class KeywordSearchBooksTest(unittest.TestCase):
    def test_indexed_path(self):
        with mock.patch.object(retrieval.search_index, "is_ready", return_value=True), \
             mock.patch.object(retrieval.db, "fetch_all", return_value=[
                 {"id": 4321, "bookid": "67", "score": 2.5, "text_content": "نص"}]) as fa:
            out = retrieval.keyword_search_books(["67"], ["مدينه الزيتون"], 10)
        self.assertIn("FROM alwaraq_chunk_text", fa.call_args.args[0])
        self.assertEqual(out[0]["document_id"], "67")
        self.assertEqual(out[0]["chunk_id"], 4321)  # needed for neighbour context

    def test_fallback_path(self):
        with mock.patch.object(retrieval.search_index, "is_ready", return_value=False), \
             mock.patch.object(retrieval.db, "fetch_all", return_value=[]) as fa:
            retrieval.keyword_search_books(["67"], ["مدينه الزيتون"], 10)
        self.assertIn("translate(text_content", fa.call_args.args[0])

    def test_every_spelling_of_a_keyword_is_searched(self):
        """أمريكا in the question must also match أميركة in an older translation."""
        with mock.patch.object(retrieval.search_index, "is_ready", return_value=True), \
             mock.patch.object(retrieval.db, "fetch_all", return_value=[]) as fa:
            retrieval.keyword_search_books(["3066"], ["أمريكا"], 10)
        sql, params = fa.call_args.args
        self.assertIn("%اميركه%", params)
        self.assertIn("ILIKE", sql)  # Latin titles are matched whatever the casing

    def test_matching_is_case_insensitive(self):
        with mock.patch.object(retrieval.search_index, "is_ready", return_value=False), \
             mock.patch.object(retrieval.db, "fetch_all", return_value=[]) as fa:
            retrieval.keyword_search_books(["3066"], ["Three Essays On America"], 10)
        sql, params = fa.call_args.args
        self.assertIn("norm ILIKE", sql)
        self.assertIn("%Three Essays On America%", params)


class ExpandNeighboursTest(unittest.TestCase):
    """A work and its author routinely sit in different chunks."""

    def _passage(self, chunk_id, doc="3066"):
        return {"key": "k", "source": "books", "chunk_id": chunk_id, "document_id": doc, "text": "MIDDLE"}

    def test_adds_text_from_the_chunks_either_side(self):
        rows = [
            {"id": 9, "bookid": "3066", "text_content": "BEFORE"},
            {"id": 11, "bookid": "3066", "text_content": "AFTER"},
        ]
        with mock.patch.object(retrieval.db, "fetch_all", return_value=rows) as fa:
            out = retrieval.expand_neighbours([self._passage(10)], chars=100)
        self.assertEqual(out[0]["text"], "BEFORE MIDDLE AFTER")
        self.assertEqual(out[0]["key"], "k")  # citation identity is untouched
        self.assertEqual(sorted(fa.call_args.args[1][0]), [9, 11])

    def test_never_crosses_into_another_book(self):
        rows = [{"id": 11, "bookid": "OTHER", "text_content": "AFTER"}]
        with mock.patch.object(retrieval.db, "fetch_all", return_value=rows):
            out = retrieval.expand_neighbours([self._passage(10)], chars=100)
        self.assertEqual(out[0]["text"], "MIDDLE")

    def test_window_is_bounded(self):
        rows = [{"id": 9, "bookid": "3066", "text_content": "x" * 5000}]
        with mock.patch.object(retrieval.db, "fetch_all", return_value=rows):
            out = retrieval.expand_neighbours([self._passage(10)], chars=50)
        self.assertEqual(out[0]["text"], "x" * 50 + " MIDDLE")

    def test_disabled_and_page_level_passages_are_left_alone(self):
        page = {"key": "p:1", "source": "passages", "document_id": "3066", "text": "PAGE"}
        with mock.patch.object(retrieval.db, "fetch_all") as fa:
            self.assertEqual(retrieval.expand_neighbours([page], chars=100)[0]["text"], "PAGE")
            self.assertEqual(retrieval.expand_neighbours([self._passage(10)], chars=0)[0]["text"], "MIDDLE")
        fa.assert_not_called()

    def test_lookup_failure_leaves_passages_usable(self):
        with mock.patch.object(retrieval.db, "fetch_all", side_effect=RuntimeError("db down")):
            out = retrieval.expand_neighbours([self._passage(10)], chars=100)
        self.assertEqual(out[0]["text"], "MIDDLE")


class ReadOnlyBooksTableTest(unittest.TestCase):
    """The new module must never write to the shared `books` table."""

    def test_no_writes_to_books(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        files = [root / "app/langchain/alwaraq.py", root / "app/server/alwaraq_routes.py"]
        files += list((root / "app/langchain/alwaraq_lib").glob("*.py"))
        files += list((root / "migrations").glob("*_alwaraq_*.sql"))
        files += list((root / "migrations").glob("*_query_log_*.sql"))
        pattern = re.compile(
            r"(INSERT\s+INTO|UPDATE|DELETE\s+FROM|ALTER\s+TABLE|DROP\s+TABLE|TRUNCATE)\s+books\b",
            re.IGNORECASE,
        )
        for f in files:
            with self.subTest(file=f.name):
                self.assertIsNone(pattern.search(f.read_text(encoding="utf-8")))

    def test_no_imports_from_other_domain_modules(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        files = [root / "app/langchain/alwaraq.py"] + list((root / "app/langchain/alwaraq_lib").glob("*.py"))
        banned = ["chroma_store", "motanabi", "batuta_books", "articles", "diaralaqool", "awards"]
        for f in files:
            src = f.read_text(encoding="utf-8")
            for name in banned:
                with self.subTest(file=f.name, module=name):
                    self.assertNotIn(f"app.langchain.{name}", src)


if __name__ == "__main__":
    unittest.main()
