import unittest
from unittest import mock

from app.langchain.alwaraq_lib import book_names

HEAD = b'{"name":"The Story of Civilzation-Vol I ","authorid":319,"author":"Will Durant","pages":['


def _response(body: bytes, status: int = 200, chunk_size: int = book_names.HEAD_BYTES):
    """A stand-in for requests' streaming response that records what was read."""
    read = []

    def iter_content(size):
        for i in range(0, len(body), chunk_size):
            piece = body[i : i + chunk_size]
            read.append(piece)
            yield piece

    resp = mock.MagicMock(status_code=status)
    resp.iter_content.side_effect = iter_content
    resp.__enter__.return_value = resp
    return resp, read


class FetchBookMetaTest(unittest.TestCase):
    def test_api_error_returns_none(self):
        resp, _ = _response(b"", status=500)
        with mock.patch.object(book_names.requests, "get", return_value=resp):
            self.assertIsNone(book_names.fetch_book_meta("67"))

    def test_only_the_head_of_a_huge_body_is_read(self):
        """The endpoint returns the whole book (megabytes); the name is in its first bytes."""
        resp, read = _response(HEAD + b"x" * 5_000_000)
        with mock.patch.object(book_names.requests, "get", return_value=resp) as get:
            meta = book_names.fetch_book_meta("603")
        self.assertEqual(meta, {"name": "The Story of Civilzation-Vol I", "author": "Will Durant"})
        self.assertEqual(len(read), 1)  # one chunk, then the connection is dropped
        self.assertTrue(get.call_args.kwargs["stream"])

    def test_a_name_split_across_chunks_is_still_found(self):
        resp, read = _response(HEAD, chunk_size=16)
        with mock.patch.object(book_names.requests, "get", return_value=resp):
            self.assertEqual(book_names.fetch_book_meta("603")["author"], "Will Durant")
        self.assertGreater(len(read), 1)

    def test_arabic_name_and_escapes(self):
        body = '{"name":" تحفة \\"النظار\\" ","author":"ابن بطوطة"}'.encode("utf-8")
        resp, _ = _response(body)
        with mock.patch.object(book_names.requests, "get", return_value=resp):
            self.assertEqual(
                book_names.fetch_book_meta("67"), {"name": 'تحفة "النظار"', "author": "ابن بطوطة"}
            )

    def test_gives_up_instead_of_downloading_everything(self):
        resp, read = _response(b'{"pages":[' + b"x" * 3_000_000, chunk_size=200_000)
        with mock.patch.object(book_names.requests, "get", return_value=resp):
            self.assertIsNone(book_names.fetch_book_meta("603"))
        self.assertLess(sum(len(c) for c in read), 3_000_000)

    def test_a_name_with_no_author_is_still_a_name(self):
        resp, _ = _response(b'{"name":"David Copperfield","isPdfAvailable":false}')
        with mock.patch.object(book_names.requests, "get", return_value=resp):
            self.assertEqual(book_names.fetch_book_meta("90051"), {"name": "David Copperfield", "author": None})


class SaveTest(unittest.TestCase):
    """A name goes in the column matching its script, a title and an author apart."""

    def test_a_latin_name_and_author(self):
        with mock.patch.object(book_names.db, "execute") as ex:
            book_names._save("90051", {"name": "David Copperfield", "author": "Charles Dickens"})
        book_id, legacy, title_ar, title_en, author_ar, author_en, genre = ex.call_args.args[1]
        self.assertEqual((book_id, legacy), ("90051", "90051"))
        self.assertIsNone(title_ar)
        self.assertEqual(title_en, "David Copperfield")
        self.assertIsNone(author_ar)
        self.assertEqual(author_en, "Charles Dickens")

    def test_an_arabic_name_and_author(self):
        with mock.patch.object(book_names.db, "execute") as ex:
            book_names._save("67", {"name": "تحفة النظار", "author": "ابن بطوطة"})
        _, _, title_ar, title_en, author_ar, author_en, _ = ex.call_args.args[1]
        self.assertEqual((title_ar, author_ar), ("تحفة النظار", "ابن بطوطة"))
        self.assertIsNone(title_en)
        self.assertIsNone(author_en)

    def test_an_arabic_title_by_a_latin_author(self):
        """قصة الحضارة by Will Durant — the two fields are judged separately."""
        with mock.patch.object(book_names.db, "execute") as ex:
            book_names.save_entries([{"book_id": "628", "name": "قصة الحضارة",
                                      "author": "Will Durant", "genre": "The Legacy of humanity"}])
        _, _, title_ar, title_en, author_ar, author_en, genre = ex.call_args.args[1]
        self.assertEqual(title_ar, "قصة الحضارة")
        self.assertIsNone(title_en)
        self.assertEqual(author_en, "Will Durant")
        self.assertIsNone(author_ar)
        self.assertEqual(genre, "The Legacy of humanity")

    def test_language_is_never_touched(self):
        """It is detected from the book's own text, which is the better signal."""
        with mock.patch.object(book_names.db, "execute") as ex:
            book_names.save_entries([{"book_id": "1", "name": "x", "author": None, "genre": None}])
        self.assertNotIn("language", ex.call_args.args[0])

    def test_writes_in_batches(self):
        entries = [{"book_id": str(i), "name": f"n{i}", "author": None, "genre": None} for i in range(7)]
        with mock.patch.object(book_names.db, "execute") as ex:
            self.assertEqual(book_names.save_entries(entries, batch_size=3), 7)
        self.assertEqual(ex.call_count, 3)


class SyncCatalogueTest(unittest.TestCase):
    """The listing covers books we do not hold; only our own ids are written."""

    PAGES = {
        1: ({"books": [{"bookid": 628, "name": "قصة الحضارة", "author": "Will Durant",
                        "subjectName": "The Legacy of humanity"},
                       {"bookid": 999, "name": "A book we do not have", "author": "X"}],
             "total": 4, "isLastPage": False}),
        2: ({"books": [{"bookid": 90051, "name": "David Copperfield", "author": "Charles Dickens"},
                       {"bookid": 1, "name": "", "author": None}],   # no name: skipped
             "total": 4, "isLastPage": True}),
    }

    def _get(self, url, params=None, timeout=None):
        return mock.Mock(status_code=200, json=mock.Mock(return_value=self.PAGES[params["page"]]))

    def test_only_books_in_our_store_are_written(self):
        with mock.patch.object(book_names, "LIST_PAGE_SIZE", 2), \
             mock.patch.object(book_names.requests, "get", side_effect=self._get), \
             mock.patch.object(book_names, "save_entries", side_effect=len) as save:
            result = book_names.sync_catalogue(["628", "90051", "4242"])
        written = sorted(b["book_id"] for b in save.call_args.args[0])
        self.assertEqual(written, ["628", "90051"])   # 999 is upstream only, 4242 is not listed
        self.assertEqual(result["named"], 2)
        self.assertEqual(result["not_listed"], 1)     # 4242
        self.assertEqual(result["in_library"], 3)

    def test_a_failing_page_does_not_lose_the_rest(self):
        def flaky(url, params=None, timeout=None):
            if params["page"] == 2:
                raise RuntimeError("upstream down")
            return self._get(url, params, timeout)

        with mock.patch.object(book_names, "LIST_PAGE_SIZE", 2), \
             mock.patch.object(book_names.requests, "get", side_effect=flaky), \
             mock.patch.object(book_names, "save_entries", side_effect=len) as save:
            book_names.sync_catalogue(["628", "90051"])
        self.assertEqual([b["book_id"] for b in save.call_args.args[0]], ["628"])

    def test_an_http_error_page_is_empty_not_fatal(self):
        with mock.patch.object(book_names.requests, "get",
                               return_value=mock.Mock(status_code=503)):
            self.assertEqual(book_names.fetch_book_list_page(1), ([], False, 0))


class EnsureNamesTest(unittest.TestCase):
    def setUp(self):
        book_names._attempted.clear()

    def test_looks_them_up_together_and_reports_what_landed(self):
        with mock.patch.object(book_names, "_fetch_and_save", side_effect=lambda b, cb: f"name-{b}") as f:
            saved = book_names.ensure_names(["603", "611", "607"], budget_s=5)
        self.assertEqual(saved, 3)
        self.assertEqual(sorted(c.args[0] for c in f.call_args_list), ["603", "607", "611"])

    def test_a_book_is_not_looked_up_twice_in_the_retry_window(self):
        with mock.patch.object(book_names, "_fetch_and_save", return_value="n") as f:
            book_names.ensure_names(["603", "611"], budget_s=5)
            book_names.ensure_names(["603", "609"], budget_s=5)
        self.assertEqual(sorted(c.args[0] for c in f.call_args_list), ["603", "609", "611"])

    def test_a_failed_lookup_is_not_counted(self):
        with mock.patch.object(book_names, "_fetch_and_save", return_value=None):
            self.assertEqual(book_names.ensure_names(["603"], budget_s=5), 0)

    def test_nothing_to_do(self):
        self.assertEqual(book_names.ensure_names([], budget_s=5), 0)

    def test_slow_lookups_do_not_hold_the_answer_up(self):
        import time

        def slow(book_id, on_saved):
            time.sleep(2)
            return "late"

        with mock.patch.object(book_names, "_fetch_and_save", side_effect=slow):
            started = time.time()
            saved = book_names.ensure_names(["603"], budget_s=0.2)
        self.assertEqual(saved, 0)
        self.assertLess(time.time() - started, 1.5)  # returned without waiting for it


class RequestNamesTest(unittest.TestCase):
    def test_each_book_attempted_once_per_retry_window(self):
        book_names._attempted.clear()
        with mock.patch.object(book_names, "_queue") as q, mock.patch.object(book_names.threading, "Thread"):
            book_names.request_names(["67", "94"])
            book_names.request_names(["67", "5"])
        queued = [c.args[0] for c in q.put.call_args_list]
        self.assertEqual(queued, ["67", "94", "5"])


if __name__ == "__main__":
    unittest.main()
