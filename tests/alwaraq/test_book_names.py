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
    def test_a_latin_name_is_stored_as_the_english_title(self):
        with mock.patch.object(book_names.db, "execute") as ex:
            book_names._save("603", {"name": "David Copperfield", "author": "Charles Dickens"})
        params = ex.call_args.args[1]
        self.assertEqual(params[2], "David Copperfield")  # title_ar (NOT NULL)
        self.assertEqual(params[3], "David Copperfield")  # title_en — what an English answer reads
        self.assertEqual(params[5], "en")

    def test_an_arabic_name_leaves_the_english_title_empty(self):
        with mock.patch.object(book_names.db, "execute") as ex:
            book_names._save("67", {"name": "تحفة النظار", "author": None})
        params = ex.call_args.args[1]
        self.assertEqual(params[2], "تحفة النظار")
        self.assertIsNone(params[3])
        self.assertEqual(params[5], "ar")


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
