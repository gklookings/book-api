import unittest
from unittest import mock

from app.langchain.alwaraq_lib import book_names


class FetchBookMetaTest(unittest.TestCase):
    def test_api_error_returns_none(self):
        with mock.patch.object(book_names.requests, "get", return_value=mock.Mock(status_code=500)):
            self.assertIsNone(book_names.fetch_book_meta("67"))

    def test_name_and_author(self):
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"name": " تحفة النظار ", "author": "ابن بطوطة", "fullBookPages": "..."}
        with mock.patch.object(book_names.requests, "get", return_value=resp):
            self.assertEqual(book_names.fetch_book_meta("67"), {"name": "تحفة النظار", "author": "ابن بطوطة"})


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
