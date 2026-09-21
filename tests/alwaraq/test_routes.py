"""Route tests on a minimal app that mounts only the /alwaraq router."""

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.langchain import alwaraq
from app.server import alwaraq_routes


def _client():
    app = FastAPI()
    app.include_router(alwaraq_routes.router)
    return TestClient(app)


class AnswerRouteTest(unittest.TestCase):
    def test_forwards_params_and_session_header(self):
        with mock.patch.object(alwaraq, "answer_question", return_value=({"status": "ok"}, 200)) as ans:
            resp = _client().get(
                "/alwaraq/answer",
                params={"query": "سؤال", "document_id": "123"},
                headers={"Sessiontoken": "tok"},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok", "status_code": 200})
        kwargs = ans.call_args.kwargs
        self.assertEqual(kwargs["query"], "سؤال")
        self.assertEqual(kwargs["document_id"], "123")
        self.assertEqual(kwargs["session_token"], "tok")

    def test_document_id_is_optional(self):
        with mock.patch.object(alwaraq, "answer_question", return_value=({"status": "ok"}, 200)) as ans:
            _client().get("/alwaraq/answer", params={"query": "q"})
        self.assertIsNone(ans.call_args.kwargs["document_id"])
        self.assertIsNone(ans.call_args.kwargs["session_token"])

    def test_query_is_required(self):
        self.assertEqual(_client().get("/alwaraq/answer").status_code, 422)

    def test_error_shape_matches_chromadb_answer(self):
        with mock.patch.object(alwaraq, "answer_question", return_value=({"error": "nope"}, 404)):
            resp = _client().get("/alwaraq/answer", params={"query": "q", "document_id": "x"})
        self.assertEqual(resp.json(), {"error": "nope", "status_code": 404})


class MemoryRoutesTest(unittest.TestCase):
    def test_clear_requires_session_token(self):
        self.assertEqual(_client().delete("/alwaraq/memory").status_code, 400)

    def test_clear_uses_alwaraq_domain(self):
        with mock.patch.object(alwaraq, "memory_service") as svc:
            svc.clear_session.return_value = {"memory_deleted": 1, "history_deleted": 4}
            resp = _client().delete("/alwaraq/memory", headers={"Sessiontoken": "tok"})
        svc.clear_session.assert_called_once_with(session_token="tok", domain="alwaraq")
        self.assertEqual(resp.json()["history_deleted"], 4)
        self.assertEqual(resp.json()["domain"], "alwaraq")

    def test_history(self):
        stored = [
            {"role": "user", "content": "q", "created_at": None},
            {"role": "assistant", "content": "a", "created_at": None,
             "metadata": {"books_cited": ["90051"], "answer_mode": "composed"}},
        ]
        with mock.patch.object(alwaraq, "memory_repository") as repo:
            repo.get_chat_history.return_value = stored
            resp = _client().get("/alwaraq/history", headers={"Sessiontoken": "tok"})
        repo.get_chat_history.assert_called_once_with("tok", "alwaraq", limit=50, offset=0)
        messages = resp.json()["messages"]
        self.assertEqual(messages[0], {"role": "user", "content": "q", "metadata": {}, "created_at": None})
        # what the answer was drawn from travels with it, not only its prose
        self.assertEqual(messages[1]["metadata"]["books_cited"], ["90051"])


class AdminRoutesTest(unittest.TestCase):
    def test_admin_requires_token(self):
        resp = _client().post("/alwaraq/admin/profiles/build", json={})
        self.assertEqual(resp.status_code, 401)

    def test_register_books_defaults_legacy_bookid(self):
        client = _client()
        with mock.patch.object(alwaraq_routes, "verify_token", return_value="admin"), \
             mock.patch.object(alwaraq, "register_books", return_value=1) as reg:
            resp = client.post(
                "/alwaraq/admin/books",
                json={"books": [{"book_id": "55", "title_ar": "مروج الذهب"}], "build_profiles": False},
                headers={"Authorization": "Bearer t"},
            )
        self.assertEqual(resp.json()["count"], 1)
        self.assertEqual(reg.call_args.args[0][0]["legacy_bookid"], "55")


if __name__ == "__main__":
    unittest.main()
