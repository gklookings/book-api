"""End-to-end tests of alwaraq.answer_question with DB, embeddings, LLM and memory mocked."""

import unittest
from unittest import mock

from app.langchain import alwaraq
from app.langchain.alwaraq_lib import retrieval

CHINA = "وهي من أعظم المدن وأكبرها في بلاد الصين، ومرساها من أعظم المراسي في الدنيا."
CAIRO = "ثم وصلت إلى مدينة مصر، هي أم البلاد، متناهية في كثرة العمارة، متباهية بالحسن والنضارة."


def _cand(key, text, doc="ib"):
    return {
        "key": key, "document_id": doc, "text": text, "source": "books", "citation_level": "book",
        "passage_id": None, "volume": None, "page": None, "chapter": None, "score": 0.8, "rrf": 0.03,
    }


def fake_llm(understanding=None, scores=None, composed=None):
    """Return a fake _invoke_json that answers by prompt type and records prompts."""
    calls = []

    def _invoke(model, prompt, usage):
        calls.append(prompt)
        if "query-analysis step" in prompt:
            return understanding or {
                "standalone_question": "ماذا كتب ابن بطوطة عن الصين؟",
                "language": "ar",
                "sub_queries": ["بلاد الصين"],
                "keywords": ["الصين"],
                "entities": ["ابن بطوطة"],
                "candidate_books": ["تحفة النظار"],
            }
        if "grading passages" in prompt:
            return {"scores": scores or {"C1": 3, "C2": 1}}
        return composed or {
            "no_evidence": False,
            "summary": "وصف ابن بطوطة مدن الصين بالعظمة [P1].",
            "sections": [{
                "heading": "الصين",
                "claims": [{
                    "text": "الزيتون من أعظم مدن الصين [P1]",
                    "citations": ["P1"],
                    "quotes": [{"passage": "P1", "text": "وهي من أعظم المدن وأكبرها في بلاد الصين"}],
                }],
            }],
            "disagreements": [],
            "follow_ups": ["ماذا قال عن الهند؟"],
        }

    return _invoke, calls


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.patches = [
            mock.patch.object(retrieval, "get_book_info", return_value={
                "book_id": "ib", "legacy_bookid": "ib", "title_ar": "تحفة النظار", "author_ar": "ابن بطوطة"}),
            mock.patch.object(retrieval, "book_exists", return_value=True),
            mock.patch.object(retrieval, "embed", return_value=[[0.0] * 768]),
            mock.patch.object(alwaraq, "_log_query", return_value="qid-1"),
            mock.patch.object(alwaraq.search_index, "maybe_sync_in_background"),
            mock.patch.object(alwaraq.book_names, "request_names"),
        ]
        self.mocks = [p.start() for p in self.patches]
        self.retrieve = mock.patch.object(
            retrieval, "retrieve", return_value=[_cand("k1", CHINA), _cand("k2", CAIRO)]
        ).start()
        self.memory = mock.patch.object(alwaraq, "memory_service").start()
        self.memory.load_context.return_value = ""

    def tearDown(self):
        mock.patch.stopall()

    def test_book_scope_cited_answer(self):
        invoke, _ = fake_llm()
        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("ماذا كتب ابن بطوطة عن الصين؟", document_id="ib")

        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["scope"], "book")
        self.assertEqual(data["books_searched"], [{"bookId": "ib", "bookName": "تحفة النظار"}])
        self.assertEqual(data["query_id"], "qid-1")
        self.assertFalse(data["hasMemory"])
        self.assertEqual(list(data["sources"]), ["P1"])
        src = data["sources"]["P1"]
        self.assertEqual(src["book"], "تحفة النظار")
        self.assertEqual(src["citation_level"], "book")
        self.assertEqual(src["quote"], "وهي من أعظم المدن وأكبرها في بلاد الصين")
        self.assertEqual(data["answer"]["sections"][0]["claims"][0]["confidence"], "confirmed")
        # only the passage graded >= 2 was sent to the composer
        self.retrieve.assert_called_once()
        self.assertEqual(self.retrieve.call_args.args[0], ["ib"])
        self.memory.save_exchange.assert_not_called()
        self.memory.load_context.assert_not_called()

    def test_unknown_document_id_is_404(self):
        retrieval.book_exists.return_value = False
        invoke, _ = fake_llm()
        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("سؤال", document_id="nope")
        self.assertEqual(status, 404)
        self.assertIn("nope", data["error"])

    def test_library_scope_routes_books_without_any_setup(self):
        invoke, calls = fake_llm()
        with mock.patch.object(retrieval, "route_books", return_value=["ib", "ms"]) as route, \
             mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("ماذا كتب ابن بطوطة عن الصين؟")
        self.assertEqual(status, 200)
        self.assertEqual(data["scope"], "library")
        self.assertIsNone(data["document_id"])
        self.assertEqual(
            data["books_searched"],
            [{"bookId": "ib", "bookName": "تحفة النظار"}, {"bookId": "ms", "bookName": "تحفة النظار"}],
        )
        self.assertIn("تحفة النظار", route.call_args.kwargs["names"])  # candidate_books passed to routing
        self.assertEqual(route.call_args.kwargs["keywords"], ["الصين"])
        self.assertTrue(self.retrieve.call_args.kwargs["include_global_passages"])
        self.assertIn("whole library", calls[0])

    def test_library_scope_nothing_routed(self):
        invoke, _ = fake_llm()
        with mock.patch.object(retrieval, "route_books", return_value=[]), \
             mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("سؤال عام")
        self.assertEqual(status, 404)

    def test_no_evidence_when_nothing_relevant(self):
        invoke, calls = fake_llm(scores={"C1": 0, "C2": 1})
        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("ما رأي المؤلف في الهواتف الذكية؟", document_id="ib")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "no_evidence")
        self.assertIn("لم يُعثر", data["answer"]["summary"])
        self.assertEqual(data["answer"]["sections"], [])
        self.assertTrue(data["sources"])  # closest passages still shown
        self.assertEqual(len(calls), 2)  # composer never called

    def test_invented_quotes_do_not_survive(self):
        composed = {
            "summary": "x [P1]",
            "sections": [{"heading": "h", "claims": [{
                "text": "invented", "citations": ["P5"],
                "quotes": [{"passage": "P1", "text": "كلام مخترع تماما لا يوجد في أي مصدر من المصادر"}],
            }]}],
        }
        invoke, _ = fake_llm(composed=composed)
        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, _ = alwaraq.answer_question("سؤال", document_id="ib")
        self.assertEqual(data["status"], "no_evidence")

    def test_session_memory_is_loaded_used_and_saved(self):
        self.memory.load_context.return_value = "User: ماذا كتب ابن بطوطة عن الصين؟\nAssistant: ..."
        invoke, calls = fake_llm(understanding={
            "standalone_question": "ماذا كتب ابن بطوطة عن مرسى الزيتون في الصين؟",
            "language": "ar", "sub_queries": ["مرسى الزيتون"], "keywords": ["الزيتون"],
        })
        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("وماذا عن مرساها؟", document_id="ib", session_token="tok")

        self.assertEqual(status, 200)
        self.assertTrue(data["hasMemory"])
        self.assertEqual(data["standalone_question"], "ماذا كتب ابن بطوطة عن مرسى الزيتون في الصين؟")
        self.memory.load_context.assert_called_once_with(session_token="tok", domain="alwaraq")
        # history reaches the understand and compose prompts
        self.assertIn("Conversation history", calls[0])
        self.assertIn("Conversation history", calls[-1])
        # retrieval uses the standalone question, never the raw history
        queries = self.retrieve.call_args.args[1]
        self.assertEqual(queries[0], "ماذا كتب ابن بطوطة عن مرسى الزيتون في الصين؟")
        self.assertFalse(any("Assistant:" in q for q in queries))
        # compact answer saved
        kwargs = self.memory.save_exchange.call_args.kwargs
        self.assertEqual(kwargs["domain"], "alwaraq")
        self.assertEqual(kwargs["question"], "وماذا عن مرساها؟")
        self.assertIn("Sources: تحفة النظار", kwargs["answer"])
        self.assertNotIn("[P1]", kwargs["answer"])

    def test_memory_failures_do_not_fail_the_answer(self):
        self.memory.load_context.side_effect = RuntimeError("db down")
        self.memory.save_exchange.side_effect = RuntimeError("db down")
        invoke, _ = fake_llm()
        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("سؤال", document_id="ib", session_token="tok")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")

    def test_understand_failure_falls_back_to_raw_question(self):
        def invoke(model, prompt, usage):
            if "query-analysis step" in prompt:
                raise ValueError("bad json")
            return fake_llm()[0](model, prompt, usage)

        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, status = alwaraq.answer_question("What did Ibn Battuta write about China?", document_id="ib")
        self.assertEqual(status, 200)
        self.assertEqual(data["language"], "en")
        self.assertEqual(self.retrieve.call_args.args[1][0], "What did Ibn Battuta write about China?")

    def test_unknown_book_names_are_null_and_looked_up(self):
        retrieval.get_book_info.return_value = None
        invoke, _ = fake_llm()
        with mock.patch.object(alwaraq, "_invoke_json", side_effect=invoke):
            data, _ = alwaraq.answer_question("سؤال", document_id="67")
        self.assertEqual(data["books_searched"], [{"bookId": "67", "bookName": None}])
        alwaraq.book_names.request_names.assert_called_once()
        self.assertEqual(alwaraq.book_names.request_names.call_args.args[0], ["67"])

    def test_empty_query(self):
        data, status = alwaraq.answer_question("   ")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
