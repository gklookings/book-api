import unittest

from app.langchain.alwaraq_lib.verify import (
    verify_composition,
    locate_quote,
    strip_unknown_markers,
    verify_answer,
)

PASSAGE_1 = (
    "ثم سافرنا إلى مدينة الزيتون، وهي مدينة عظيمة كبيرة، تُصنع بها ثياب الكمخا والأطلس، "
    "وهي من أعظم المدن وأكبرها في بلاد الصين، ومرساها من أعظم المراسي في الدنيا."
)
PASSAGE_2 = "وأهل الصين أعظم الأمم إحكاما للصناعات وأشدهم إتقانا فيها، وذلك مشهور من حالهم."


def _passages():
    return {"P1": {"text": PASSAGE_1}, "P2": {"text": PASSAGE_2}}


class LocateQuoteTest(unittest.TestCase):
    def test_verbatim_ignoring_diacritics_and_punctuation(self):
        found = locate_quote("وهي من أعظم المدن وأكبرها في بلاد الصين", PASSAGE_1)
        self.assertEqual(found, {"verbatim": True})
        found = locate_quote("تصنع بها ثياب الكمخا", PASSAGE_1)  # passage has tashkeel
        self.assertEqual(found, {"verbatim": True})

    def test_near_match_tolerates_small_drift(self):
        # one word differs ("اكبر" instead of "اعظم") in a long quote
        quote = "وهي من أكبر المدن وأكبرها في بلاد الصين ومرساها من أعظم المراسي في الدنيا"
        self.assertEqual(locate_quote(quote, PASSAGE_1), {"verbatim": False})

    def test_invented_quote_rejected(self):
        self.assertIsNone(locate_quote("وكان أهلها يعبدون النار ويشربون الشاي كل صباح", PASSAGE_1))
        self.assertIsNone(locate_quote("ab", PASSAGE_1))


class VerifyAnswerTest(unittest.TestCase):
    def test_drops_unknown_citations_invented_quotes_and_empty_claims(self):
        composed = {
            "summary": "مدينة الزيتون عظيمة [P1] [P9]",
            "sections": [
                {
                    "heading": "المدن",
                    "claims": [
                        {
                            "text": "الزيتون من أكبر مدن الصين [P1]",
                            "citations": ["P1", "P9"],
                            "quotes": [
                                {"passage": "P1", "text": "وهي من أعظم المدن وأكبرها في بلاد الصين"},
                                {"passage": "P1", "text": "نص مخترع لا وجود له في المصدر إطلاقا"},
                            ],
                        },
                        {"text": "ادعاء بلا مصدر", "citations": ["P7"], "quotes": []},
                    ],
                },
                {"heading": "فارغ", "claims": [{"text": "بلا مصدر", "citations": []}]},
            ],
            "disagreements": [{"topic": "x", "citations": ["P1"]}],
        }
        result = verify_answer(composed, _passages())

        self.assertTrue(result["has_evidence"])
        self.assertEqual(len(result["sections"]), 1)
        claims = result["sections"][0]["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["citations"], ["P1"])
        self.assertEqual(len(claims[0]["quotes"]), 1)
        self.assertEqual(claims[0]["confidence"], "confirmed")  # verbatim quote
        self.assertEqual(result["disagreements"], [])  # needs >= 2 valid citations
        self.assertEqual(result["cited_labels"], ["P1"])
        self.assertEqual(result["dropped"], {"claims": 2, "quotes": 1, "citations": 2})
        self.assertIn("P1", result["quote_info"])

    def test_confidence_levels(self):
        composed = {
            "sections": [
                {
                    "heading": "h",
                    "claims": [
                        {"text": "two sources", "citations": ["P1", "P2"]},
                        {"text": "one source paraphrase", "citations": ["P2"]},
                        {"text": "hedged", "citations": ["P2"], "uncertain": True},
                        {"text": "conflict", "citations": ["P1", "P2"], "disputed": True},
                        {"text": "conflict one side", "citations": ["P1"], "disputed": True},
                    ],
                }
            ]
        }
        claims = verify_answer(composed, _passages())["sections"][0]["claims"]
        self.assertEqual(
            [c["confidence"] for c in claims],
            ["confirmed", "probable", "uncertain", "disputed", "uncertain"],
        )

    def test_quote_adds_its_passage_as_citation(self):
        composed = {
            "sections": [
                {
                    "heading": "h",
                    "claims": [
                        {
                            "text": "c",
                            "citations": [],
                            "quotes": [{"passage": "P2", "text": "أعظم الأمم إحكاما للصناعات"}],
                        }
                    ],
                }
            ]
        }
        claims = verify_answer(composed, _passages())["sections"][0]["claims"]
        self.assertEqual(claims[0]["citations"], ["P2"])

    def test_no_evidence_flag(self):
        result = verify_answer({"no_evidence": True, "sections": []}, _passages())
        self.assertFalse(result["has_evidence"])

    def test_strip_unknown_markers(self):
        self.assertEqual(strip_unknown_markers("a [P1] b [P9] c", {"P1"}), "a [P1] b c")


class VerifyCompositionTest(unittest.TestCase):
    """A composed piece is our writing: lines need no citation, quotes still do."""

    def test_uncited_lines_are_kept(self):
        composed = {"summary": "two lines", "sections": [{"heading": "", "claims": [
            {"text": "a line the assistant wrote"}]}]}
        result = verify_composition(composed, _passages())
        self.assertTrue(result["has_evidence"])
        claim = result["sections"][0]["claims"][0]
        self.assertEqual(claim["confidence"], "composed")
        self.assertEqual(claim["citations"], [])

    def test_a_quote_that_is_not_in_the_passage_is_dropped(self):
        composed = {"summary": "s", "sections": [{"heading": "", "claims": [
            {"text": "a line", "quotes": [{"passage": "P1", "text": "كلام لم يرد في أي نص من النصوص"}]}]}]}
        result = verify_composition(composed, _passages())
        self.assertEqual(result["sections"][0]["claims"][0]["quotes"], [])
        self.assertEqual(result["dropped"]["quotes"], 1)
        self.assertTrue(result["has_evidence"])  # the line survives, the invention does not

    def test_a_summary_alone_is_a_valid_piece(self):
        self.assertTrue(verify_composition({"summary": "a two line hook"}, _passages())["has_evidence"])

    def test_nothing_written_is_not_a_piece(self):
        self.assertFalse(verify_composition({"summary": "   ", "sections": []}, _passages())["has_evidence"])
        self.assertFalse(verify_composition({"summary": "x", "no_evidence": True}, _passages())["has_evidence"])


if __name__ == "__main__":
    unittest.main()
