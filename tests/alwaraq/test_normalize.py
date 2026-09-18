import unittest

from app.langchain.alwaraq_lib.normalize import (
    detect_language,
    normalize_arabic,
    normalize_for_match,
)


class NormalizeArabicTest(unittest.TestCase):
    def test_cases(self):
        cases = [
            ("مُحَمَّدٌ", "محمد"),                 # tashkeel
            ("الكتـــاب", "الكتاب"),               # tatweel
            ("أحمد إبراهيم آدم ٱلله", "احمد ابراهيم ادم الله"),  # alef forms
            ("مصطفى", "مصطفي"),                   # alef maqsura
            ("مدينة", "مدينه"),                   # ta marbuta
            ("مؤمن بئر", "مومن بير"),              # hamza carriers
            ("سنة ٧٢٥ و۱۲", "سنه 725 و12"),        # Arabic-Indic / Persian digits
            ("  سمرقند \n\t بخارى ", "سمرقند بخاري"),
            ("", ""),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_arabic(raw), expected)

    def test_match_normalization_strips_punctuation(self):
        self.assertEqual(normalize_for_match("قالَ: «دخلنا القاهرة»، وهي..."), "قال دخلنا القاهره وهي")

    def test_detect_language(self):
        self.assertEqual(detect_language("ماذا كتب ابن بطوطة عن الصين؟"), "ar")
        self.assertEqual(detect_language("What did Ibn Battuta write about China?"), "en")
        self.assertEqual(detect_language(""), "ar")


if __name__ == "__main__":
    unittest.main()
