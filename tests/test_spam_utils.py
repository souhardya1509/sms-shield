import unittest
from spam_utils import validate_message, classify_message


class FakeModel:
    classes_ = ["ham", "spam"]

    def __init__(self, ham, spam):
        self.scores = [ham, spam]
        self.seen_messages = None

    def predict_proba(self, messages):
        self.seen_messages = messages
        return [self.scores]


class ValidationTests(unittest.TestCase):
    def test_empty_string_is_rejected(self):
        self.assertIn("Please enter", validate_message(""))

    def test_whitespace_is_rejected(self):
        self.assertIn("Please enter", validate_message("   \n  "))

    def test_symbols_and_numbers_only_are_rejected(self):
        self.assertIn("containing letters", validate_message("!!! $$$ 12345 ???"))

    def test_message_over_limit_is_rejected(self):
        self.assertIn("maximum 5,000", validate_message("a" * 5001))

    def test_normal_text_passes_validation(self):
        self.assertIsNone(validate_message("Hey, are you coming tomorrow?"))

    def test_exactly_5000_characters_passes_length_check(self):
        self.assertIsNone(validate_message("a" * 5000))


class ClassificationTests(unittest.TestCase):
    def test_high_spam_score_is_spam(self):
        model = FakeModel(0.10, 0.90)
        result = classify_message(model, "claim your prize")
        self.assertEqual(result["result"], "spam")
        self.assertAlmostEqual(result["spam_probability"], 90.0)

    def test_high_ham_score_is_ham(self):
        model = FakeModel(0.90, 0.10)
        result = classify_message(model, "see you tomorrow")
        self.assertEqual(result["result"], "ham")

    def test_middle_scores_are_uncertain(self):
        model = FakeModel(0.60, 0.40)
        result = classify_message(model, "your order has shipped")
        self.assertEqual(result["result"], "uncertain")

    def test_missing_expected_class_raises_error(self):
        model = FakeModel(0.10, 0.90)
        model.classes_ = ["legitimate", "spam"]
        with self.assertRaisesRegex(ValueError, "expected 'ham' and 'spam'"):
            classify_message(model, "message")


if __name__ == "__main__":
    unittest.main(verbosity=2)
