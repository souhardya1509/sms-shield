"""Small, testable input-validation and prediction helpers for SMS Shield."""

MAX_MESSAGE_LENGTH = 5000


def validate_message(message, max_length=MAX_MESSAGE_LENGTH):
    """Return an error message when input is invalid; otherwise return None."""
    if not isinstance(message, str) or not message.strip():
        return "Please enter a message before classifying it."
    if len(message) > max_length:
        return f"Please enter a shorter message (maximum {max_length:,} characters)."
    if not any(char.isalpha() for char in message):
        return "Please enter a message containing letters. Symbols and numbers alone cannot be classified reliably."
    return None


def classify_message(model, message, spam_threshold=0.50, ham_threshold=0.70):
    """Return a result and class scores using a fitted sklearn-style model."""
    probabilities = model.predict_proba([message])[0]
    classes = [str(c).strip().lower() for c in model.classes_]
    if "ham" not in classes or "spam" not in classes:
        raise ValueError("The saved model does not contain expected 'ham' and 'spam' classes.")

    ham_probability = float(probabilities[classes.index("ham")])
    spam_probability = float(probabilities[classes.index("spam")])

    if spam_probability >= spam_threshold:
        result = "spam"
    elif ham_probability >= ham_threshold:
        result = "ham"
    else:
        result = "uncertain"

    return {
        "result": result,
        "ham_probability": ham_probability * 100,
        "spam_probability": spam_probability * 100,
    }
