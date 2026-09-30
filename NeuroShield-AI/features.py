import re

URGENCY_WORDS = [
    "urgent", "immediately", "now", "verify",
    "suspended", "blocked", "claim", "winner", "prize"
]


def extract_features(text):
    text = str(text)

    characters = len(text)
    words = text.split()
    word_count = len(words)

    digit_count = sum(c.isdigit() for c in text)
    uppercase_count = sum(c.isupper() for c in text)

    special_count = len(
        re.findall(r"[!$₹£€%@#]", text)
    )

    url_count = len(
        re.findall(
            r"http[s]?://|www\.|bit\.ly|tinyurl",
            text.lower()
        )
    )

    urgency_count = sum(
        1
        for word in URGENCY_WORDS
        if word in text.lower()
    )

    uppercase_ratio = (
        uppercase_count / characters
        if characters else 0
    )

    digit_ratio = (
        digit_count / characters
        if characters else 0
    )

    return [
        characters,
        word_count,
        digit_count,
        digit_ratio,
        uppercase_ratio,
        special_count,
        url_count,
        urgency_count,
    ]