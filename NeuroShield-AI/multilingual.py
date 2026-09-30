from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

DetectorFactory.seed = 42


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Urdu",
    "ne": "Nepali",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-cn": "Chinese",
}


def detect_language(text):
    """
    Detect the language of the supplied message.
    Returns language code and readable name.
    """

    try:
        code = detect(text)

        name = LANGUAGE_NAMES.get(
            code,
            code.upper()
        )

        return code, name

    except Exception:
        return "unknown", "Unknown"


def translate_to_english(text, language_code):
    """
    Translate non-English text into English.

    English text is returned unchanged.
    """

    if language_code == "en":
        return text, False

    try:
        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

        return translated, True

    except Exception:
        return text, False


def prepare_multilingual_message(text):
    """
    Complete multilingual preprocessing layer.

    Detect language -> translate when required ->
    return English text for existing NeuroShield models.
    """

    language_code, language_name = detect_language(text)

    translated_text, translated = translate_to_english(
        text,
        language_code
    )

    return {
        "original_text": text,
        "language_code": language_code,
        "language_name": language_name,
        "translated_text": translated_text,
        "was_translated": translated,
    }