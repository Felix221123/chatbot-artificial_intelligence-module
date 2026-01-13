# multilingual.py
import os
from langdetect import detect
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


class Translator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def detect_lang(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "en"

    # functionality for translation
    def translate(self, text: str, target_lang: str) -> str:
        if not text.strip():
            return text

        # Translate only
        resp = self.client.responses.create(
            model="gpt-4.1-mini",
            input=f"Translate to {target_lang}. Return only the translation:\n\n{text}"
        )
        return resp.output_text.strip()

    # turning text to english 
    def to_english(self, text: str):
        lang = self.detect_lang(text)
        if lang == "en":
            return "en", text
        return lang, self.translate(text, "English")

    # turning text from english to lang  
    def from_english(self, text: str, target_lang: str):
        if target_lang == "en":
            return text
        return self.translate(text, target_lang)
