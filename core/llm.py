from __future__ import annotations
import google.generativeai as genai

class GeminiLLM:
    def __init__(self, model_name: str):
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        try:
            out = self.model.generate_content([prompt])
            return (out.text or "").strip()
        except Exception as e:
            return f"__LLM_ERROR__ {type(e).__name__}: {e}"
