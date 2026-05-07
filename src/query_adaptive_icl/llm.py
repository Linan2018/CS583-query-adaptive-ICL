from __future__ import annotations

import os
import re


def extract_answer(llm_output: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", llm_output, re.DOTALL)
    return match.group(1).strip() if match else llm_output.strip()


def create_gemini_model(model_name: str, api_key: str | None = None):
    import google.generativeai as genai

    effective_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not effective_key:
        raise RuntimeError(
            "Missing Google API key. Set GOOGLE_API_KEY in your environment before running LLM stages."
        )

    genai.configure(api_key=effective_key)
    return genai.GenerativeModel(model_name)


def request_gemini_text(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    if not response or not hasattr(response, "candidates") or not response.candidates:
        return "[No valid response]"
    if not response.candidates[0].content.parts:
        return "[Empty Response]"
    return response.candidates[0].content.parts[0].text
