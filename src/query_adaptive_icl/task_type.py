from __future__ import annotations

import re


def detect_task_type(question: str) -> str:
    """Infer the prompt constraint from the question format."""
    if re.search(r"Options:\n- Yes\n- No", question, re.IGNORECASE):
        return "binary_choice"

    if re.search(r"Is the following sentence", question):
        return "binary_choice"

    if re.search(r"Options:\n- valid\n- invalid", question, re.IGNORECASE):
        return "binary_choice"

    if re.search(r"\(A\)|\(B\)|\(C\)|\(D\)", question):
        return "multiple_choice"

    if re.search(r"[0-9+\-*/=]", question) and "=" in question:
        return "free_text"

    return "free_text"
