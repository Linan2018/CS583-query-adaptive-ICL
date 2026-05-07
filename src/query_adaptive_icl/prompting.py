from __future__ import annotations


TASK_CONSTRAINTS = {
    "multiple_choice": "answer option letter only",
    "binary_choice": "Yes or No only",
    "free_text": "",
}


def generate_prompt(question: str, task_type: str, examples: str = "") -> str:
    return "\n".join(
        [
            "You will be given a question. Think step by step before giving a final answer to this question.",
            f"Show your final answer {TASK_CONSTRAINTS.get(task_type, '')} between <answer> and </answer>.",
            "",
            examples,
            "==",
            question,
            "{{llm()}}",
        ]
    )
