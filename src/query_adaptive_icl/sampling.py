from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def load_indexed_examples(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_indexed_examples(examples: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    indexed = {str(index): example for index, example in enumerate(examples)}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(indexed, handle, ensure_ascii=False, indent=2)
    return indexed


def format_examples(example_records: Iterable[dict[str, str]]) -> str:
    return "\n\n".join(
        f"Q: {example['question']}\nA: {example['answer']}" for example in example_records
    )


def format_examples_by_ids(indexed_examples: dict[str, dict[str, str]], example_ids: Iterable[str]) -> str:
    return format_examples(indexed_examples[str(example_id)] for example_id in example_ids)


def sample_example_ids(
    indexed_examples: dict[str, dict[str, str]],
    *,
    max_samples: int = 3,
    rng: random.Random | None = None,
) -> list[str]:
    rng = rng or random
    keys = sorted(indexed_examples.keys(), key=int)
    sample_size = rng.randint(1, min(max_samples, len(keys)))
    return rng.sample(keys, sample_size)


def combination_label(example_ids: Iterable[str]) -> str:
    return "_".join(sorted((str(example_id) for example_id in example_ids), key=int))
