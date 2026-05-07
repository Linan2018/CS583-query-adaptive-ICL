from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

from .config import TaskPaths
from .llm import create_gemini_model, extract_answer, request_gemini_text
from .prompting import generate_prompt
from .sampling import combination_label, format_examples_by_ids, load_indexed_examples, sample_example_ids, write_indexed_examples
from .task_type import detect_task_type


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_candidate_pool(
    paths: TaskPaths,
    *,
    rounds: int = 6,
    batch_size: int = 15,
    model_name: str = "gemini-1.5-pro-001",
    sleep_seconds: float = 5.0,
) -> None:
    queries = load_json(paths.train_file)
    if not queries:
        raise RuntimeError(f"No training queries found in {paths.train_file}")

    task_type = detect_task_type(queries[0]["question"])
    model = create_gemini_model(model_name)

    accuracy_log: dict[str, float] = {}
    correct_index_sets: list[set[int]] = []

    for run_id in range(rounds):
        run_results = []
        correct_indices: set[int] = set()
        correct_count = 0

        for batch_start in range(0, len(queries), batch_size):
            batch = queries[batch_start : batch_start + batch_size]
            for offset, query in enumerate(batch):
                prompt = generate_prompt(query["question"], task_type, examples="")
                response_text = request_gemini_text(model, prompt)
                predicted_answer = extract_answer(response_text)
                is_correct = int(predicted_answer == query["answer"])
                query_index = batch_start + offset
                if is_correct:
                    correct_indices.add(query_index)
                correct_count += is_correct
                run_results.append(
                    {
                        "query_index": query_index,
                        "question": query["question"],
                        "answer": query["answer"],
                        "predicted_answer": predicted_answer,
                        "correct": is_correct,
                    }
                )
            time.sleep(sleep_seconds)

        accuracy = round(correct_count / len(queries) * 100, 2)
        accuracy_log[f"run_{run_id}"] = accuracy
        correct_index_sets.append(correct_indices)

        run_dir = paths.candidate_pool_runs_dir / f"run_{run_id}"
        save_json(run_dir / "processed_answers.json", run_results)
        save_json(run_dir / "correct_answers.json", [item for item in run_results if item["correct"] == 1])

    consistent_indices = sorted(set.intersection(*correct_index_sets)) if correct_index_sets else []
    consistent_queries = [queries[index] for index in consistent_indices]

    save_json(paths.candidate_pool_accuracy_file, accuracy_log)
    save_json(paths.candidate_pool_consistent_file, consistent_queries)
    write_indexed_examples(consistent_queries, paths.candidate_pool_indexed_file)


def build_interaction_matrix(
    paths: TaskPaths,
    *,
    max_iter: int = 20,
    threshold: float = 95.0,
    min_correct_rounds: int = 1,
    min_accuracy: float = 64.0,
    max_samples: int = 3,
    repeat_per_query: int = 1,
    model_name: str = "gemini-1.5-pro-001",
    seed: int = 7,
    sleep_seconds: float = 5.0,
) -> None:
    train_queries = load_json(paths.train_file)
    indexed_examples = load_indexed_examples(paths.candidate_pool_indexed_file)
    task_type = detect_task_type(train_queries[0]["question"])
    rng = random.Random(seed)
    model = create_gemini_model(model_name)

    used_combinations: set[frozenset[str]] = set()
    accepted_combination_labels: list[str] = []
    results_matrix: list[list[int]] = []
    accuracy_records: list[dict[str, object]] = []
    discarded_records: list[dict[str, object]] = []

    while len(accepted_combination_labels) < max_iter:
        sampled_ids = sample_example_ids(indexed_examples, max_samples=max_samples, rng=rng)
        sampled_key = frozenset(sampled_ids)
        if sampled_key in used_combinations:
            continue
        used_combinations.add(sampled_key)

        label = combination_label(sampled_ids)
        formatted_examples = format_examples_by_ids(indexed_examples, sampled_ids)

        iteration_results: list[int] = []
        processed_answers: list[dict[str, object]] = []
        wrong_query_ids: list[int] = []

        for query_index, query in enumerate(train_queries):
            predictions = []
            for _ in range(repeat_per_query):
                prompt = generate_prompt(query["question"], task_type, formatted_examples)
                response_text = request_gemini_text(model, prompt)
                predictions.append(extract_answer(response_text))
            final_prediction = Counter(predictions).most_common(1)[0][0]
            is_correct = int(final_prediction == query["answer"])
            iteration_results.append(is_correct)
            if not is_correct:
                wrong_query_ids.append(query_index)
            processed_answers.append(
                {
                    "query_index": query_index,
                    "question": query["question"],
                    "answer": query["answer"],
                    "predicted_answer": final_prediction,
                    "all_predictions": predictions,
                    "example_ids": sampled_ids,
                    "combination_label": label,
                    "correct": is_correct,
                }
            )
        time.sleep(sleep_seconds)

        accuracy = round(sum(iteration_results) / len(train_queries) * 100, 2)
        if accuracy < min_accuracy:
            discarded_records.append(
                {
                    "combination_label": label,
                    "sampled_example_ids": sampled_ids,
                    "accuracy": accuracy,
                    "wrong_query_ids": wrong_query_ids,
                }
            )
            continue

        accepted_combination_labels.append(label)
        results_matrix.append(iteration_results)
        accuracy_records.append(
            {
                "iteration": len(accepted_combination_labels),
                "combination_label": label,
                "sampled_example_ids": sampled_ids,
                "accuracy": accuracy,
            }
        )
        save_json(
            paths.interaction_matrix_processed_answers_dir / f"{len(accepted_combination_labels):02d}_{label}.json",
            processed_answers,
        )

        cumulative_correct_counts = [0] * len(train_queries)
        for accepted_results in results_matrix:
            for index, value in enumerate(accepted_results):
                cumulative_correct_counts[index] += value
        covered = sum(1 for count in cumulative_correct_counts if count >= min_correct_rounds)
        cumulative_accuracy = round(covered / len(train_queries) * 100, 2)
        if cumulative_accuracy >= threshold:
            break

    save_csv(paths.interaction_matrix_csv, accepted_combination_labels, list(map(list, zip(*results_matrix))) if results_matrix else [])
    save_json(paths.interaction_matrix_accuracy_file, accuracy_records)
    save_json(paths.interaction_matrix_discarded_file, discarded_records)
    save_json(
        paths.interaction_matrix_summary_file,
        {
            "accepted_combinations": len(accepted_combination_labels),
            "threshold": threshold,
            "min_correct_rounds": min_correct_rounds,
            "min_accuracy": min_accuracy,
            "repeat_per_query": repeat_per_query,
            "matrix_shape": [len(train_queries), len(accepted_combination_labels)],
        },
    )


def encode_queries(paths: TaskPaths, *, splits: tuple[str, ...] = ("train", "test")) -> None:
    import numpy as np

    from .embeddings import BertEncoder

    encoder = BertEncoder()
    for split in splits:
        source_file = paths.train_file if split == "train" else paths.test_file
        output_file = paths.train_embeddings_file if split == "train" else paths.test_embeddings_file
        data = load_json(source_file)
        questions = [item["question"] for item in data]
        embeddings = encoder.batch_encode(questions)
        np.save(output_file, embeddings)


def prepare_matrix_arrays(paths: TaskPaths) -> None:
    import numpy as np

    with paths.interaction_matrix_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        matrix = [[int(value) for value in row] for row in reader]

    np.save(paths.interaction_matrix_npy, np.array(matrix, dtype=np.float32))
    np.save(paths.example_combinations_npy, np.array(header, dtype=object))
    save_csv(paths.example_combinations_csv, ["combination_label"], [[label] for label in header])


def train_demo_embeddings(
    paths: TaskPaths,
    *,
    epochs: int = 1600,
    lr: float = 0.01,
    batch_size: int = 16,
    loss_name: str = "mse",
    positive_weight: float | None = None,
) -> None:
    import numpy as np

    from .matrix_factorization import train_matrix_factorization

    query_embeddings = np.load(paths.train_embeddings_file)
    interaction_matrix = np.load(paths.interaction_matrix_npy)
    learned_embeddings, loss_history, metrics = train_matrix_factorization(
        query_embeddings,
        interaction_matrix,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        loss_name=loss_name,
        positive_weight=positive_weight,
    )

    output_dir = paths.model_loss_dir(loss_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.learned_example_embeddings_file(loss_name), learned_embeddings)
    save_json(paths.loss_history_file(loss_name), loss_history)
    save_json(paths.train_metrics_file(loss_name), metrics)


def recommend_demos(paths: TaskPaths, *, loss_name: str = "mse", top_k: int = 1) -> None:
    import numpy as np

    learned_embeddings = np.load(paths.learned_example_embeddings_file(loss_name))
    query_embeddings = np.load(paths.test_embeddings_file)
    combination_labels = np.load(paths.example_combinations_npy, allow_pickle=True)

    embedding_dim = query_embeddings.shape[1]
    normalized_queries = query_embeddings / np.sqrt(embedding_dim)
    normalized_demos = learned_embeddings / np.sqrt(embedding_dim)
    r_hat = 1 / (1 + np.exp(-np.dot(normalized_queries, normalized_demos.T)))

    ranked_indices = np.argsort(-r_hat, axis=1)
    ranked_labels = [[str(combination_labels[idx]) for idx in row[:top_k]] for row in ranked_indices]
    best_labels = [labels[0] for labels in ranked_labels]

    recommendation_dir = paths.recommendation_loss_dir(loss_name)
    recommendation_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.predicted_r_hat_npy(loss_name), r_hat)
    np.save(paths.best_example_ids_npy(loss_name), np.array(best_labels, dtype=object))
    save_json(paths.best_example_ids_json(loss_name), best_labels)
    save_json(paths.ranked_recommendations_json(loss_name), ranked_labels)

    csv_header = ["query_index", "best_example_id", *map(str, combination_labels)]
    csv_rows = [
        [query_index, best_labels[query_index], *list(map(float, scores))]
        for query_index, scores in enumerate(r_hat)
    ]
    save_csv(paths.predicted_r_hat_csv(loss_name), csv_header, csv_rows)


def evaluate_test_queries(
    paths: TaskPaths,
    *,
    loss_name: str = "mse",
    batch_size: int = 15,
    model_name: str = "gemini-1.5-pro-001",
    best_ids_source: str = "json",
    sleep_seconds: float = 15.0,
) -> None:
    import numpy as np

    test_queries = load_json(paths.test_file)
    indexed_examples = load_indexed_examples(paths.candidate_pool_indexed_file)
    task_type = detect_task_type(test_queries[0]["question"])
    model = create_gemini_model(model_name)

    if best_ids_source == "npy" or not paths.best_example_ids_json(loss_name).exists():
        best_example_ids = np.load(paths.best_example_ids_npy(loss_name), allow_pickle=True).tolist()
    else:
        best_example_ids = load_json(paths.best_example_ids_json(loss_name))

    results = []
    correct_count = 0
    for batch_start in range(0, len(test_queries), batch_size):
        batch_queries = test_queries[batch_start : batch_start + batch_size]
        batch_example_ids = best_example_ids[batch_start : batch_start + batch_size]
        for offset, (query, example_label) in enumerate(zip(batch_queries, batch_example_ids)):
            example_ids = str(example_label).split("_")
            prompt = generate_prompt(
                query["question"],
                task_type,
                examples=format_examples_by_ids(indexed_examples, example_ids),
            )
            response_text = request_gemini_text(model, prompt)
            predicted_answer = extract_answer(response_text)
            is_correct = int(predicted_answer == query["answer"])
            correct_count += is_correct
            results.append(
                {
                    "query_index": batch_start + offset,
                    "question": query["question"],
                    "answer": query["answer"],
                    "predicted_answer": predicted_answer,
                    "best_example_id": example_label,
                    "correct": is_correct,
                }
            )
        time.sleep(sleep_seconds)

    accuracy = round(correct_count / len(test_queries) * 100, 2)
    save_json(paths.evaluation_results_file(loss_name), results)
    save_json(paths.evaluation_accuracy_file(loss_name), {"accuracy": accuracy})
