from __future__ import annotations

import argparse

from .config import build_task_paths
from .pipeline import (
    build_candidate_pool,
    build_interaction_matrix,
    encode_queries,
    evaluate_test_queries,
    prepare_matrix_arrays,
    recommend_demos,
    train_demo_embeddings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query-adaptive ICL pipeline")
    parser.add_argument("--task", default="causal_judgement", help="Task name under data/raw/bbh/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_stage1 = subparsers.add_parser("stage1-candidate-pool", help="Build a stable candidate demo pool")
    parser_stage1.add_argument("--rounds", type=int, default=6)
    parser_stage1.add_argument("--batch-size", type=int, default=15)
    parser_stage1.add_argument("--model-name", default="gemini-1.5-pro-001")
    parser_stage1.add_argument("--sleep-seconds", type=float, default=5.0)

    parser_stage2 = subparsers.add_parser("stage2-interaction-matrix", help="Sample demo combinations and build the interaction matrix")
    parser_stage2.add_argument("--max-iter", type=int, default=20)
    parser_stage2.add_argument("--threshold", type=float, default=95.0)
    parser_stage2.add_argument("--min-correct-rounds", type=int, default=1)
    parser_stage2.add_argument("--min-accuracy", type=float, default=64.0)
    parser_stage2.add_argument("--max-samples", type=int, default=3)
    parser_stage2.add_argument("--repeat-per-query", type=int, default=1)
    parser_stage2.add_argument("--model-name", default="gemini-1.5-pro-001")
    parser_stage2.add_argument("--seed", type=int, default=7)
    parser_stage2.add_argument("--sleep-seconds", type=float, default=5.0)

    parser_stage3_encode = subparsers.add_parser("stage3-encode", help="Encode train and test queries with BERT")
    parser_stage3_encode.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])

    subparsers.add_parser("stage3-prepare-matrix", help="Convert matrix_results.csv into NumPy arrays")

    parser_stage3_train = subparsers.add_parser("stage3-train", help="Train demo-combination embeddings")
    parser_stage3_train.add_argument("--epochs", type=int, default=1600)
    parser_stage3_train.add_argument("--lr", type=float, default=0.01)
    parser_stage3_train.add_argument("--batch-size", type=int, default=16)
    parser_stage3_train.add_argument("--loss", choices=["mse", "bce"], default="mse")
    parser_stage3_train.add_argument("--positive-weight", type=float)

    parser_stage4 = subparsers.add_parser("stage4-recommend", help="Recommend demo combinations for the test set")
    parser_stage4.add_argument("--loss", choices=["mse", "bce"], default="mse")
    parser_stage4.add_argument("--top-k", type=int, default=1)

    parser_stage5 = subparsers.add_parser("stage5-evaluate", help="Evaluate the test set with recommended demos")
    parser_stage5.add_argument("--loss", choices=["mse", "bce"], default="mse")
    parser_stage5.add_argument("--batch-size", type=int, default=15)
    parser_stage5.add_argument("--model-name", default="gemini-1.5-pro-001")
    parser_stage5.add_argument("--best-ids-source", choices=["json", "npy"], default="json")
    parser_stage5.add_argument("--sleep-seconds", type=float, default=15.0)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = build_task_paths(args.task)

    if args.command == "stage1-candidate-pool":
        build_candidate_pool(
            paths,
            rounds=args.rounds,
            batch_size=args.batch_size,
            model_name=args.model_name,
            sleep_seconds=args.sleep_seconds,
        )
    elif args.command == "stage2-interaction-matrix":
        build_interaction_matrix(
            paths,
            max_iter=args.max_iter,
            threshold=args.threshold,
            min_correct_rounds=args.min_correct_rounds,
            min_accuracy=args.min_accuracy,
            max_samples=args.max_samples,
            repeat_per_query=args.repeat_per_query,
            model_name=args.model_name,
            seed=args.seed,
            sleep_seconds=args.sleep_seconds,
        )
    elif args.command == "stage3-encode":
        encode_queries(paths, splits=tuple(args.splits))
    elif args.command == "stage3-prepare-matrix":
        prepare_matrix_arrays(paths)
    elif args.command == "stage3-train":
        train_demo_embeddings(
            paths,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            loss_name=args.loss,
            positive_weight=args.positive_weight,
        )
    elif args.command == "stage4-recommend":
        recommend_demos(paths, loss_name=args.loss, top_k=args.top_k)
    elif args.command == "stage5-evaluate":
        evaluate_test_queries(
            paths,
            loss_name=args.loss,
            batch_size=args.batch_size,
            model_name=args.model_name,
            best_ids_source=args.best_ids_source,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()
