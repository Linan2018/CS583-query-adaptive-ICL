from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "bbh"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
LEGACY_DIR = ROOT / "legacy"


@dataclass(frozen=True)
class TaskPaths:
    task_name: str

    @property
    def raw_dir(self) -> Path:
        return RAW_DATA_DIR / self.task_name

    @property
    def train_file(self) -> Path:
        return self.raw_dir / "train.json"

    @property
    def test_file(self) -> Path:
        return self.raw_dir / "test.json"

    @property
    def artifacts_dir(self) -> Path:
        return ARTIFACTS_DIR / self.task_name

    @property
    def candidate_pool_dir(self) -> Path:
        return self.artifacts_dir / "candidate_pool"

    @property
    def candidate_pool_runs_dir(self) -> Path:
        return self.candidate_pool_dir / "rounds"

    @property
    def candidate_pool_accuracy_file(self) -> Path:
        return self.candidate_pool_dir / "accuracy_across_runs.json"

    @property
    def candidate_pool_consistent_file(self) -> Path:
        return self.candidate_pool_dir / "consistently_correct_queries.json"

    @property
    def candidate_pool_indexed_file(self) -> Path:
        return self.candidate_pool_dir / "indexed_candidate_pool.json"

    @property
    def interaction_matrix_dir(self) -> Path:
        return self.artifacts_dir / "interaction_matrix"

    @property
    def interaction_matrix_processed_answers_dir(self) -> Path:
        return self.interaction_matrix_dir / "processed_answers"

    @property
    def interaction_matrix_accuracy_file(self) -> Path:
        return self.interaction_matrix_dir / "accuracy.json"

    @property
    def interaction_matrix_discarded_file(self) -> Path:
        return self.interaction_matrix_dir / "discarded_combinations.json"

    @property
    def interaction_matrix_csv(self) -> Path:
        return self.interaction_matrix_dir / "matrix_results.csv"

    @property
    def interaction_matrix_summary_file(self) -> Path:
        return self.interaction_matrix_dir / "summary.json"

    @property
    def embeddings_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    @property
    def train_embeddings_file(self) -> Path:
        return self.embeddings_dir / "train_embeddings.npy"

    @property
    def test_embeddings_file(self) -> Path:
        return self.embeddings_dir / "test_embeddings.npy"

    @property
    def interaction_matrix_npy(self) -> Path:
        return self.embeddings_dir / "interaction_matrix.npy"

    @property
    def example_combinations_npy(self) -> Path:
        return self.embeddings_dir / "example_combinations.npy"

    @property
    def example_combinations_csv(self) -> Path:
        return self.embeddings_dir / "example_combinations.csv"

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "model"

    def model_loss_dir(self, loss_name: str) -> Path:
        return self.model_dir / loss_name.lower()

    def learned_example_embeddings_file(self, loss_name: str) -> Path:
        return self.model_loss_dir(loss_name) / "learned_example_embeddings.npy"

    def loss_history_file(self, loss_name: str) -> Path:
        return self.model_loss_dir(loss_name) / "loss_history.json"

    def train_metrics_file(self, loss_name: str) -> Path:
        return self.model_loss_dir(loss_name) / "train_metrics.json"

    @property
    def recommendations_dir(self) -> Path:
        return self.artifacts_dir / "recommendations"

    def recommendation_loss_dir(self, loss_name: str) -> Path:
        return self.recommendations_dir / loss_name.lower()

    def predicted_r_hat_npy(self, loss_name: str) -> Path:
        return self.recommendation_loss_dir(loss_name) / "predicted_R_hat.npy"

    def predicted_r_hat_csv(self, loss_name: str) -> Path:
        return self.recommendation_loss_dir(loss_name) / "predicted_R_hat.csv"

    def best_example_ids_npy(self, loss_name: str) -> Path:
        return self.recommendation_loss_dir(loss_name) / "best_example_ids.npy"

    def best_example_ids_json(self, loss_name: str) -> Path:
        return self.recommendation_loss_dir(loss_name) / "best_example_ids.json"

    def ranked_recommendations_json(self, loss_name: str) -> Path:
        return self.recommendation_loss_dir(loss_name) / "ranked_recommendations.json"

    @property
    def evaluation_dir(self) -> Path:
        return self.artifacts_dir / "evaluation"

    def evaluation_loss_dir(self, loss_name: str) -> Path:
        return self.evaluation_dir / loss_name.lower()

    def evaluation_results_file(self, loss_name: str) -> Path:
        return self.evaluation_loss_dir(loss_name) / "test_processed_answers.json"

    def evaluation_accuracy_file(self, loss_name: str) -> Path:
        return self.evaluation_loss_dir(loss_name) / "test_accuracy.json"

    def ensure_directories(self) -> None:
        for directory in [
            self.raw_dir,
            self.candidate_pool_dir,
            self.candidate_pool_runs_dir,
            self.interaction_matrix_dir,
            self.interaction_matrix_processed_answers_dir,
            self.embeddings_dir,
            self.model_dir,
            self.recommendations_dir,
            self.evaluation_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def build_task_paths(task_name: str) -> TaskPaths:
    paths = TaskPaths(task_name=task_name)
    paths.ensure_directories()
    return paths
