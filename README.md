# Query-Adaptive Retrieval and Ranking of In-Context Examples for LLM Inference

**Authors**  
Tian Tan, Linan Zheng, Xinyu Guo

This repository implements the CS583 project on query-adaptive retrieval and ranking of in-context examples for LLM inference.

The codebase is organized as follows:

- `src/`: reusable source code and the pipeline CLI
- `scripts/`: thin entrypoints for running the pipeline
- `data/`: raw datasets only
- `artifacts/`: generated intermediate files and experiment outputs
- `reports/`: project report and writeups

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── query_adaptive_icl/
├── data/
│   └── raw/
│       └── bbh/
│           └── causal_judgement/
│               ├── train.json
│               └── test.json
├── artifacts/
│   └── causal_judgement/
│       ├── candidate_pool/
│       ├── interaction_matrix/
│       ├── embeddings/
│       ├── model/
│       ├── recommendations/
│       └── evaluation/
└── reports/
    └── CS583_project_final.pdf
```

## Main Pipeline

The project follows five stages:

1. `stage1-candidate-pool`
   Build a reliable demo pool by repeatedly running zero-shot inference on the training set and keeping queries that are consistently answered correctly.

2. `stage2-interaction-matrix`
   Randomly sample demo combinations from the candidate pool, run them against the training queries, and build the query-demo interaction matrix.

3. `stage3-encode` + `stage3-prepare-matrix` + `stage3-train`
   Encode queries with BERT, convert the interaction matrix to NumPy arrays, and train demo-combination embeddings with matrix factorization.

4. `stage4-recommend`
   Use the learned demo-combination embeddings to recommend the best demo combination for each test query.

5. `stage5-evaluate`
   Re-run the test set with the recommended demo combinations and compute final accuracy.

## Current Included Data and Outputs

The repo already contains:

- Raw BBH data for `causal_judgement` in `data/raw/bbh/causal_judgement/`
- Historical curated artifacts in `artifacts/causal_judgement/`
- The original unrefactored experiment files in `legacy/`

That means you can either:

- inspect the current outputs directly under `artifacts/`, or
- rerun the pipeline from scratch with the new CLI

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Gemini API key before running any LLM-dependent stage:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

`stage1`, `stage2`, and `stage5` call Gemini. `stage3` and `stage4` are local.

## How To Run

All commands below run from the repo root.

### 1. Build the candidate demo pool

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage1-candidate-pool \
  --rounds 6 \
  --batch-size 15
```

Outputs:

- `artifacts/causal_judgement/candidate_pool/consistently_correct_queries.json`
- `artifacts/causal_judgement/candidate_pool/indexed_candidate_pool.json`
- `artifacts/causal_judgement/candidate_pool/rounds/`

### 2. Build the interaction matrix

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage2-interaction-matrix \
  --max-iter 20 \
  --threshold 95 \
  --min-correct-rounds 1 \
  --max-samples 3
```

If you want to reduce LLM randomness with majority voting, increase `--repeat-per-query`:

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage2-interaction-matrix \
  --repeat-per-query 3
```

Outputs:

- `artifacts/causal_judgement/interaction_matrix/matrix_results.csv`
- `artifacts/causal_judgement/interaction_matrix/processed_answers/`
- `artifacts/causal_judgement/interaction_matrix/accuracy.json`

### 3. Encode queries and prepare training arrays

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage3-encode --splits train test
python3 scripts/run_pipeline.py --task causal_judgement stage3-prepare-matrix
```

Outputs:

- `artifacts/causal_judgement/embeddings/train_embeddings.npy`
- `artifacts/causal_judgement/embeddings/test_embeddings.npy`
- `artifacts/causal_judgement/embeddings/interaction_matrix.npy`
- `artifacts/causal_judgement/embeddings/example_combinations.npy`

### 4. Train the matrix-factorization model

Default MSE version:

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage3-train \
  --loss mse \
  --epochs 1600 \
  --lr 0.01 \
  --batch-size 16
```

Optional BCE version:

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage3-train \
  --loss bce \
  --epochs 1600 \
  --lr 0.01 \
  --batch-size 16
```

Outputs:

- `artifacts/causal_judgement/model/mse/learned_example_embeddings.npy`
- `artifacts/causal_judgement/model/mse/loss_history.json`
- `artifacts/causal_judgement/model/mse/train_metrics.json`

### 5. Recommend demo combinations for the test set

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage4-recommend --loss mse --top-k 1
```

Outputs:

- `artifacts/causal_judgement/recommendations/mse/best_example_ids.json`
- `artifacts/causal_judgement/recommendations/mse/best_example_ids.npy`
- `artifacts/causal_judgement/recommendations/mse/predicted_R_hat.csv`

### 6. Evaluate the test set with recommended demos

```bash
python3 scripts/run_pipeline.py --task causal_judgement stage5-evaluate --loss mse
```

Outputs:

- `artifacts/causal_judgement/evaluation/mse/test_processed_answers.json`
- `artifacts/causal_judgement/evaluation/mse/test_accuracy.json`
# CS583-query-adaptive-ICL
