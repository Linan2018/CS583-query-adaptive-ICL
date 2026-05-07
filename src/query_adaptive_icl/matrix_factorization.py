from __future__ import annotations


def train_matrix_factorization(
    query_embeddings,
    interaction_matrix,
    *,
    epochs: int = 1600,
    lr: float = 0.01,
    batch_size: int = 16,
    loss_name: str = "mse",
    positive_weight: float | None = None,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    loss_name = loss_name.lower()
    q_tensor = torch.tensor(query_embeddings, dtype=torch.float32)
    r_tensor = torch.tensor(interaction_matrix, dtype=torch.float32)
    _, embedding_dim = q_tensor.shape
    num_combinations = r_tensor.shape[1]

    class QueryExampleDataset(Dataset):
        def __init__(self, q_values, r_values):
            self.pairs = [
                (i, j, r_values[i, j])
                for i in range(q_values.shape[0])
                for j in range(r_values.shape[1])
            ]

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            query_index, combination_index, label = self.pairs[idx]
            return (
                torch.tensor(query_index, dtype=torch.long),
                torch.tensor(combination_index, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )

    class MatrixFactorizationModel(nn.Module):
        def __init__(self, q_values, num_demo_combinations, dim):
            super().__init__()
            self.query_embeddings = q_values
            self.dim = dim
            self.demo_embeddings = nn.Embedding(num_demo_combinations, dim)
            nn.init.xavier_uniform_(self.demo_embeddings.weight)

        def forward(self, query_indices, demo_indices):
            q_embed = self.query_embeddings[query_indices] / torch.sqrt(torch.tensor(self.dim).float())
            d_embed = self.demo_embeddings(demo_indices) / torch.sqrt(torch.tensor(self.dim).float())
            return torch.sigmoid(torch.sum(q_embed * d_embed, dim=1))

    dataset = QueryExampleDataset(q_tensor, r_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MatrixFactorizationModel(q_tensor, num_combinations, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history: list[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        for query_indices, demo_indices, labels in dataloader:
            optimizer.zero_grad()
            predictions = model(query_indices, demo_indices)
            if loss_name == "bce":
                if positive_weight is None:
                    loss = F.binary_cross_entropy(predictions, labels)
                else:
                    weights = torch.where(labels == 1, torch.tensor(positive_weight), torch.tensor(1.0))
                    loss = F.binary_cross_entropy(predictions, labels, weight=weights)
            else:
                loss = torch.mean((labels - predictions) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss / len(dataloader))

    learned_demo_embeddings = model.demo_embeddings.weight.detach().cpu().numpy()
    r_hat = 1 / (1 + np.exp(-np.dot(query_embeddings / np.sqrt(embedding_dim), (learned_demo_embeddings / np.sqrt(embedding_dim)).T)))
    predicted = (r_hat > 0.5).astype(int)
    accuracy = float(np.mean(predicted == interaction_matrix))

    metrics = {
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "loss_name": loss_name,
        "positive_weight": positive_weight,
        "final_loss": loss_history[-1],
        "classification_accuracy": accuracy,
        "r_hat_mean": float(np.mean(r_hat)),
        "r_hat_std": float(np.std(r_hat)),
    }

    return learned_demo_embeddings, loss_history, metrics
