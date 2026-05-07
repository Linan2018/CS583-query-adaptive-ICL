from __future__ import annotations


class BertEncoder:
    """Frozen BERT encoder for query text."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        from transformers import BertModel, BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def batch_encode(self, strings: list[str]):
        import torch

        inputs = self.tokenizer(strings, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
