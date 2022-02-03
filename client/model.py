import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from transformers import BertModel, BertTokenizer
import bentoml
from save_bento import load_model_to_bentoml


class BertModel:
    def __init__(self):
        self.model, self.tokenizer = \
            self.load_model_from_bentoml()
        self.device = self.get_device()

    def load_model_from_bentoml(self):
        try:
            model = bentoml.pytorch.load('bert-model')
            tokenizer = bentoml.pytorch.load('bert-tokenizer')
        except FileNotFoundError:
            load_model_to_bentoml()
            model = bentoml.pytorch.load('bert-model')
            tokenizer = bentoml.pytorch.load('bert-tokenizer')
        return model, tokenizer

    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def embed_responses(self, responses) -> np.array:
        encodings = self.tokenizer(
            responses,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        # disable gradient descent
        with torch.no_grad():
            embeds = self.model(**encodings)
        embeds = embeds[0]
        CLSs = embeds[:, 0, :]
        # normalize the CLS token embeddings
        normalized = f.normalize(CLSs, p=2, dim=1)
        # cosine similarity
        cls_dist = normalized.matmul(normalized.T)
        cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
        cls_dist = cls_dist.numpy()
        return cls_dist

    def visualize(self, distances, figsize=(10, 5), titles=None):
        # TODO: show the reduced embedding space
        plt.show()
