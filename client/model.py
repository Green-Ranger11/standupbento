import plotly.express as px
import numpy as np
import torch
import torch.nn.functional as f
from sklearn.manifold import TSNE
from sklearn import decomposition
from transformers import BertModel, BertTokenizer
import bentoml
from save_bento import load_model_to_bentoml


class BertModel:
    def __init__(self):
        self.model, self.tokenizer = \
            self.load_model_from_bentoml()
        self.device = self.get_device()
        self.tsne = TSNE(n_components=2)
        self.pca = decomposition.PCA(n_components=2)

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
        CLSs = embeds[0][:, 0, :]
        # normalize the CLS token embeddings
        normalized = f.normalize(CLSs, p=2, dim=1)
        return normalized

    def visualize(self, responses, type='pca'):
        if type == 'pca':
            XY = self.pca.fit_transform(
                self.embed_responses(responses)
            )
        elif type == 'tsne':
            XY = self.tsne.fit_transform(
                self.embed_responses(responses)
            )
        plt.scatter(*zip(*XY))