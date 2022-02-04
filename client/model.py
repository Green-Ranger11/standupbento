import plotly.express as px
import numpy as np
import torch
import torch.nn.functional as f
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import BertModel, BertTokenizer
import bentoml
from save_bento import load_model_to_bentoml


class BertModel:
    def __init__(self):
        self.model, self.tokenizer = \
            self.load_model_from_bentoml()
        self.device = self.get_device()
        self.tsne = TSNE(n_components=3)
        self.pca = PCA(n_components=3)

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

    def reduce_dimensionality(self, responses, method):
        if method == 'pca':
            XY = self.pca.fit_transform(
                self.embed_responses(responses)
            )
        elif method == 'tsne':
            XY = self.tsne.fit_transform(
                self.embed_responses(responses)
            )
        return XY

    def visualize(self, responses, method='pca'):
        XY = self.reduce_dimensionality(responses, method)
        # basic coloring scheme
        color = []
        for response in responses:
            if 'UI' in response:
                color.append('red')
            elif 'API' in response:
                color.append('blue')
            else:
                color.append('green')
        fig = px.scatter_3d(
            XY, x=0, y=1, z=2,
            color=color,
            hover_data={
                'response': responses
            },
            title=f'{method.upper()} visualization of responses',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()
