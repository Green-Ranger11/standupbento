import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_version = 'bert-large-cased'
tokenizer = BertTokenizer.from_pretrained(bert_version)
model = BertModel.from_pretrained(bert_version)

# TODO: import model into BentoML

# TODO: show the reduced embedding space


def visualize(distances, figsize=(10, 5), titles=None):
    plt.show()


# evalute the model instead of training
model = model.eval()
model = model.to(device)

texts = [
    'Have you gotten the API to work?',
    'The API seems broken! HELP!',
    'It seems like there\'s a bug in the API, I\'ll try fixing it soon!',
    'I\'m having trouble getting the API to work!',
    'I love the new UI',
    'Who designed the UI? It looks wonderful!',
]

encodings = tokenizer(
    texts,
    padding=True,
    return_tensors='pt'
).to(device)

# disable gradient descent
with torch.no_grad():
    embeds = model(**encodings)
embeds = embeds[0]

CLSs = embeds[:, 0, :]

# normalize the CLS token embeddings
normalized = f.normalize(CLSs, p=2, dim=1)
# cosine similarity
cls_dist = normalized.matmul(normalized.T)
cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
cls_dist = cls_dist.numpy()

# visualize

visualize([cls_dist], titles=["CLS"])
