import torch
from transformers import BertModel, BertTokenizer
import bentoml


def load_model_to_bentoml(bert_version='bert-large-cased') -> None:
    # load the model #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    model = BertModel.from_pretrained(bert_version)
    # evalute the model instead of training
    model = model.eval()
    model = model.to(device)
    # save the model to bentoml #
    bentoml.pytorch.save(name="bert-model", model=model)
    bentoml.pytorch.save(name="bert-tokenizer", model=tokenizer)
