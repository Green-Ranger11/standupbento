import torch
from transformers import BertModel, BertTokenizer
import bentoml


# load the model #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-cased')
# evalute the model instead of training
model = model.eval()
model = model.to(device)
# save the model to bentoml #
bentoml.pytorch.save(name="bert-model", model=model)
bentoml.pytorch.save(name="bert-tokenizer", model=tokenizer)