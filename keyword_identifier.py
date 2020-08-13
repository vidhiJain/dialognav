"""
Vocab contains certain keywords (which the object detection module can identify in the env)
Training a classifier to recognize the keyword from the given statements 
"""

import torch
import torchtext
from torchtext.data import TabularDataset, Field
from attention_model import AttentionModel

train_data = TabularDataset('train_data.csv', 'csv', [
    ('text', Field()),
    ('target', Field())
])

test_data = TabularDataset('test_data.csv', 'csv', [
    ('text', Field()),
    ('target', Field())
])

model = AttentionModel(
    batch_size=4, output_size=3, hidden_size=100,
    vocab_size=500, embedding_length=10, weights=None)


breakpoint()
print("done")