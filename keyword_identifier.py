"""
Vocab contains certain keywords (which the object detection module can identify in the env)
Training a classifier to recognize the keyword from the given statements 
"""

import torch
import torchtext
from torchtext.data import TabularDataset, Field

train_data = TabularDataset('train_data.csv', 'csv', [
    ('text', Field()),
    ('target', Field())
], fieldnames=['text', 'target'])

test_data = TabularDataset('test_data.csv', 'csv', [
    ('text', Field()),
    ('target', Field())
])

class Model(nn.Module):
    def __init__(self):
        


breakpoint()
