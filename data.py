import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
import torch

def title_tokenize(t):
    return t[:t.find('(') - 1]

def desc_tokenize(d):
    return d[d.find('-')+2:]
    
title_field = Field(lower=True, init_token='<sos>', eos_token='<eos>', tokenize=title_tokenize)
desc_field = Field(lower=True, init_token='<sos>', eos_token='<eos>', tokenize=desc_tokenize)

fields = {
    'Title' : ('t', title_field),
    'Description' : ('d', desc_field),
}

train_data, test_data = TabularDataset.splits(
    path='/home/nnpy/dataset/ag-news',
    train='/home/nnpy/dataset/ag-news/train.csv',
    test='/home/nnpy/dataset/ag-news/test.csv',
    fields = fields,
    format='csv',
)

title_field.build_vocab(train_data, max_size=10_000, min_freq=2)
desc_field.build_vocab(train_data, max_size=10_000, min_freq=2)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    sort_within_batch = True,
    sort_key = lambda x : len(x.t),
    device='cuda',
)
