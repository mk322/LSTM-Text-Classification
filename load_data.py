#from cProfile import label
import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    tokenize = lambda x: x.split(" ")
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()
    '''
    train_data = ''
    train_label = []
    valid_data = []
    valid_label = []
    with open('train.tsv', 'r', encoding='cp850') as t:
        lines = t.read().splitlines()
        for line in lines:
            parts = line.split('\t')
            train_data+=parts[0]
            train_label.append(int(parts[1]))
    with open('valid.tsv', 'r', encoding='cp850') as v:
        lines = v.read().splitlines()
        for line in lines:
            parts = line.split('\t')
            valid_data.append(parts[0])
            valid_label.append(int(parts[1]))
    print(len(train_data))
    print(set(train_label))
    '''
    train_data, valid_data = data.TabularDataset.splits(path='', format='tsv', train='train1.tsv', validation='valid.tsv', skip_header=True, fields=[('text', TEXT), ('label', LABEL)])
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)
    #print(LABEL.vocab.stoi)
    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    #train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=64, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter

load_dataset()