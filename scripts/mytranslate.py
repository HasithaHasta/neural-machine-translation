import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

import torchtext
from torchtext.vocab import Vocab
from collections import Counter

import sys
sys.path.append('../model')
import model
from model import *

# print(torch.__version__)
# print(torchtext.__version__)


def tokenizer(text):
  return [tok for tok in text.split(' ') if tok != '']

en_vocab = torch.load('../scripts/en.vocab')
hi_vocab = torch.load('../scripts/hi.vocab')

# print(len(en_vocab.itos))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_DIM = len(en_vocab)
OUTPUT_DIM = len(hi_vocab)
HID_DIM = 256
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)


SRC_PAD_IDX = en_vocab.stoi['<pad>']
TRG_PAD_IDX = hi_vocab.stoi['<pad>']

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameters')

model.load_state_dict(torch.load("../scripts/weights_15.pt", map_location = torch.device(device)))

def translate (sent, tokenize, src_vocab, trg_vocab, model, device, max_len = 40):
    model.eval()
        
    if isinstance(sent, str):
        tokens = tokenize(sent)
    else:
        tokens = [token.lower() for token in sent]

    tokens = [src_vocab['<sos>']] + tokens + [src_vocab['<eos>']]
        
    src_indexes = [src_vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor).to(device)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask).to(device)

    trg_indexes = [trg_vocab.stoi['<sos>']]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor).to(device)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.stoi['<eos>']:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:-1]  #, attention

# inp = str(input("Input: "))

# print(translate(inp, tokenizer, en_vocab, hi_vocab, model, device))

def gen_translation(inp):
    s = ''
    for i in translate(inp, tokenizer, en_vocab, hi_vocab, model, device):
        s = s+i+' '
    return s

# inp = 'I am in the house, very bored.'
# print(gen_translation(inp))