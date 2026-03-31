"""
Based closely on Karpathy Shakespeare character-level prepare.py.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import random
import numpy as np


# alphabet = [c for c in 'abcdefghijklmnopqrstuvwxyz']
alphabet = [
    "1930 Uruguay\n",
    "1934 Italy\n",
    "1938 Italy\n",
    "1950 Uruguay\n",
    "1954 Germany\n",
    "1958 Brazil\n",
    "1962 Brazil\n",
    "1966 England\n",
    "1970 Brazil\n",
    "1974 Germany\n",
    "1978 Argentina\n",
    "1982 Italy\n",
    "1986 Argentina\n",
    "1990 Germany\n",
    "1994 Brazil\n",
    "1998 France\n",
    "2002 Brazil\n",
    "2006 Italy\n",
    "2010 Spain\n",
    "2014 Germany\n",
    "2018 France\n",
    "2022 Argentina\n"
]




target_length = 1_200_000  # about 1MB will be plenty of training data, with 0.2MB validation

# construct a string consisting of repeated copies of the building_blocks in random order, 
# such that it has length at least target_length
total_length = 0
lines = []
while total_length < target_length:
    for line in alphabet:
        lines.append(line)
        total_length = total_length + len(line)
        if total_length>= target_length:
            break

print("First 20 lines of data:")
for i in range(20):
    print(lines[i].strip())

data = ''.join(lines)
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: |{'|'.join(map(repr, chars))}|")
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    # encoder: take a string, output a list of integers
    return [stoi[c] for c in s]


def decode(l):
    # decoder: take a list of integers, output a string
    return ''.join([itos[i] for i in l])


# split training and validation data 80/20
tr_proportion = 0.8
cutoff = int(len(lines) * tr_proportion)
train_data = ''.join(lines[:cutoff])
val_data = ''.join(lines[cutoff:])

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

