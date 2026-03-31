import os
import numpy as np
import pickle
import sys

# allow importing generators.py from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from generators_sub import generate1DigitSimpleExamples

target_length = 1_200_000  # ~1MB total characters
total_length = 0

lines = []
while total_length < target_length:
    # generate1DigitSimpleExamples now returns "a-b=c\n"
    example = generate1DigitSimpleExamples()
    lines.append(example)
    total_length += len(example)

print("First 20 lines of data:")
for i in range(20):
    print(lines[i].strip())

# join all examples into one long string
data = ''.join(lines)
print(f"length of dataset in characters: {len(data):,}")

# get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: |{'|'.join(map(repr, chars))}|")
print(f"vocab size: {vocab_size:,}")

# character ↔ integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# train / validation split (by examples, not characters)
tr_proportion = 0.8
cutoff = int(len(lines) * tr_proportion)

train_data = ''.join(lines[:cutoff])
val_data = ''.join(lines[cutoff:])

# encode to integers
train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# save binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

base_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

# save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)