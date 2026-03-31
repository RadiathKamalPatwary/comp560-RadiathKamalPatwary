"""
Evaluate the accuracy of a trained model (SUBTRACTION)
"""
import os
import pickle
from contextlib import nullcontext
import torch
import sys
import argparse

# add comp560-nanogpt to path
nanogpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../comp560-nanoGPT"))
sys.path.append(nanogpt_path)

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--train_digits', type=int, required=True)
parser.add_argument('--eval_digits', type=int, required=True)
parser.add_argument('--pad_eval', action='store_true')
args = parser.parse_args()

dataset = args.dataset
eval_digits = args.eval_digits

if args.eval_digits == 1:
    eval_low, eval_high = 0, 10
elif args.eval_digits == 2:
    eval_low, eval_high = 10, 100
else:
    raise ValueError("Only 1 or 2 digits supported")

out_dir = f'out/{dataset}'
init_from = 'resume'
start = "\n"
num_samples = 10
max_new_tokens = 500
temperature = 1.0
top_k = 1
seed = 1337
device = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# load meta for encoding/decoding
meta_path = f'data/{dataset}/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def generateAnswer(prompt):
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.eval_digits + 1,
            temperature=1.0,
            top_k=1
        )

    output = decode(y[0].tolist())
    generated = output[len(prompt):]
    return generated.strip()

def has_borrow(a, b):
    """
    Borrow happens in subtraction when any digit of a is smaller than the corresponding digit of b
    after accounting for any previous borrow.
    Here we only support 1 or 2 digit evaluation, but this works generally.
    Assumes a >= b.
    """
    borrow = 0
    while a > 0 or b > 0:
        a_digit = (a % 10) - borrow
        b_digit = (b % 10)
        if a_digit < b_digit:
            return True
        borrow = 1 if a_digit < b_digit else 0
        a //= 10
        b //= 10
    return False

def build_prompt(a, b):
    if args.pad_eval:
        return f"{a:0{args.train_digits}d}-{b:0{args.train_digits}d}="
    else:
        return f"{a}-{b}="

# check for accuracy
total_no_borrow = 0
correct_no_borrow = 0

total_borrow = 0
correct_borrow = 0

# to analyse borrow predictions
from collections import Counter
borrow_predictions = Counter()
length_counter = Counter()
printed = 0

# loop over all valid (a,b) with a >= b
if args.eval_digits == 1:
    a_range = range(0, 10)
    for a in a_range:
        for b in range(0, a + 1):

            prompt = build_prompt(a, b)

            if args.pad_eval:
                correct_answer = f"0{a-b}"
            else:
                correct_answer = str(a - b)

            model_answer = generateAnswer(prompt)
            length_counter[len(model_answer)] += 1

            if not has_borrow(a, b):
                total_no_borrow += 1
                if model_answer == correct_answer:
                    correct_no_borrow += 1
            else:
                # For 1-digit subtraction this should basically never happen,
                # but keeping structure consistent.
                if printed < 10:
                    print(f"{prompt}{model_answer} Correct Answer: {correct_answer}")
                    printed += 1
                total_borrow += 1
                borrow_predictions[model_answer] += 1
                if model_answer == correct_answer:
                    correct_borrow += 1

elif args.eval_digits == 2:
    for a in range(10, 100):
        for b in range(10, a + 1):  # ensures a >= b

            prompt = build_prompt(a, b)

            if args.pad_eval:
                correct_answer = f"{(a-b):0{args.train_digits}d}"
            else:
                correct_answer = str(a - b)

            model_answer = generateAnswer(prompt)
            length_counter[len(model_answer)] += 1

            if not has_borrow(a, b):
                total_no_borrow += 1
                if model_answer == correct_answer:
                    correct_no_borrow += 1
            else:
                if printed < 10:
                    print(f"{prompt}{model_answer} Correct Answer: {correct_answer}")
                    printed += 1
                total_borrow += 1
                borrow_predictions[model_answer] += 1 
                if model_answer == correct_answer:
                    correct_borrow += 1

# Print results safely (avoid divide-by-zero)
if total_no_borrow > 0:
    print(f"No Borrow Accuracy: {correct_no_borrow/total_no_borrow * 100}")
else:
    print("No Borrow Accuracy: N/A (no cases)")

if total_borrow > 0:
    print(f"Borrow Accuracy: {correct_borrow/total_borrow * 100}")
else:
    print("Borrow Accuracy: N/A (no cases)")

print(f"Borrow prediction distribution: {borrow_predictions}")
print(f"Output length distribution: {length_counter}")
print(model_answer)