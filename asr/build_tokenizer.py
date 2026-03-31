# build_tokenizer.py
import sys
import os
from speechbrain.dataio.encoder import CTCTextEncoder

root = "C:/Projects/bitblind/asr"
out_dir = os.path.join(root, "trainings", "wav2vec2-base-960h")
os.makedirs(out_dir, exist_ok=True)

tokenizer = CTCTextEncoder()

# Add each label individually using add_label
# Order matters - blank=0, bos=1, eos=2 to match model config
labels = [
    "<blank>",  # index 0
    "<bos>",    # index 1
    "<eos>",    # index 2
    " ",        # index 3
    "'",        # index 4
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

for label in labels:
    tokenizer.add_label(label)

out_path = os.path.join(out_dir, "tokenizer.ckpt")
tokenizer.save(out_path)
print(f"Saved tokenizer with {len(tokenizer)} tokens to {out_path}")
print("Labels:", tokenizer.lab2ind)