import os
import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# 1) Load Ubuntu Dialogue QA dataset (REAL DATA)
# ============================================================

OUTPUT_DIR = "data"
MERGED_FILE = os.path.join(OUTPUT_DIR, "ubuntu_dialogue_qa.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nLoading dataset: sedthh/ubuntu_dialogue_qa")

ds = load_dataset(
    "sedthh/ubuntu_dialogue_qa",
    split="train",
    verification_mode="no_checks"
)


print(f"Using {len(ds)} samples")


# ============================================================
# 2) Convert to instruction format and save JSONL
# ============================================================

print("\nFormatting and saving dataset...")

with open(MERGED_FILE, "w", encoding="utf-8") as f:
    count = 0
    for row in tqdm(ds):

        question = row.get("INSTRUCTION", "").strip()
        answer = row.get("RESPONSE", "").strip()

        if not question or not answer:
            continue

        text = (
            "### Instruction:\n"
            f"{question}\n\n"
            "### Response:\n"
            f"{answer}"
        )

        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        count += 1

print(f"✔ Saved {count} samples to {MERGED_FILE}")


# ============================================================
# 3) TextDataset class (autoregressive LM)
# ============================================================

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"\nLoading merged dataset: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l)["text"] for l in f]

        print(f"Tokenizing {len(lines)} samples...")

        self.examples = []

        for text in tqdm(lines):
            if not text.strip():
                continue

            tokens = tokenizer.encode(text, add_special_tokens=True)

            # chunk into max_length pieces
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length]
                if len(chunk) > 2:
                    self.examples.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Final dataset token chunks: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return tokens[:-1], tokens[1:]


# ============================================================
# 4) Collate function (padding)
# ============================================================

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)

    padded_inp = []
    padded_tgt = []

    for x, y in zip(inputs, targets):
        pad_len = max_len - len(x)
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)
            y = F.pad(y, (0, pad_len), value=-100)

        padded_inp.append(x)
        padded_tgt.append(y)

    return torch.stack(padded_inp), torch.stack(padded_tgt)


# ============================================================
# 5) HOW TO USE
# ============================================================

"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", use_fast=True)

dataset = TextDataset(MERGED_FILE, tokenizer)
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for input_ids, labels in loader:
    print(input_ids.shape, labels.shape)
    break
"""
