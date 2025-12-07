import os
import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# 1) Load dataset mixture config
# ============================================================

CONFIG_PATH = "./DistillMiniQwen/general_instruction_mixture.json"  
OUTPUT_DIR = "data"
MERGED_FILE = os.path.join(OUTPUT_DIR, "general_instruction_merged.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    mix_cfg = json.load(f)

datasets_cfg = mix_cfg["datasets"]

TOTAL_SAMPLES = 24000   # final dataset size — small & perfect for KD

# assign samples by weight
for ds in datasets_cfg:
    ds["samples"] = int(TOTAL_SAMPLES * ds["weight"])


print("\n========= DOWNLOAD PLAN =========")
for ds in datasets_cfg:
    print(f"{ds['name']} : {ds['samples']} samples")
print("=================================\n")


# ============================================================
# 2) Download + merge all datasets in one JSONL file
# ============================================================

def load_first_n(dataset_path, n):
    """
    Loads the first N rows of an HF dataset safely.
    """
    try:
        # Try streaming first for large datasets
        ds = load_dataset(dataset_path, split="train", streaming=True)
        samples = []
        for i, sample in enumerate(ds):
            if i >= n:
                break
            samples.append(sample)
        return samples
    except:
        try:
            # Fall back to direct split indexing
            return load_dataset(dataset_path, split=f"train[:{n}]")
        except:
            # Last resort: load full dataset and select
            ds = load_dataset(dataset_path, split="train")
            return ds.select(range(min(n, len(ds))))


print("Starting download and merge...")

with open(MERGED_FILE, "w", encoding="utf-8") as f_out:

    for ds in datasets_cfg:
        name = ds["name"]
        path = ds["path"]
        n = ds["samples"]

        print(f"\nDownloading {name} ({path}) with {n} samples...")

        try:
            split = load_first_n(path, n)

            count = 0
            for row in split:
                # Handle different dataset formats
                if "text" in row:
                    text = row["text"]
                elif "instruction" in row and "response" in row:
                    text = f"{row['instruction']}\n{row['response']}"
                elif "instruction" in row and "output" in row:
                    text = f"{row['instruction']}\n{row['output']}"
                elif "prompt" in row and "completion" in row:
                    text = f"{row['prompt']}\n{row['completion']}"
                elif "question" in row and "answer" in row:
                    text = f"{row['question']}\n{row['answer']}"
                elif "conversations" in row:
                    # Handle ShareGPT format
                    convs = row["conversations"]
                    text = "\n".join([f"{c.get('from', '')}: {c.get('value', '')}" for c in convs])
                else:
                    text = " ".join([str(v) for v in row.values()])

                if text.strip():
                    f_out.write(json.dumps({"text": text.strip()}) + "\n")
                    count += 1

            print(f"✔ Saved {count} samples from {name}")
            
        except Exception as e:
            print(f"⚠ ERROR loading {name}: {e}")
            continue

print("\nDONE — merged dataset saved at:", MERGED_FILE)


# ============================================================
# 3) TextDataset class for autoregressive LM
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

            tokens = tokenizer.encode(text.strip(), add_special_tokens=True)

            # chunk into max_length pieces
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i+max_length]
                if len(chunk) > 2:
                    self.examples.append(torch.tensor(chunk))

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

# Example (uncomment after you load your tokenizer):
#
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
#
# dataset = TextDataset(MERGED_FILE, tokenizer)
#
# loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
#
# for batch in loader:
#     input_ids, labels = batch
#     print(input_ids.shape, labels.shape)
#     break