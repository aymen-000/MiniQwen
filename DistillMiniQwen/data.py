from datasets import load_dataset  , load_from_disk
import os 
import torch 
from torch.utils.data import Dataset 
import torch.nn.functional as F

from tqdm import tqdm



def load_dataset(data_name = "nl2sh_alfa_dataset") : 
    ds = load_from_disk(data_name)
    


class TextDataset(Dataset):
    """Dataset for text sequences"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        print("Tokenizing data...")
        self.examples = []
        for text in tqdm(texts):
            tokens = tokenizer.encode(text.strip(), add_special_tokens=True)
            # Split into chunks of max_length
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length]
                if len(chunk) > 1:  # Need at least 2 tokens for input/target
                    self.examples.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Input is all but last token, target is all but first token
        return tokens[:-1], tokens[1:]


def collate_fn(batch):
    """Collate function with padding"""
    inputs, targets = zip(*batch)
    
    # Find max length in batch
    max_len = max(len(seq) for seq in inputs)
    
    # Pad sequences
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = F.pad(inp, (0, pad_len), value=0)
            tgt = F.pad(tgt, (0, pad_len), value=-100)  # -100 is ignore index
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)