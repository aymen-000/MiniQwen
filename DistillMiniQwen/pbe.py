"""
bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
sequences of integers
This implementation is based on openai's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""
""" 
I didn't use this in my code it is just for learning porpuse 
"""
import json
import os

import regex as re
import torch
from utils import get_file
# -----------------------------------------------------------------------------

def byte_to_unicode() : 
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0 
    for b in range(2**8) : 
        if b not in bs : 
            bs.append(b)
            cs.append(2**8+n)
            n += 1 
            
    cs = [chr(n) for n in cs]
    return dict(zip(bs , cs))
    
    
    
def get_paris(word) : 
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set() 
    prev_char = word[0] 
    for char in word[1:] : 
        pairs.add((prev_char , char))
        prev_char = char
        
    return pairs 

## print(get_paris("HELLO"))


class Encoder : 
    def __init__(self , encoder , bpe_merges , errors ='replace'):
        # byte encoder/decode
        self.encoder = encoder 
        self.decoder = {v:k for k , v in self.encoder.items()} 
        # bpe token encoder/decoder
        self.errors = errors 
        self.byte_encoder =   byte_to_unicode()
        self.byte_decoder = {v:k for k , v in self.byte_encoder.items()}
        
        self.bpe_ranks = dict(zip(bpe_merges ,range(len(bpe_merges)))) 
        
        # separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}
    def bpe(self , token) : 
        if token in self.cache : 
            return self.cache[token]
        
        word = tuple(token)
        # print(word)
        pairs = get_paris(word)
        

        if not pairs : 
            return token 
        
        while True : 
            bigram = min(pairs , key=lambda pair : self.bpe_ranks.get(pair , float("inf")))
            if bigram not in self.bpe_ranks : 
                break 
            
            first , second = bigram 
            
            new_word = [] 
            i = 0 
            while i < len(word) : 
                try: 
                    j = word.index(first , i)
                    new_word.extend(word[i:j])
                    i = j 
                    
                except : 
                    new_word.extend(word[i:])
                    break 
                
                if word[i] == first and i<len(word)-1 and word[i+1] == second : 
                    new_word.append(first+second)
                    
                    i += 2
                    
                else : 
                    new_word.append(word[i])
                    
                    i+= 1
                    
            new_word = tuple(new_word)
            # print("new word =====" , new_word)
            word = new_word
            
            if len(word) == 1 : 
                break 
            
            else : 
                pairs = get_paris(word)
                
        word = ' '.join(word)
        
        self.cache[token] = word 
        
        return word 
    
    
    def encode(self , text) : 
        bpe_tokens = []
        for token in re.findall(self.pat , text) : 
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8')) 
            bpe_tokens.extend(self.encoder[bpe_token]  for bpe_token in self.bpe(token).split(" ")  )
        return bpe_tokens
    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
            
        
        
        
def get_encoder():
    """
    Returns an instance of the GPT BPE Encoder/Decoder
    and handles caching of "database" files.
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    # load encoder.json that has the raw mappings from token -> bpe index
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == 50257 # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token

    # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure
    # in the form tuples (a, b), that indicate that (a, b) is to be merged to one token ab
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # light postprocessing: strip the version on first line and the last line is a blank
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000 # 50,000 merged tokens

    # construct the Encoder object and return
    enc = Encoder(encoder, bpe_merges)
    return enc

# -----------------------------------------------------------------------------

class BPETokenizer:
    """ PyTorch-aware class that wraps the Encoder above """

    def __init__(self):
        self.encoder = get_encoder()
        self.eot_char = "<|endoftext|>" # NOTE: Encoding this is broken!
        self.eot_token = 50256

    def __call__(self, text, return_tensors='pt'):
        # PyTorch only; here because we want to match huggingface/transformers interface
        assert return_tensors == 'pt'
        # single string input for now, in the future potentially a list of strings
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1
        idx = [self.encoder.encode(text)]
        # wrap into PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # decode indices to text
        text = self.encoder.decode(idx.tolist())
        return text
if __name__ == '__main__':

    tokenizer = BPETokenizer()
    text = "find . -type f -empty"    
    tokens = tokenizer(text)
    print("Tokens:", tokens)
    print("Decoded:", tokenizer.decode(tokens[0]))