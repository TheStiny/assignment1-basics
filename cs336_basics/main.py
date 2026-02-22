import regex as re
from collections import Counter
import time


#start = time.perf_counter()
#end = time.perf_counter()
#print(f"Reading took {end-start:.4f} seconds")

#with open("/home/fast/dokpekpe/Experiments/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
#    text = f.read()
#print(len(text))

#special_tokens = ["<|endoftext|>"]
#patter = "|".join([re.escape(t) for t in special_tokens])
#chunks = text.split(patter)

#PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#counter = Counter()

#for chunk in chunks:
#    words = [i.group() for i in re.finditer(PAT, chunk)]
#    temp_counter = Counter(tuple(bytes([b]) for b in word.encode("utf-8")) for word in words)
#    counter.update(temp_counter)

#text = '''low low low low low
#lower lower widest widest widest
#newest newest newest newest newest newest
#'''

#text = text.split()
#counter = Counter(tuple(bytes([b]) for b in word.encode("utf-8")) for word in text)
#vocabulary = {i: bytes([i]) for i in range(256)}

def get_top_merge(counter):
    new_counter = {}
    for string, count in counter.items():
        for i in range(len(string)-1):
            pair = (string[i], string[i+1])
            new_counter[pair] = new_counter.get(pair, 0) + count
    
    maxx = max(new_counter.values())
    top_merge = max({k:v for k,v in new_counter.items() if v == maxx})
    return top_merge

def do_merge(counter, top_merge):
    new_counter = {}
    pair1, pair2 = top_merge
    for string, count in counter.items():
        if pair1 in string and pair2 in string:
            current_string = []
            i = 0
            while i < len(string):
                if i < len(string)-1 and string[i] == pair1 and string[i+1] == pair2:
                    current_string.append(pair1+pair2)
                    i += 2
                else:
                    current_string.append(string[i])
                    i += 1
            new_counter[tuple(current_string)] = count
        else:
            new_counter[tuple(string)] = count
    return new_counter

def bpe(counter, vocabulary, max_iter):
    merges = []
    for _ in range(max_iter):
        top_merge = get_top_merge(counter)
        merges.append(top_merge)
        vocabulary[len(vocabulary)] = top_merge[0] + top_merge[1]
        counter = do_merge(counter, top_merge)
    return vocabulary, merges

def main_bpe(input_path, vocab_size, special_tokens):
    assert vocab_size > 256 + len(special_tokens)
    max_iter = vocab_size - (256 + len(special_tokens))
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    pattern = "|".join([re.escape(t) for t in special_tokens])
    chunks = re.split(pattern, text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    counter = Counter()

    for chunk in chunks:
        words = [i.group() for i in re.finditer(PAT, chunk)]
        temp_counter = Counter(tuple(bytes([b]) for b in word.encode("utf-8")) for word in words)
        counter.update(temp_counter)
    
    vocabulary = {i: bytes([i]) for i in range(256)}
    
    vocabulary, merges = bpe(counter, vocabulary, max_iter)

    for t in special_tokens:
        vocabulary[len(vocabulary)] = t.encode("utf-8")

    return vocabulary, merges
    
    
#path = "/home/fast/dokpekpe/Experiments/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
#special_tokens = ["<|endoftext|>"]
#v, m = main_bpe(path, 262, special_tokens)
#v[len(v)] = "<|endoftext|>".encode("utf-8")
#print(v)
#print(m)