import regex as re
from collections import Counter, defaultdict
import time
from typing import TypeAlias
import json
import resource
import os
from pathlib import Path


Word: TypeAlias = tuple[bytes, ...]
Pair: TypeAlias = tuple[bytes, bytes]


#start = time.perf_counter()
#end = time.perf_counter()
#print(f"Reading took {end-start:.4f} seconds")

def get_pair_dict(
    words: dict[Word, int]
) -> tuple[dict[Pair, int], dict[Pair, Word]]:
    pair_to_count = Counter()
    pair_to_words = defaultdict(set)

    for word, count in words.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_to_count[pair] += count
            pair_to_words[pair].add(word)
    
    return pair_to_count, pair_to_words

def merge_word(
    word: Word, 
    pair: Pair
) -> Word:
    new_word = []

    i = 0
    while i < len(word):
        if i < (len(word) - 1) and word[i] == pair[0] and word[i+1] == pair[1]:
            new_word.append(word[i] + word[i+1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
            
    return tuple(new_word)

def bpe(
    words: dict[Word, int], 
    vocabulary: dict[int, bytes], 
    max_iter: int
) -> tuple[dict[int, bytes], list[Pair]]:
    merges = []

    pair_to_count, pair_to_words = get_pair_dict(words)

    for _ in range(max_iter):
        top_pair = max(pair_to_count, key=lambda k: (pair_to_count[k], k))

        affected_words = list(pair_to_words[top_pair])

        for word in affected_words:
            count = words.pop(word)

            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_to_count[pair] -= count
                pair_to_words[pair].discard(word)
            
            new_word = merge_word(word, top_pair)
            words[new_word] = count

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i+1])
                pair_to_count[pair] += count
                pair_to_words[pair].add(new_word)
            
        del pair_to_count[top_pair]
        del pair_to_words[top_pair]
        merges.append(top_pair)
        vocabulary[len(vocabulary)] = top_pair[0] + top_pair[1]
    
    return vocabulary, merges


def main_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:
    assert vocab_size > 256 + len(special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    pattern = "|".join([re.escape(t) for t in special_tokens])
    chunks = re.split(pattern, text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    counter = Counter()

    print(f"There are {len(chunks)} chunks")

    byte_list = [bytes([i]) for i in range(256)]
    for i, chunk in enumerate(chunks):
        if i % 100_000 == 0:
            print(f"Processing chunk {i}/{len(chunks)}")
        words = (i.group() for i in re.finditer(PAT, chunk))
        temp_counter = Counter(tuple(byte_list[b] for b in word.encode("utf-8")) for word in words)
        counter.update(temp_counter)
    
    vocabulary = {i: bytes([i]) for i in range(256)}
    max_iter = vocab_size - 256 - len(special_tokens)
    
    vocabulary, merges = bpe(counter, vocabulary, max_iter)
    

    for t in special_tokens:
        vocabulary[len(vocabulary)] = t.encode("utf-8")


    return vocabulary, merges

script_dir = Path(__file__).parent.absolute()

input_path = script_dir.parent / "data/TinyStoriesV2-GPT4-train.txt"
#input_path = "/home/fast/dokpekpe/Experiments/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 10_000
special_tokens = ["<|endoftext|>"]

print("Start training")
start = time.perf_counter()

vocab, merges = main_bpe(input_path, vocab_size, special_tokens)

end = time.perf_counter()
print("End of training")

print(f"Training took: {(end-start)/60:.4f} minutes")

peak_memory_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
print(f"Peak memory usage: {peak_memory_gb} GB")

readable_vocab = {k: v.decode("latin-1") for k, v in vocab.items()}
readable_merges = [[merge[0].decode("latin-1"), merge[1].decode("latin-1")] for merge in merges]

output_dir = script_dir.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
vocab_dir = output_dir / "vocab.json"
merges_dir = output_dir / "merges.json"

with open(vocab_dir, "w", encoding="utf-8") as f:
    json.dump(readable_vocab, f, indent=4)

with open(merges_dir, "w", encoding="utf-8") as f:
    json.dump(readable_merges, f, indent=4)

longest_token = max(vocab.values(), key=len)
print(f"The longest token is: {longest_token.decode("utf-8", errors="replace")} with lenght {len(longest_token)}")