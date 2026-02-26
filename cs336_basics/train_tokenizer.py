import regex as re
import time
import json
import resource
from pathlib import Path
import sys
from typing import TypeAlias

from cs336_basics.pre_tokenization import pre_tokenization, get_available_cpus
from cs336_basics.BPE import bpe

Word: TypeAlias = tuple[bytes, ...]
Pair: TypeAlias = tuple[bytes, bytes]

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:

    assert vocab_size > (256 + len(special_tokens))

    num_workers = get_available_cpus()
    print(f"Currently using {num_workers} cpus")

    start = time.perf_counter()
    print("Start pre-trokenization")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counter = pre_tokenization(input_path, special_tokens, num_workers, PAT)

    end = time.perf_counter()   
    print("End pre-tokenization")
    print(f"Pre-tokenization took: {(end-start)/60:.4f} minutes")

    start = time.perf_counter()
    print("Start BPE training")

    vocabulary = {i: bytes([i]) for i in range(256)}
    max_iter = vocab_size - 256 - len(special_tokens) 
    vocabulary, merges = bpe(counter, vocabulary, max_iter)
    
    for t in special_tokens:
        vocabulary[len(vocabulary)] = t.encode("utf-8")

    end = time.perf_counter()
    print("End of BPE training")
    print(f"BPE training took: {(end-start)/60:.4f} minutes")

    return vocabulary, merges

if __name__ == "__main__":

    script_dir = Path(__file__).parent.absolute()
    input_path = script_dir.parent / "data/TinyStoriesV2-GPT4-train.txt" 
    #input_path = script_dir.parent / "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10_000
    #input_path = script_dir.parent / "data/owt_train.txt"
    #input_path = script_dir.parent / "data/owt_valid.txt"
    #vocab_size = 32_000

    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    peak_memory_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"Peak memory usage: {peak_memory_gb} GB")

    readable_vocab = {k: v.decode("latin-1") for k, v in vocab.items()}
    readable_merges = [[merge[0].decode("latin-1"), merge[1].decode("latin-1")] for merge in merges]

    output_dir = script_dir.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir = output_dir / "vocabTS.json"
    merges_dir = output_dir / "mergesTS.json"

    with open(vocab_dir, "w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, indent=4)

    with open(merges_dir, "w", encoding="utf-8") as f:
        json.dump(readable_merges, f, indent=4)

    longest_token = max(vocab.values(), key=len)
    print(f"The longest token is: {longest_token.decode("utf-8", errors="replace")} with lenght {len(longest_token)}")