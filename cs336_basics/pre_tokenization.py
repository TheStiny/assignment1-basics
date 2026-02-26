import os
from typing import BinaryIO
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import regex as re

from cs336_basics.utils import find_chunk_boundaries

def get_available_cpus():
    condor_cpus = os.environ.get("_CONDOR_NPROCS")
    if condor_cpus:
        return int(condor_cpus)
    else:
        return os.cpu_count() or 1

def process_chunk(args) -> Counter:
    
    file_path, start, end, special_tokens, Pat_str = args

    local_counter = Counter()
    split_pattern = re.compile("|".join([re.escape(t) for t in special_tokens]))
    word_pat = re.compile(Pat_str)
    byte_list = [bytes([i]) for i in range(256)]
    
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode("utf-8", errors="replace")

        text_chunks = split_pattern.split(chunk_text)
        
        for text_chunk in text_chunks:
            for match in word_pat.finditer(text_chunk):
                word = match.group()
                word_tuple = tuple(byte_list[b] for b in word.encode("utf-8"))
                local_counter[word_tuple] += 1
            
    return local_counter


def pre_tokenization(
    input_path: str, 
    special_tokens: list[str], 
    num_workers: int,
    Pat_str: str
) -> Counter:
    
    with open(input_path, "rb") as f:
        boundary_token = special_tokens[0].encode("utf-8") if special_tokens else b""
        boundaries = find_chunk_boundaries(f, num_workers, boundary_token)
    
    tasks = [(input_path, boundaries[i], boundaries[i+1], special_tokens, Pat_str) for i in range(len(boundaries) - 1)]

    total_counter = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for local_counter in executor.map(process_chunk, tasks):
            total_counter.update(local_counter)

    return total_counter

if __name__ == "__main__":
    num_cpus = get_available_cpus()
    print(f"There are {num_cpus} cpus")