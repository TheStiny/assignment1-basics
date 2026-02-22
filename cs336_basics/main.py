import regex as re
from collections import Counter, defaultdict
import time


#start = time.perf_counter()
#end = time.perf_counter()
#print(f"Reading took {end-start:.4f} seconds")

def get_pair_dict(words):
    pair_to_count = Counter()
    pair_to_words = defaultdict(set)

    for word, count in words.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_to_count[pair] += count
            pair_to_words[pair].add(word)
    
    return pair_to_count, pair_to_words

def merge_word(word, pair):
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

def bpe(words, vocabulary, max_iter):
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


def main_bpe(input_path, vocab_size, special_tokens):
    assert vocab_size > 256 + len(special_tokens)

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
    max_iter = vocab_size - 256 - len(special_tokens)
    
    vocabulary, merges = bpe(counter, vocabulary, max_iter)

    for t in special_tokens:
        vocabulary[len(vocabulary)] = t.encode("utf-8")

    return vocabulary, merges