from typing import TypeAlias
from collections import Counter, defaultdict

Word: TypeAlias = tuple[bytes, ...]
Pair: TypeAlias = tuple[bytes, bytes]


def get_pair_dict(
    words: dict[Word, int]
) -> tuple[dict[Pair, int], dict[Pair, set[Word]]]:
    
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

    for iter in range(max_iter):
        if iter % 1000 == 0:
            print(f"BPE at iteration {iter}/{max_iter}", flush=True)

        top_pair = max((pair_to_count[pair], pair) for pair in pair_to_count)[1]

        affected_words = list(pair_to_words[top_pair])

        for word in affected_words:
            count = words.pop(word)

            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_to_count[pair] -= count
                pair_to_words[pair].discard(word)

                if pair_to_count[pair] <= 0:
                    pair_to_count.pop(pair, None)

                if len(pair_to_words[pair]) == 0:
                    pair_to_words.pop(pair, None)
            
            new_word = merge_word(word, top_pair)
            words[new_word] = count

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i+1])
                pair_to_count[pair] += count
                pair_to_words[pair].add(new_word)
            
        pair_to_count.pop(top_pair, None)
        pair_to_words.pop(top_pair, None)

        merges.append(top_pair)
        vocabulary[len(vocabulary)] = top_pair[0] + top_pair[1]
    
    return vocabulary, merges

if __name__ == "__main__":
    pass