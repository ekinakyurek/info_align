PAD = "_"
START = "[START]"
SEP = "[SEP]"
END = "[END]"
HOLE1 = "[HOLE1]"
HOLE2 = "[HOLE2]"
SKIP1 = "[SKIP1]"
SKIP2 = "[SKIP2]"
UNK = "[UNK]"

INIT_VOCAB = {
    PAD: 0,
    START: 1,
    SEP: 2,
    END: 3,
    HOLE1: 4,
    HOLE2: 5,
    SKIP1: 6,
    SKIP2: 7,
    UNK: 8,
}

def decode(seq, vocab, rev_vocab):
    if vocab[END] in seq:
        seq = seq[:seq.index(vocab[END])+1]
    return " ".join(rev_vocab[s] for s in seq)


from typing import Iterator, Iterable, List, Union
import itertools
def batch(inputs: Union[List, Iterable], n: int):
    "Batch data into iterators of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError("n must be at least one")

    if not isinstance(inputs, Iterator):
        inputs = iter(inputs)

    while True:
        chunk_it = itertools.islice(inputs, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)