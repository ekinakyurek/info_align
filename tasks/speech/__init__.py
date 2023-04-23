import os
import torch

DATA_PATH = f"{os.path.dirname(__file__)}/train_mined_t1.09_ekin.filter200.fast"

def format_y(y):
    return y

class SPEECHVocab:
    def __init__(self):
        vocab = {}
        for special_token in ["<pad>", "<skip1>", "<hole1>", "</s>"]:
            vocab[special_token] = len(vocab)
        with open(DATA_PATH) as reader:
            for line in reader:
                x, y = line.strip().split(" ||| ")
                for token in x.split() + format_y(y).split():
                    if token not in vocab:
                        vocab[token] = len(vocab)

        self.SKIP1 = vocab["<skip1>"]
        self.HOLE1 = vocab["<hole1>"]
        self.START = self.HOLE1 # vocab["<start>"]
        self.SEP = vocab["</s>"]
        self.END = vocab["</s>"]
        self.PAD = vocab["<pad>"]

        self.vocab = vocab
        self.rev_vocab = {v: k for k, v in vocab.items()}

    def encode(self, seq):
        return [self.vocab[c] for c in seq]

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy().tolist()
        seq = [s for s in seq if s != self.PAD]
        return " ".join(self.rev_vocab[c] for c in seq)

    def __len__(self):
        return len(self.vocab)

def load():
    vocab = SPEECHVocab()
    data = []
    with open(DATA_PATH) as reader:
        for line in reader:
            x, y = line.split(" ||| ")
            x = vocab.encode(x.split())
            y = vocab.encode(format_y(y).split())
            data.append((x, y))
    return data, vocab
