from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn
from transformers import MT5ForConditionalGeneration
from transformers import AutoTokenizer

import utils

N_HIDDEN = 512

# Estimates (conditional and unconditional) substring probabilities via counting.
# The `observe` functions increment the frequency of the corresponding event.
class CountModel:
    def __init__(self, vocab):
        self.counts = defaultdict(Counter)
        self.totals = Counter()

        self.src_counts = defaultdict(Counter)
        self.src_totals = Counter()
        self.tgt_counts = defaultdict(Counter)
        self.tgt_totals = Counter()
        self.vocab = vocab

    def observe(self, x, y):
        x = tuple(x)
        y = tuple(y)
        self.counts[x][y] += 1
        self.totals[x] += 1

    def observe_src(self, x, y, scale):
        x = tuple(x)
        y = tuple(y)
        self.src_counts[x][y] += 1. / scale
        self.src_totals[x] += 1. / scale

    def observe_tgt(self, x, y, scale):
        x = tuple(x)
        y = tuple(y)
        self.tgt_counts[x][y] += 1. / scale
        self.tgt_totals[x] += 1. / scale

    def h_src(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.src_counts[x][y]) - np.log(self.src_totals[x]))

    def h_tgt(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.tgt_counts[x][y]) - np.log(self.tgt_totals[x]))

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.counts[x][y]) - np.log(self.totals[x]))

# Estimates substring probabilities by fine-tuning a pre-trained model.
class PretrainedModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        self.vocab = vocab
        new_tokens = set(list(self.vocab.vocab.keys())) - set(self.tokenizer.vocab.keys())
        print("new tokens", new_tokens)
        self.tokenizer.add_tokens(list(new_tokens))

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inp, out, predict=False, teacher_forcing=False):
        # TODO double-check that this is necessary
        input_str  = [self.vocab.decode(x).replace("<pad>", self.tokenizer.pad_token)
                        for x in inp.numpy()]

        output_str = [self.vocab.decode(x).replace("<pad>", self.tokenizer.pad_token)
                        for x in out.numpy()]

        new_inp = self.tokenizer(input_str,
                             return_tensors="pt",
                             padding=True,
                             max_length=512,
                             truncation=True,
                             add_special_tokens=False)

        new_out = self.tokenizer(output_str,
                             return_tensors="pt",
                             max_length=512,
                             padding=True,
                             truncation=True,
                             add_special_tokens=False)


        labels = new_out.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        labels = labels.cuda()
        inputs = new_inp.input_ids.cuda()
        attention_mask = new_inp.attention_mask.cuda()


        if predict and not teacher_forcing:
            output_tokens = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=128,
                eos_token_id=self.vocab.END)

            return output_tokens


        output = self.model(input_ids=inputs,
                            attention_mask=attention_mask,
                            labels=labels)

        if predict and teacher_forcing:
            output_tokens = output.logits.argmax(dim=-1)
            return output.loss, output_tokens, labels

        return output.loss


# Estimates substring probabilities by training a transformer from scratch.
class Model(nn.Module):
    def __init__(self, vocab, max_length=1024):
        super().__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), N_HIDDEN)
        self.pos_embedding = nn.Embedding(max_length, N_HIDDEN)
        self.transformer = nn.Transformer(N_HIDDEN, batch_first=True)
        self.pred = nn.Linear(N_HIDDEN, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.PAD, reduction="none")
        self.max_length = max_length
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(max_length))


    def forward(self, inp, out):
        out_from = out[:, :-1]
        out_to = out[:, 1:]
        tgt_shape = out_to.shape

        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)
        out_pos = torch.arange(out_from.shape[1], device=out_from.device)[None, :]
        emb_out = self.embedding(out_from) + self.pos_embedding(out_pos)

        tgt_mask = self.mask[:emb_out.shape[1], :emb_out.shape[1]]
        src_key_padding_mask = inp == self.vocab.PAD
        memory_key_padding_mask = src_key_padding_mask

        enc = self.transformer(emb_inp,
                               emb_out,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        pred = self.pred(enc)

        pred = pred.reshape(-1, len(self.vocab))
        out_to = out_to.reshape(-1)

        loss = self.loss(pred, out_to).view(tgt_shape)
        loss = loss.sum(dim=1)
        return loss

    @torch.no_grad()
    def decode(self, inp, greedy=True, tgt = None, max_len=250):
        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)

        out = torch.tensor([[self.vocab.START]] * inp.shape[0]).cuda()

        if tgt is not None:
            max_len = tgt.shape[1]
            out = tgt[:, :1]
            gen_mask = tgt != self.vocab.PAD

        corrects = 0.0
        total = 0.0

        for i in range(max_len):
            out_pos = torch.arange(out.shape[1], device=out.device)[None, :]
            emb_out = self.embedding(out) + self.pos_embedding(out_pos)

            tgt_mask = self.mask[:emb_out.shape[1], :emb_out.shape[1]]
            src_key_padding_mask = (inp == self.vocab.PAD)
            memory_key_padding_mask = src_key_padding_mask

            enc = self.transformer(emb_inp,
                                   emb_out,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)
            pred = self.pred(enc)
            if tgt is not None:
                choice = pred[:, -1:].argmax(dim=2)
                mask = gen_mask[:, i:i+1]

                corrects += ((choice == tgt[:, i:i+1]) * mask).sum().item()
                total += mask.sum().item()
                choice = tgt[:, i:i+1]
            else:
                if greedy:
                    choice = pred[:, -1:].argmax(dim=2)
                else:
                    choice = torch.multinomial(torch.exp(pred[:, -1]), 1)

            out = torch.cat((out, choice), dim=1)

        print("Accuracy:", corrects / total)
        results = []
        for row in out:
            row = row.cpu().numpy().tolist()
            if self.vocab.END in row:
                row = row[:row.index(self.vocab.END)+1]
            results.append(row)
        return results
