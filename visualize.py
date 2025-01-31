from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import info
from trainer import make_batch
from utils import batch as batcher
import pickle
from tqdm import tqdm


@torch.no_grad()
def visualize(model, vocab, data, vis_path):
    model.eval()
    counts = Counter()

    vis_data = []

    for i, (src, tgt) in tqdm(enumerate(data[:10])):
        if src == tgt:
            continue
        src_toks = tuple(vocab.decode(src).split())
        tgt_toks = tuple(vocab.decode(tgt).split())

        # print("=============")
        #print("i: ", i, "src: ", vocab.decode(src), "tgt: ",vocab.decode(tgt))

        # for (s0, s1), (t0, t1), score in info.parse_greedy(src, tgt, model, vocab):
        #     print(src_toks[s0:s1], tgt_toks[t0:t1], score)
        #     counts[src_toks[s0:s1], tgt_toks[t0:t1]] += score
        parse = info.parse_greedy(src, model, vocab)
        vis_data.append({"source": src_toks, "parse": parse})

    with open("vis_data.pkl", "wb") as f:
        pickle.dump(vis_data, f)
        # for (s0, s1),  score in info.parse_greedy(src, model, vocab):
        #     print(src_toks[s0:s1], score)
        #     counts[src_toks[s0:s1]] += score


        # with open(f"{vis_path}/counts.html", "w") as count_writer:
        #     print("<html><head><meta charset='utf-8'></head><body><table>", file=count_writer)
        #     for ((k, v), c) in sorted(counts.most_common(1000), key=lambda x: -x[1]):
        #         print("<tr><td>", k, "</td><td>", v, "</td><td>", c, "</tr>", file=count_writer)
        #     print("</table></body><html>", file=count_writer)
        #     count_writer.flush()
