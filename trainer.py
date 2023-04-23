import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import pickle
import info
import utils
import itertools
from utils import batch as batcher


#N_EPOCH = 500
#N_ITER = 500
N_EPOCH = 500
N_ITER = 200

# picks an interval uniformly at random from [1, n]
def rand_interval(n, random):
    probs = np.arange(n, 0., -1.)
    probs /= probs.sum()
    start = random.choice(n, p=probs)
    end = random.randint(start, n+1)
    return start, end


# creates a random masked sequence
def make_example(src, tgt, random, vocab):
    if random.randint(2) == 0:
        return make_bi_example(src, tgt, random, vocab)
    else:
        if random.randint(2) == 0:
            return make_mono_example(src, random, vocab)
        else:
            return make_mono_example(tgt, random, vocab)

# creates a random masked bitext sequence
def make_bi_example(src, tgt, random, vocab, max_span_length=10):
    # possibilities = set()

    # second_operand_task = np.random.choice(["mask", "ignore", "predict"])
    # first_operand = np.random.choice(["source", "target"])

    # if first_operand ==  "source":
    #     src_mode = "predict"
    #     tgt_mode = second_operand_task
    # else:
    #     tgt_mode = "predict"
    #     src_mode = second_operand_task

    pred_src = np.random.randint(2)
    other_action = np.random.randint(3)
    if other_action == 0:
        other_mode = "mask"
    elif other_action == 1:
        other_mode = "ignore"
    elif other_action == 2:
        other_mode = "predict"
    if pred_src:
        src_mode = "predict"
        tgt_mode = other_mode
    else:
        tgt_mode = "predict"
        src_mode = other_mode

    if len(src) > 1:
        span_length = np.random.randint(1, min(len(src)+1, max_span_length))
        x0 = np.random.randint(len(src) - span_length)
        x1 = x0 + span_length
    else:
        x0, x1 = 0, 1

    if len(tgt) > 1:
        span_length = np.random.randint(1, min(len(tgt)+1, max_span_length))
        y0 = np.random.randint(len(tgt) - span_length)
        y1 = y0 + span_length
    else:
        y0, y1 = 0, 1

    if src_mode == "ignore":
        x0, x1 = 0, 0
    if tgt_mode == "ignore":
        y0, y1 = 0, 0

    return info.mask(src, tgt, x0, x1, y0, y1, src_mode, tgt_mode, vocab)


# creates a random masked (source- or target-only) sequence.
def make_mono_example(seq, random, vocab, max_span_length=10):
    # TODO(ekin): sample limited length spans
    if len(seq) > 1:
        span_length = np.random.randint(1, min(len(seq)+1, max_span_length))
        s = np.random.randint(len(seq) - span_length)
        e = s + span_length
    else:
        s, e = 0, 1
    #s, e = rand_interval(len(seq), random)
    p = random.randint(s, e+1)
    mode = random.choice(["left", "right", "both"])
    return info.mask_one(seq, s, p, e, mode, vocab)


# creates a batch of examples for training on
def make_batch(data, random, vocab):
    inps = []
    outs = []
    for i in range(len(data)):
        src, tgt = data[i]
        if len(src) == 0 or len(tgt) == 0:
            continue
        inp, out = make_example(src, tgt, random, vocab)
        inps.append(inp)
        outs.append(out)
    max_inp_len = max(len(i) for i in inps)
    max_out_len = max(len(o) for o in outs)
    for inp in inps:
        inp.extend([vocab.PAD] * (max_inp_len - len(inp)))
    for out in outs:
        out.extend([vocab.PAD] * (max_out_len - len(out)))
    return torch.tensor(inps), torch.tensor(outs)


def validate(model, vocab, val_data, n_batch=32):
    random = np.random.RandomState(0)
    total_items = 0.0
    val_loss = 0.0
    token_corrects = 0.0
    token_totals = 0.0

    for batched_data in batcher(val_data, n=n_batch):
        batched_data = list(batched_data)
        batch = make_batch(batched_data, random, vocab)
        loss, predictions, targets = model(*batch, predict=True, teacher_forcing=True)
        mask = (targets != -100)
        token_corrects += ((predictions == targets) * mask).sum().item()
        token_totals += mask.sum().item()
        val_loss += loss.item()
        total_items += 1

    val_loss = val_loss / total_items
    token_accuracy = token_corrects / token_totals
    return val_loss, token_accuracy

# trains a neural model
def train(model, vocab, data, save_path, random, params, validate_every: int = 500):
    random.shuffle(data)
    val_data = data[:500]
    data = data[500:]

    model.train()
    opt = optim.AdamW(model.parameters(), lr=params["lr"])
    iter = 0

    for i in tqdm(range(N_EPOCH)):
        print("Epoch: ", i)

        train_loss = 0.0
        train_items = 0.0
        random.shuffle(data)
        # print("len iter in epoch: ", len(data) // params["n_batch"])
        pbar = tqdm(batcher(data, n=params["n_batch"]), total=len(data) // params["n_batch"])

        for batched_data in pbar:
            batched_data = list(batched_data)
            batch = make_batch(batched_data, random, vocab)
            loss = model(*batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_items += 1
            pbar.set_description(f"loss:  {train_loss / train_items}")
            pbar.update()
            iter += 1
            if iter % validate_every == 0:
                print("Epoch ", i, " iter ", iter, " loss: ", train_loss / train_items)
                with torch.no_grad():
                    val_loss, val_token_accuracy = \
                             validate(model, vocab, val_data, n_batch=params["n_batch"])

                    print("validation loss: ", val_loss)
                    print("token accuracy: ", val_token_accuracy)

                    train_eval_loss, train_eval_token_accuracy = \
                          validate(model, vocab, data[:1024], n_batch=params["n_batch"])

                    print("sample train loss: ", train_eval_loss)
                    print("sample token accuracy: ", train_eval_token_accuracy)

                    batched_data = next(batcher(val_data, n=5))
                    batched_data = list(batched_data)
                    batch = make_batch(batched_data, random, vocab)
                    predictions = model(*batch, predict=True)
                    predictions = model.tokenizer.batch_decode(predictions,
                                                            skip_special_tokens=True)
                    inp, out = batch
                    for j in range(len(predictions)):
                        print("inputs: ", vocab.decode(inp[j]).strip())
                        print("outputs: ", vocab.decode(out[j]).strip())
                        print("predictions: ", predictions[j].replace("[/s]", "").strip())

                    torch.save(model.state_dict(), save_path)

        torch.save(model.state_dict(), save_path)




# trains a count-based model
def train_count(model, vocab, data, save_path):
    for src, tgt in tqdm(data):
        handled = set()
        for i in range(len(src)+1):
            for ii in range(i, len(src)+1):
                for j in range(len(tgt)+1):
                    for jj in range(j, len(tgt)+1):
                        for pred_src in range(2):
                            for other_action in range(3):
                                if other_action == 0:
                                    other_mode = "mask"
                                elif other_action == 1:
                                    other_mode = "ignore"
                                elif other_action == 2:
                                    other_mode = "predict"
                                if pred_src:
                                    src_mode = "predict"
                                    tgt_mode = other_mode
                                else:
                                    tgt_mode = "predict"
                                    src_mode = other_mode

                                # TODO clean up
                                for x0, x1 in [(i, ii)]:
                                    for y0, y1 in [(j, jj)]:
                                        if src_mode == "ignore":
                                            x0, x1 = 0, 0
                                        if tgt_mode == "ignore":
                                            y0, y1 = 0, 0
                                        sig = (x0, x1, y0, y1, src_mode, tgt_mode)
                                        if sig in handled:
                                            continue
                                        handled.add(sig)

                                        x, y = info.mask(src, tgt, x0, x1, y0, y1, src_mode, tgt_mode, vocab)
                                        model.observe(x, y)

        # TODO DOUBLE-CHECK THIS
        # I think we should only count the `both` case once.
        for i_s in range(len(src)+1):
            for i_e in range(i_s, len(src)+1):
                for i_p in range(i_s, i_e+1):
                    for mode in ["left", "right", "both"]:
                        x, y = info.mask_one(src, i_s, i_p, i_e, mode, vocab)
                        model.observe_src(x, y, i_e - i_s + 1)
        for j_s in range(len(tgt)+1):
            for j_e in range(j_s, len(tgt)+1):
                for j_p in range(j_s, j_e+1):
                    for mode in ["left", "right", "both"]:
                        x, y = info.mask_one(tgt, j_s, j_p, j_e, mode, vocab)
                        model.observe_tgt(x, y, j_e - j_s + 1)

    with open(save_path, "wb") as writer:
        pickle.dump(model, writer)


