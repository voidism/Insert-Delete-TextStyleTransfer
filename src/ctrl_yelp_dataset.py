# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset
import Levenshtein as lev

def edit_ops(a, b):
    c = ''.join([chr(x) for x in a])
    d = ''.join([chr(x) for x in b])
    ops = lev.editops(c, d)
    return ops


def find_slot(subindex, original_length):
    slots = []  # format: list of (position, start_index, end_index)
    insert_label = [0]
    for i in range(1, len(subindex)):
        if subindex[i] != (subindex[i - 1] + 1):
            slots.append((i, subindex[i - 1] + 1, subindex[i]))
            insert_label.append(1)
        else:
            insert_label.append(0)
    return slots, insert_label

def sample(sequence, pad_len=512, bos=0, eos=2, pad=1):
    assert len(sequence) <= (pad_len + 2)
    sequence = [bos] + sequence + [eos]
    z = list(range(1, len(sequence)-1))
    random.shuffle(z)
    length = random.randint(0, len(z) - 1)
    subindex = [0] + sorted(z[:length]) + [len(sequence) - 1]
    slots, insert_label = find_slot(subindex, len(sequence))
    sampled_sequence = torch.tensor(sequence)[subindex]
    predict_label = [-1] * len(subindex)
    for slot in slots:
        (position, start_index, end_index) = slot
        rand_position = random.randint(start_index, end_index-1)
        predict_label[position] = sequence[rand_position]
    return sampled_sequence, torch.tensor(predict_label), torch.tensor(insert_label), torch.ones_like(sampled_sequence) # , slots, sequence

def insert_mask(sampled_sequence, predict_label, insert_label, mask=50264):
    assert len(sampled_sequence) == len(predict_label)
    assert len(sampled_sequence) == len(insert_label)
    inserted_sequence = []
    inserted_predict_label = []
    for i in range(len(sampled_sequence)):
        if insert_label[i]:
            inserted_sequence.append(mask)
            inserted_predict_label.append(int(predict_label[i]))
        inserted_sequence.append(int(sampled_sequence[i]))
        inserted_predict_label.append(-1)
    inserted_sequence = torch.tensor(inserted_sequence)
    return inserted_sequence, torch.tensor(inserted_predict_label), insert_label, torch.ones_like(inserted_sequence)

    
from transformers import RobertaTokenizer
import tqdm

class IndexedDataset(Dataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""
    pad_idx = [1, -1, -1, 0]

    def __init__(self, path, max_len=256):
        self.max_len = max_len
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        tokens_list_neg_a, sizes_neg_a = self.read_data(path + ".0.flt.idx", attr=0)
        tokens_list_pos_a, sizes_pos_a = self.read_data(path + ".1.flt.idx", attr=1)
        self.tokens_list_a = tokens_list_neg_a + tokens_list_pos_a
        self.sizes_a = sizes_neg_a + sizes_pos_a
        tokens_list_neg_b, sizes_neg_b = self.read_data(path + ".0.cor.idx", attr=0)
        tokens_list_pos_b, sizes_pos_b = self.read_data(path + ".1.cor.idx", attr=1)
        self.tokens_list_b = tokens_list_neg_b + tokens_list_pos_b
        self.sizes_b = sizes_neg_b + sizes_pos_b
        self.size = len(self.tokens_list_a)
        assert self.size == len(self.tokens_list_b)
        self.aggregate_samples()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


    def read_data(self, path, attr, offset=50261):
        lines = []
        tokens_list = []
        sizes = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                lines.append(line)
                tokens = [offset + attr] + [int(x) for x in line.split(' ')] if len(line) > 0 else []
                # tokens = torch.tensor(tokens).long()
                tokens_list.append(tokens)
                sizes.append(len(tokens))
        sizes = np.array(self.sizes)
        return tokens_list, sizes

    def aggregate_samples(self):
        self.samples_list = []
        for i in tqdm.trange(len(self.tokens_list_a)):
            self.samples_list += self.generate_sample(i)
        self.size = len(self.samples_list)

        

    def generate_sample(self, idx):
        a, b = self.get_sent_pair(idx)[:]
        # ops = edit_ops(a, b)
        actions = []
        while True:
            ops = edit_ops(a, b)
            input_ids = a[:]
            predict_label = [-1] * len(a)
            insert_label = [0] * len(a)
            if len(ops) == 0:
                if len(actions) == 0:
                    actions.append((input_ids, insert_label, predict_label))
                break
            random.shuffle(ops)
            remain_ops = []
            for op in ops:
                if op[0] == 'insert':
                    if insert_label[op[1]] == 0 and predict_label[op[1]] == -1:
                        insert_label[op[1]] = 1
                        predict_label[op[1]] = b[op[2]]
                    else:
                        remain_ops.append(op)
                        continue
                elif op[0] == 'delete':
                    if insert_label[op[1]] != 1:
                        insert_label[op[1]] = 2
                    else:
                        remain_ops.append(op)
                        continue
                elif op[0] == 'replace':
                    if insert_label[op[1]] != 1 and insert_label[op[1]+1] != 2 and predict_label[op[1]+1] == -1:
                        insert_label[op[1]] = 2
                        insert_label[op[1]+1] = 1
                        predict_label[op[1]+1] = b[op[2]]
                    else:
                        remain_ops.append(op)
                        continue
            a = self.decode_step(a, insert_label, predict_label)
            actions.append((input_ids, insert_label, predict_label))
            if len(remain_ops) == 0:
                break
            # else:
                # print("Remaining ops:", remain_ops)
        return actions

    def decode_step(self, raw_ids, ins, prd):
        ret = []
        for i in range(len(ins)):
            if ins[i] == 2:
                # print("deletion:", self.tokenizer.decode([raw_ids[i]]), "is deleted")
                continue
            if ins[i] == 1:
                ret.append(prd[i])
            ret.append(raw_ids[i])
        return ret
        # return self.tokenizer.decode(ret), self.tokenizer.decode(all_tokens)

    def unittest(self, idx):
        actions = self.generate_sample(idx)
        a, b = self.get_sent_pair(idx)
        output = a[:]
        # print("Before all Action: {}".format(self.tokenizer.decode(a)))
        for i in range(len(actions)):
            assert output == actions[i][0] or print(output, actions[i][0])
            # print("Action {} (before): {}".format(i, self.tokenizer.decode(actions[i][0])))
            output = self.decode_step(*actions[i])
            # print("Action {} (after ): {}".format(i, self.tokenizer.decode(output)))
        # print("After all Actions: {}".format(self.tokenizer.decode(b)))
        assert b == output
        
    
    def insert_noise(self, data, max_length=512, noise_rate=0.05):
        sampled_sequence, predict_label, insert_label, _ = data
        length = len(predict_label)
        insert_coda = max_length - length
        inserted_sequence = []
        inserted_predict_label = []
        inserted_insert_label = []
        for i in range(len(sampled_sequence)):
            if random.random() < noise_rate and insert_coda > 0 and i > 0:
                inserted_sequence.append(self.rand_word())
                inserted_predict_label.append(-1)
                inserted_insert_label.append(2)
                insert_coda -= 1
            inserted_sequence.append(int(sampled_sequence[i]))
            inserted_predict_label.append(int(predict_label[i]))
            inserted_insert_label.append(int(insert_label[i]))
        return torch.tensor(inserted_sequence), torch.tensor(inserted_predict_label), torch.tensor(inserted_insert_label), torch.ones(len(inserted_sequence))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
    
    @lru_cache(maxsize=8)
    def get_sent_pair(self, i):
        self.check_index(i)
        return self.tokens_list_a[i][:self.max_len-1] + [2], self.tokens_list_b[i][:self.max_len-1] + [2]

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        term = self.samples_list[i]
        return torch.tensor(term[0]), torch.tensor(term[2]), torch.tensor(term[1]), torch.ones(len(term[0]))
        # return [0] + self.tokens_list_a[i] + [2], [0] + self.tokens_list_b[i] + [2]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def collater(sents_list):
        '''
        get the return object of `unpack_audiovec` and turn it into a single tensor with padding.
        seq_len = sents_lens, so we do not return sents_lens.
        '''
        n_items = len(sents_list[0])
        pad_idx = IndexedDataset.pad_idx
        if pad_idx is None:
            pad_idx = [0] * n_items
        sents_lens = [len(x[0]) for x in sents_list]
        max_sent_len = max(sents_lens)
        new_sents_list = []
        for i in range(len(sents_list)):
            new_sents_list.append([torch.cat((sents_list[i][j], (torch.zeros(max_sent_len - sents_lens[i], dtype=sents_list[i][j].dtype, device=sents_list[i][j].device) + pad_idx[j])), dim=0).unsqueeze(0) for j in range(n_items)])
        return [torch.cat([new_sents_list[j][i] for j in range(len(sents_lens))], dim=0) for i in range(n_items)]

if __name__ == '__main__':
    d = IndexedDataset("yelp/bpe/indexed/sentiment.test")
    sequence = d[0]
    #sampled_sequence, predict_label, insert_label, input_mask = sequence
    #from torch.utils.data import DataLoader
    #dd = DataLoader(d, batch_size=4, collate_fn=IndexedDataset.collater)
    # slots, insert_label = find_slot(subindex, len(sequence))
