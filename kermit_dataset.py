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
        



class IndexedDataset(Dataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""
    pad_idx = [1, -1, -1, 0]

    def __init__(self, path):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                self.lines.append(line)
                tokens = [int(x) for x in line.split(' ')]
                # tokens = torch.tensor(tokens).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)
        self.freq = torch.tensor([int(x.strip()) for x in open(os.path.split(path)[0] + ".freq", 'r').readlines()]).float()
    
    def rand_word(self):
        return int(torch.multinomial(self.freq, 1))

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
    def __getitem__(self, i):
        self.check_index(i)
        return self.insert_noise(sample(self.tokens_list[i]))

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
    d = IndexedDataset("indexed_openwebtext/urlsf_subset10-308_data.idx")
    sequence = d[0]
    sampled_sequence, predict_label, insert_label, input_mask = sequence
    from torch.utils.data import DataLoader
    dd = DataLoader(d, batch_size=4, collate_fn=IndexedDataset.collater)
    # slots, insert_label = find_slot(subindex, len(sequence))