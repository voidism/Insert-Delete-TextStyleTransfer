# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KERMIT finetuning runner."""

from __future__ import absolute_import, division, print_function, unicode_literals
from functools import lru_cache
import argparse
import logging
import os
import random
from io import open
import queue

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import RobertaForMaskedLM, RobertaTokenizer
# from grammar_dataset import IndexedDataset
from optimization_fp16 import AdamW, WarmupLinearSchedule
import pickle
import time, math

use_wandb = False
if 'WANDB' in os.environ:
    if os.environ['WANDB'] == '1':
        use_wandb = True
        import wandb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def decode_step(roberta_model, insert_model, tokenizer, input_text, temperature=0):
    raw_ids = tokenizer.encode(input_text.replace("<|endoftext|>", ''))
    input_ids = torch.tensor(raw_ids).unsqueeze(0).cuda()
    input_mask = torch.ones_like(input_ids).cuda()
    seq_output, _ = roberta.roberta(input_ids, attention_mask=input_mask)
    prediction_scores = roberta.lm_head(seq_output) / (temperature if temperature > 0 else 1.)
    insert_score = insert_net(seq_output)
    ins = insert_score.argmax(-1).squeeze(0).tolist()
    if temperature == 0:
        prd = prediction_scores.argmax(-1).squeeze().tolist()
    else:
        prd = torch.multinomial(torch.nn.functional.softmax(prediction_scores.squeeze(), dim=-1), num_samples=1).squeeze().tolist()
    ret = []
    all_tokens = []
    for i in range(len(ins)):
        if ins[i] == 2:
            print("deletion:", tokenizer.decode([raw_ids[i]]), "is deleted")
            continue
        if ins[i] == 1:
            ret.append(prd[i])
        ret.append(raw_ids[i])
        all_tokens += [prd[i], raw_ids[i]]
    return tokenizer.decode(ret), tokenizer.decode(all_tokens)

class IndexedDataset(Dataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""
    pad_idx = [1, 0]

    def __init__(self, path, max_len=256):
        self.max_len = max_len
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.tokens_list_neg, self.sizes_neg = self.read_data(path+".0.cor.idx", attr=1)
        self.tokens_list_pos, self.sizes_pos = self.read_data(path+".1.cor.idx", attr=0)
        self.size = len(self.tokens_list_neg)
        # assert self.size == len(self.tokens_list_pos)
        # self.aggregate_samples()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def set_to(self, attr=0):
        if attr == 0:
            self.tokens_list = self.tokens_list_neg
            self.sizes = self.sizes_neg
        elif attr == 1:
            self.tokens_list = self.tokens_list_pos
            self.sizes = self.sizes_pos
        else:
            raise ValueError("Invalid attribute `{}`".format(attr))

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


    def decode_step_batched(self, raw_ids, ins, prd, mask):
        ret_batch = []
        lengths = mask.sum(-1).int().tolist()
        lens = []
        for i, length in zip(range(len(raw_ids)), lengths):
            ret_batch.append(self.decode_step(raw_ids[i][:length].tolist(), ins[i][:length].tolist(), prd[i][:length].tolist()))
            lens.append(len(ret_batch[-1]))
        max_len = max(lens)
        mask = []
        for i in range(len(ret_batch)):
            ret_batch[i] = ret_batch[i] + [0] * (max_len - lens[i])
            mask.append([1]*lens[i] + [0] * (max_len - lens[i]))
        return torch.tensor(ret_batch), torch.tensor(mask)

    def decode_step(self, raw_ids, ins, prd):
        ret = []
        for i in range(len(ins)):
            if ins[i] == 2:
                #print("deletion:", self.tokenizer.decode([raw_ids[i]]), "is deleted")
                continue
                # pass
            if ins[i] == 1:
                #print("insertion:", self.tokenizer.decode([prd[i]]), "is inserted")
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
        
    
    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
    
    @lru_cache(maxsize=8)
    def get_sent_pair(self, i):
        self.check_index(i)
        return [0] + self.tokens_list_a[i][:self.max_len-2] + [2], [0] + self.tokens_list_b[i][:self.max_len-2] + [2]

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        sent = self.tokens_list[i][:self.max_len-1] + [2]
        # term = self.samples_list[i]
        # return torch.tensor(term[0]), torch.tensor(term[2]), torch.tensor(term[1]), torch.ones(len(term[0]))
        return torch.tensor(sent), torch.ones(len(sent)) #, [0] + self.tokens_list_b[i] + [2]

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


# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_corpus",
                        default="/workspace/datasets/fce/para/indexed/fce.train.gold.bea19.m2",
                        type=str,
                        required=False,
                        help="The input train corpus.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--from_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--exp_name",
                        default="ssr",
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=1000,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=100.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--epoch",
                        default=2,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--delete',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--no_pos',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--text',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fasttext',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--block',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Linear warmup over warmup_steps.")

    args = parser.parse_args()
    if use_wandb:
        wandb.init(project='grammar', name=args.exp_name, config=args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # Prepare model
    class tempmodel(nn.Module):
        def __init__(self, roberta, insert_net, delete_net):
            super().__init__()
            self.roberta = roberta
            self.insert_net = insert_net
            self.delete_net = delete_net

    roberta = RobertaForMaskedLM.from_pretrained("roberta-base")
    insert_net = nn.Linear(768, 3)
    delete_net = nn.Linear(768, 3)
    roberta.load_state_dict(torch.load(os.path.join(args.from_dir, 'bert_model.bin.%04d'%args.epoch)))
    insert_net.load_state_dict(torch.load(os.path.join(args.from_dir, 'insert_model.bin.%04d'%args.epoch)))
    delete_net.load_state_dict(torch.load(os.path.join(args.from_dir, 'delete_model.bin.%04d'%args.epoch)))
    model = tempmodel(roberta, insert_net, delete_net)
    num_train_optimization_steps = 100
    if args.do_train:

        # train_dataset = MaskLMDataset(args.train_corpus, tokenizer, seq_len=512,
        #                             corpus_lines=None, on_memory=True)
        train_dataset = IndexedDataset(args.train_corpus)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.train_batch_size, collate_fn=IndexedDataset.collater)

    # if args.fp16:
    #     model.half()
    roberta.to(device)
    insert_net.to(device)
    delete_net.to(device)
    # optimizer = torch.optim.SGD(model.speech_encoder.parameters(), lr=3e-5, momentum=0.9, weight_decay=5e-4)
    if args.do_train:
        param_optimizer = list(roberta.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': insert_net.parameters(), 'weight_decay': 0.01},
            {'params': delete_net.parameters(), 'weight_decay': 0.01}
            ]
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    
    roberta.eval()
    insert_net.eval()
    delete_net.eval()

    # scp = open(args.train_corpus+".train", 'r').readlines()
    # model.bert.eval()


    # for epoch in range(int(args.num_train_epochs)):
    #     print("EPOCH [%d/%d]"%(epoch, int(args.num_train_epochs)), flush=True)
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_steps = len(train_dataloader)
    step = 0
    start = time.time()
    prev_acc = 0
    epoch = 0
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    for attr in [0, 1]:
        fw = open(args.output_dir + ".%d" % attr, 'w')
        train_dataset.set_to(attr)
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.train_batch_size, collate_fn=IndexedDataset.collater)

        def write_file(tensor):
            for i in range(len(tensor)):
                fw.write(tokenizer.decode([x for x in tensor[i].tolist() if x not in [0, 1, 2, 50261, 50262]]) + '\n')
                
        for step, batch in enumerate(train_dataloader):
            prev = time.time()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask = batch
            with torch.no_grad():
                for iteration in range(1):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    seq_output, _ = roberta.roberta(input_ids, attention_mask=input_mask)
                    prediction_scores = roberta.lm_head(seq_output)
                    insert_score = delete_net(seq_output)
                    ins = insert_score.argmax(-1)
                    prd = prediction_scores.argmax(-1)
                    input_ids, input_mask = train_dataset.decode_step_batched(input_ids, ins, prd, input_mask)
                
                for iteration in range(5):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    seq_output, _ = roberta.roberta(input_ids, attention_mask=input_mask)
                    prediction_scores = roberta.lm_head(seq_output)
                    insert_score = insert_net(seq_output)
                    ins = insert_score.argmax(-1)
                    prd = prediction_scores.argmax(-1)
                    input_ids, input_mask = train_dataset.decode_step_batched(input_ids, ins, prd, input_mask)
                
                write_file(input_ids.cpu())

            now = time.time()
            rem = ((now - start) * ((total_steps - step) / step)) if step != 0 else - 1
            now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            text = "{time} [{epoch}/{total_epoch}] [{idx}/{total_idx}] ETA: {eta:.1f}".format(epoch=epoch,
            total_epoch=int(args.num_train_epochs), idx=step, total_idx=total_steps, eta=rem, time=now_time)
            # if use_wandb:
            #     wandb.log({'loss': float(loss), 'ppl': math.exp(float(loss)), 'insert': float(insert_loss)})
            # loss += insert_loss

            print(text, flush=True)
            step += 1
