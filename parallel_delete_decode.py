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
from kermit_dataset import IndexedDataset
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



# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_corpus",
                        default="indexed_openwebtext",
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
    # parser.add_argument("--exp_name",
    #                     default="ssr",
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model checkpoints will be written.")

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
                        default=20000.0,
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
    parser.add_argument('--no_pos',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--epoch',
                        type=int,
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
    device = "cuda" #torch.device("cuda", args.local_rank)
    n_gpu = 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    roberta = RobertaForMaskedLM.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    insert_net = nn.Linear(768, 3)
    roberta.to(device)
    insert_net.to(device)
    roberta.load_state_dict(torch.load(os.path.join(args.output_dir, "bert_model.bin.%04d"%args.epoch)))
    insert_net.load_state_dict(torch.load(os.path.join(args.output_dir, "insert_model.bin.%04d"%args.epoch)))

    def decode(input_text, steps=10, temp=1.0):
        print("original:", input_text)
        for i in range(steps):
            input_text, tmp_text = decode_step(roberta, insert_net, tokenizer, input_text, temperature=temp)
            # print(str(i) + "=" * (7 - len(str(i))) + ":", tmp_text)
            print("step " + str(i) + " " * (3 - len(str(i))) + ":", input_text)
