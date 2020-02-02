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

from transformers import RobertaForMaskedLM
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
        wandb.init(project='insertlm', name=args.exp_name, config=args)

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Prepare model
    # class tempmodel(nn.Module):
    #     def __init__(self, roberta, insert_net):
    #         super().__init__()
    #         self.roberta = roberta
    #         self.insert_net = insert_net

    roberta = RobertaForMaskedLM.from_pretrained("roberta-base")
    # insert_net = nn.Linear(768, 2)
    model = roberta
    num_train_optimization_steps = None
    if args.do_train:

        # train_dataset = MaskLMDataset(args.train_corpus, tokenizer, seq_len=512,
        #                             corpus_lines=None, on_memory=True)
        num_train_optimization_steps = int(
            (7000 / args.train_batch_size) / args.gradient_accumulation_steps) * args.num_train_epochs
            # (len(train_dataset) / args.train_batch_size) / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # if args.fp16:
    #     model.half()
    roberta.to(device)
    # insert_net.to(device)
    # model.speech_encoder.load_state_dict(torch.load("speech_model/speech_encoder.epoch.1.bin"))
    # if args.local_rank != -1:
    #     try:
    #         from apex.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #     model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)



    # Prepare optimizer
    # optimizer = torch.optim.SGD(model.speech_encoder.parameters(), lr=3e-5, momentum=0.9, weight_decay=5e-4)
    if args.do_train:
        param_optimizer = list(roberta.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # {'params': insert_net.parameters(), 'weight_decay': 0.01}
            ]
        t_total = ((7000 // args.train_batch_size) // args.gradient_accumulation_steps) * args.num_train_epochs
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
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", 7000)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        # if args.text:
            # if args.fasttext:
            #     def pad_phc_corpus(sents):
            #         lens = [len(x) for x in sents]
            #         max_len = max(lens)
            #         output = []
            #         for i in range(len(sents)):
            #             output.append([1] + [vocab_trans(x) for x in sents[i]] + [2] + [0] * (max_len - len(sents[i])))
            #         return torch.tensor(output), [(x+2) for x in lens]
            #         #     output.append(["[CLS]"] + [vocab_dict[x] for x in sents[i]] + ["[SEP]"])
                    # return output
                # plain_dataloader = DataLoader(plain_dataset, shuffle=False, batch_size=args.train_batch_size, collate_fn=lambda x: [(["[CLS]"] + i + ["[SEP]"]) for i in x])
                # label_dataloader = DataLoader(plain_dataset, shuffle=False, batch_size=args.train_batch_size, collate_fn=pad_phc_corpus)
                # plain_iter = iter(plain_dataloader)
            # else:
            #     plain_dataloader = DataLoader(plain_dataset, shuffle=False, batch_size=args.train_batch_size, collate_fn=pad_sents)
                # plain_iter = iter(plain_dataloader)
        roberta.train()
        # insert_net.train()
        scp = open(args.train_corpus+".train", 'r').readlines()
        # model.bert.eval()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        for epoch, idxfile in enumerate(scp):
            train_dataset = IndexedDataset(os.path.join(args.train_corpus, idxfile.strip()))
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_dataset)
            else:
                #TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=IndexedDataset.collater)
            print("EPOCH [%d/%d]"%(epoch, int(args.num_train_epochs)), flush=True)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            total_steps = len(train_dataloader)
            step = 0
            start = time.time()
            prev_acc = 0
            for step, batch in enumerate(train_dataloader):
                prev = time.time()
                batch = tuple(t.to(device) for t in batch)
                input_ids, predict_label, insert_label, input_mask = batch
                seq_output, _ = roberta.roberta(input_ids, attention_mask=input_mask)
                prediction_scores = roberta.lm_head(seq_output)
                # insert_score = insert_net(seq_output)
                
                loss = loss_fct(prediction_scores.view(-1, prediction_scores.shape[-1]), predict_label.view(-1))
                # insert_loss = loss_fct(insert_score.view(-1, 2), insert_label.view(-1))

                now = time.time()
                rem = ((now - start) * ((total_steps - step) / step)) if step != 0 else - 1
                now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                text = "{time} [{epoch}/{total_epoch}] [{idx}/{total_idx}] loss: {loss:.4f} ppl: {ppl:.4f} ETA: {eta:.1f}".format(epoch=epoch,
                total_epoch=int(args.num_train_epochs), idx=step, total_idx=total_steps, loss=float(loss), ppl=math.exp(float(loss)), eta=rem, time=now_time)
                if use_wandb:
                    wandb.log({'loss': float(loss), 'ppl': math.exp(float(loss))})
                # loss += insert_loss
                # else:
                #     now = time.time()
                #     rem = ((now - start) * ((total_steps - step) / step)) if step != 0 else - 1
                #     now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                #     text = "{time} [{epoch}/{total_epoch}] [{idx}/{total_idx}] LOSS: {loss:.4f} ETA: {eta:.1f}".format(epoch=epoch,
                #     total_epoch=int(args.num_train_epochs), idx=step, total_idx=total_steps, loss=float(loss), eta=rem, time=now_time)
                
                print(text, flush=True)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += batch[0].shape[0]
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(roberta.parameters(), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(insert_net.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    roberta.zero_grad()
                    # insert_net.zero_grad()
                    global_step += 1

                if (step + 1) % 10000 == 0:
                    # Save a trained model
                    logger.info("** ** * Saving fine - tuned model ** ** * \n")
                    model_to_save = roberta.module if hasattr(roberta, 'module') else roberta  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "bert_model.bin")
                    if args.do_train:
                        torch.save(roberta.state_dict(), output_model_file)
                    # output_model_file = os.path.join(args.output_dir, "insert_model.bin")
                    # if args.do_train:
                    #     torch.save(insert_net.state_dict(), output_model_file)
                step += 1

            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * \n")
            model_to_save = roberta.module if hasattr(roberta, 'module') else roberta  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "bert_model.bin")
            if args.do_train:
                torch.save(roberta.state_dict(), output_model_file)
            # output_model_file = os.path.join(args.output_dir, "insert_model.bin")
            # if args.do_train:
            #     torch.save(insert_net.state_dict(), output_model_file)



# if __name__ == "__main__":
#     main()
