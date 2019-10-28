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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from data_processors import ColaProcessor,MnliProcessor,MnliMismatchedProcessor,MrpcProcessor,Sst2Processor,StsbProcessor,QqpProcessor,QnliProcessor,RteProcessor,WnliProcessor
from train_utils import compute_metrics, convert_examples_to_features, id2onehot, InputFeatures

from lime.lime_text import LimeTextExplainer
from functools import partial

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def span_generator(plain_text, token, where2start):
    text_pointer = where2start
    token_pointer = 0
    while token_pointer < len(token):
        if token[token_pointer] == plain_text[text_pointer]:
            token_pointer += 1
            text_pointer += 1
        
        else:
            print("wrong char appear!")
            print(plain_text)
            print(token, "|", token[token_pointer],"|", plain_text[text_pointer], "|", plain_text[text_pointer - 1])
            if token[token_pointer] == "/":
                token_pointer +=1
            elif plain_text[text_pointer].isspace() and not token[token_pointer].isspace(): 
                print("plain_text emp")
                text_pointer += 1
            elif token[token_pointer].isspace() and not plain_text[text_pointer].isspace():
                print("token emp")
                token_pointer += 1
            elif token == "(":
                return (where2start-3, where2start-3), where2start-2
            else:
                exit()


        
            
    return (where2start, text_pointer-1), text_pointer

def span_generator_4unk(plain_text, where2start, white_split):
    text_pointer = where2start
    wstk_pointer = 0
    ws_char_idx = 0
    while ws_char_idx < where2start:
        ws_char_idx += len(white_split[wstk_pointer])
        wstk_pointer += 1

    if ws_char_idx != where2start:
        if ws_char_idx > where2start:
            unk_end = ws_char_idx
    else:
        unk_end = where2start + len(white_split[wstk_pointer])
    return (where2start, unk_end - 1), unk_end




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
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
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # =================model defination===================
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        if torch.cuda.is_available() and not args.no_cuda:
           device = torch.device("cuda")
        elif args.no_cuda or torch.cuda.is_available():
            device = torch.device("cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None        
    # Prepare model
    
    #==================End of Model Def=================

    eval_examples = processor.get_test_examples(args.data_dir)

    ws_spans = []
    bert_spans = []
    for inst_i, inst in enumerate(eval_examples):
        tokens = tokenizer.tokenize(inst.text_a)
        white_split = inst.text_a.split()
        plain_text = "".join(inst.text_a.split())

        index_b = 0
        
        bert_char_idx = 0
        ws_span = []
        bert_span = []
        bert_nosharp_tokens = []
        for tk in tokens:
            if "##" in tk:
                bert_nosharp_tokens.append(tk[2:])
            else:
                bert_nosharp_tokens.append(tk)

        while index_b < len(bert_nosharp_tokens):
            
            if bert_nosharp_tokens[index_b] == "[UNK]": 
                single_bert_span, bert_char_idx = span_generator_4unk(plain_text, bert_char_idx, white_split)
            else:
                single_bert_span, bert_char_idx = span_generator(plain_text, bert_nosharp_tokens[index_b], bert_char_idx)
            
            bert_span.append(single_bert_span)
            index_b += 1

        index_ws = 0
        ws_char_idx = 0

        while index_ws < len(white_split):
            single_ws_span, ws_char_idx = span_generator(plain_text, white_split[index_ws], ws_char_idx)
            ws_span.append(single_ws_span)
            index_ws += 1

        
        ws_spans.append(ws_span)
        bert_spans.append(bert_span)
    print("data: ", args.data_dir)
    print("ws_spans", len(ws_spans))
    print("bert_spans: ", len(bert_spans))
    
    with open(os.path.join(args.data_dir, "bert-spans.npz"), "wb") as f:
        np.save(f, bert_spans)
    with open(os.path.join(args.data_dir, "ws-spans.npz"), "wb") as g:
        np.save(g, ws_spans)

    print("alignment matrix created. find in: ", args.data_dir, "/bert-spans.npz")
if __name__ == "__main__":
    main()
