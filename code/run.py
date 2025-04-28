# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import scipy.sparse as sp
import numpy as np
import torch
torch.cuda.empty_cache()
import csv
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
import json
import math
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import *
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

from typing import Optional
import string

logger = logging.getLogger(__name__)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Lexer(object):
    def __init__(self, code: str):
        self.code = code

    def _is_identifier_char(self, c: str) -> bool:
        if c in string.ascii_letters + string.digits + "_":
            return True
        return False

    def _is_digit(self, c: str) -> bool:
        if c in string.digits:
            return True
        return False

    def __iter__(self):
        self.i = 0
        return self

    def _peek(self) -> Optional[str]:
        if self.i < len(self.code):
            return self.code[self.i]
        raise StopIteration

    def _get(self) -> Optional[str]:
        if self.i < len(self.code):
            val = self.code[self.i]
            self.i += 1
            return val
        raise StopIteration

    def _identifier(self) -> str:
        token = self._get()
        while self._is_identifier_char(self._peek()):
            token += self._get()
        return token

    def _number(self) -> str:
        token = self._get()
        while self._is_digit(self._peek()):
            token += self._get()
        return token

    def __next__(self) -> str:
        while True:
            c = self._peek()
            if c is None:
                break
            if not c.isspace():
                break
            self._get()

        c = self._peek()
        if c in string.ascii_letters:
            return self._identifier()
        elif c in string.digits:
            return self._number()
        elif c in "()[]{}<>=+-*/#.,:;'\"|":
            return self._get()
        else:
            return self._get()

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 all_ids,
                 edges,
                 edges_label,
                 num_nodes,
                 idx,
                 label,
                 node_target,
                 code_lines

    ):
        self.all_ids = all_ids,
        self.edges = edges
        self.edges_label = edges_label
        self.num_nodes = num_nodes
        self.idx = str(idx)
        self.label = label
        self.node_target = node_target
        self.code_lines = code_lines


def convert_graphs_to_features(js,tokenizer, args):
    codes = js['nodes_codes']
    edges = js['edges']
    edges_label = []
    edges_label_source = js['edges_label']
    for i, label in enumerate(edges_label_source):
        if len(label) != 0:
            edges_label.append(0)
        elif len(label) == 0:
            node0_code = codes[edges[i][0]]
            node1_code = codes[edges[i][1]]
            if (node0_code not in node1_code) and (node1_code not in node0_code):
                edges_label.append(1)
            else:
                edges_label.append(2)

    codes = codes[:args.block_size]
    all_ids = []
    codes = codes[:args.block_size]
    for code in codes:
        code_tokens = tokenizer.tokenize(code)
        source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        all_ids.append(source_ids)
    num_nodes = len(codes)

    if num_nodes == 0:
        return None

    return InputFeatures(all_ids, edges, edges_label, num_nodes, js['idx'], js['target'], js['node_target'], js['code_lines'])

def convert_codes_to_tokens(js, args):
    codes = js['nodes_codes']
    codes = codes[:args.block_size]
    all_tokens = []
    codes = codes[:args.block_size]
    for code in codes:
        code_tokens = code.split(' ')
        all_tokens.append(code_tokens)
    return all_tokens

# build graph function
def build_graph(node_idxs, nodes_edges, num_nodes, edges_label, w_embeddings, args):
    # print('using window size = ', window_size)
    node_idxs = node_idxs[0]
    edges = nodes_edges + [[y, x] for x, y in nodes_edges]
    edges_label = edges_label + edges_label
    adj_list = []
    for i in range(3):
        row, col, weight = [], [], []
        for j, edge in enumerate(edges):
            if edge[0] == edge[1]:
                continue
            if edge[0] < num_nodes and edge[1] < num_nodes and edges_label[j] == i:
                row.append(edge[0])
                col.append(edge[1])
                weight.append(1)
                # weight.append(edges_label[i])
        adj = sp.csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
        adj_list.append(adj)


    features_out = []
    for node_idx in node_idxs:
        # emb_seq = []
        # for token in node_idx:
        #     emb_seq.append(w_embeddings[token])
        # emb_seq = torch.tensor(emb_seq)
        emb_np = np.stack([w_embeddings[token] for token in node_idx], axis=0)
        emb_seq = torch.from_numpy(emb_np).float()
        features_out.append(emb_seq)

    return adj_list, features_out

def build_adj(nodes_edges, num_nodes):
    # print('using window size = ', window_size)
    edges = nodes_edges + [[y, x] for x, y in nodes_edges]
    row, col, weight = [], [], []
    for edge in edges:
        if edge[0] == edge[1]:
            continue
        if edge[0] < num_nodes and edge[1] < num_nodes:
            row.append(edge[0])
            col.append(edge[1])
            weight.append(1)
    adj = sp.csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))

    return adj

def convert_examples_to_features(js,tokenizer,args):
    code=' '.join(js['func'].split())

    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['idx'],js['target'],js['node_target'])


class TextDataset(Dataset):
    def __init__(self, encoder, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        self.encoder = encoder.to(args.device)
        self.args = args
        self.w_embeddings = encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()

        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                result = convert_graphs_to_features(js, tokenizer, args)
                if result is None:
                    continue
                self.examples.append(result)
        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i):
    #     adj, x_feature = build_graph(self.examples[i].all_ids, self.examples[i].edges, self.examples[i].num_nodes, self.examples[i].edges_label, self.w_embeddings, self.args)
    #     adj, adj_mask = preprocess_adj(adj, self.args.block_size)
    #     adj_feature = preprocess_features(x_feature, self.args.block_size)

    #     label_ = self.examples[i].label

    #     return torch.tensor(adj), torch.tensor(adj_mask), torch.tensor(adj_feature), torch.tensor(label_)
    def __getitem__(self, i):
        adj_list, x_feature = build_graph(
            self.examples[i].all_ids,
            self.examples[i].edges,
            self.examples[i].num_nodes,
            self.examples[i].edges_label,
            self.w_embeddings,
            self.args
        )

        # --- graph-level pads (already fixed before) ---
        adj_padded_list, adj_mask = preprocess_adj(adj_list, self.args.block_size)
        adj_feature               = preprocess_features(x_feature, self.args.block_size)

        # --- NEW: pad node targets & create mask --------------------
        node_targets        = self.examples[i].node_target          # list/1-D array
        nt_len              = len(node_targets)
        pad_len             = self.args.block_size - nt_len
        if pad_len < 0:                       # truncate too-long graphs
            node_targets = node_targets[:self.args.block_size]
            nt_len       = self.args.block_size
            pad_len      = 0
        node_targets_padded = np.pad(node_targets, (0, pad_len), "constant")
        node_mask           = np.zeros(self.args.block_size, dtype=np.float32)
        node_mask[:nt_len]  = 1.0
        # -------------------------------------------------------------

        # convert to tensors once, all with matching shapes
        adj_tensor      = torch.from_numpy(np.stack(adj_padded_list, 0)).float()
        adj_mask_tensor = torch.from_numpy(adj_mask).float()
        feat_tensor     = torch.from_numpy(adj_feature).float()
        graph_lbl       = torch.tensor(self.examples[i].label, dtype=torch.float)
        node_lbl        = torch.tensor(node_targets_padded, dtype=torch.float)
        node_msk        = torch.tensor(node_mask,         dtype=torch.float)
        lines = self.examples[i].code_lines          # <-- grab them
        lines = lines[:self.args.block_size]
        pad   = self.args.block_size - len(lines)
        lines = np.pad(lines, (0, pad), "constant", constant_values=-1)

        line_tensor = torch.tensor(lines, dtype=torch.float)

        return (adj_tensor, adj_mask_tensor, feat_tensor,
                graph_lbl,  node_lbl,         node_msk,
                line_tensor)



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.local_rank == -1:                    # single-GPU / single-process
        # ---- build per-example weights ---------------------------------
        labels = [ex.label for ex in train_dataset.examples]   # 0 / 1
        n_pos  = sum(labels)
        n_neg  = len(labels) - n_pos
        # avoid div-by-zero when there are no positives (unlikely)
        pos_weight = (n_neg / n_pos) if n_pos else 1.0
        sample_weights = torch.tensor(
            [pos_weight if lbl == 1 else 1.0 for lbl in labels],
            dtype=torch.double
        )
        # store for later so we can reuse in the loss
        args.pos_weight = pos_weight

        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),   # usually == dataset size
            replacement = True
        )
    else:                                        # distributed case
        train_sampler = DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)  # numworkers=4
    args.max_steps=args.epoch*len(train_dataloader)
    # args.save_steps=len(train_dataloader)
    args.save_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.warmup_steps=len(train_dataloader)
    args.logging_steps=len(train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, loss_scale=1024)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_f1 = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    my_metrics = []
    # for idx in range(args.start_epoch, int(args.num_train_epochs)):
    for idx in trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch"):
        tr_num = 0
        train_loss = 0

        # for step, batch in enumerate(bar):
        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training (epoch {idx+1})", leave=False)):
        #     adj = batch[0].to(args.device)
        #     adj_mask = batch[1].to(args.device)
        #     adj_feature = batch[2].to(args.device)
        #     labels = batch[3].to(args.device)
        #     n_label = batch[4].to(args.device)
        #     model.train()
        #     loss, logits = model(adj, adj_mask, adj_feature, labels, n_label)
            # ignore code_lines during training
            adj, adj_mask, adj_feat, g_lbl, n_lbl, n_msk, _ = (t.to(args.device) for t in batch)
            model.train()
            loss, _, _ = model(adj, adj_mask, adj_feat, graph_labels=g_lbl, node_labels =n_lbl, node_mask   =n_msk)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, eval_dataset, model, tokenizer,eval_when_training=True)
                        my_metric = []
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))
                            my_metric.append(round(value,4))
                        my_metrics.append(my_metric)
                        # Save model checkpoint
                        
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
        avg_loss = round(train_loss / tr_num, 5)
        logger.info("epoch {} loss {}".format(idx, avg_loss))


def p_r_f1(labels, preds):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(labels)):
        if labels[i] == 1 and preds[i] == 1:
            TP += 1
        if labels[i] == 0 and preds[i] == 1:
            FP += 1
        if labels[i] == 1 and preds[i] == 0:
            FN += 1
        if labels[i] == 0 and preds[i] == 0:
            TN += 1
    precision = (TP+1) / (TP+FP+1)
    recall = (TP+1) / (TP+FN+1)
    f1 = 2 / (1/precision + 1/recall)
    FPR = (FP + 1) / (TN + FP + 1)
    FNR = (FN + 1) / (TP + FN + 1)
    return precision, recall, f1, FPR, FNR

def evaluate(args, eval_dataset, model, tokenizer, eval_when_training=False):
    """
    Run evaluation on dev set.
    Works for both single- and multi-GPU, handles
    the new (adj, mask, feat, graph_lbl, node_lbl) tuples.
    """
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    sampler_cls = SequentialSampler if args.local_rank == -1 else DistributedSampler
    eval_sampler    = sampler_cls(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler      = eval_sampler,
        batch_size   = args.eval_batch_size,
        num_workers  = 4,
        pin_memory   = True,
    )

    if args.n_gpu > 1 and not eval_when_training:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size   = %d", args.eval_batch_size)

    model.eval()
    eval_loss, nb_steps = 0.0, 0
    all_logits, all_labels = [], []

    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        # adj         = batch[0].to(args.device)
        # adj_mask    = batch[1].to(args.device)
        # adj_feat    = batch[2].to(args.device)
        # g_labels    = batch[3].to(args.device)
        # n_labels    = batch[4].to(args.device)

        # with torch.no_grad():
        #     loss, g_logits, _ = model(
        #         adj, adj_mask, adj_feat,
        #         graph_labels=g_labels,
        #         node_labels =n_labels
        #     )
        #     eval_loss += loss.item()
        #     nb_steps  += 1
        #     all_logits.append(g_logits.cpu().numpy())
        #     all_labels.append(g_labels.cpu().numpy())
        adj, adj_mask, adj_feat, g_lbl, n_lbl, n_msk, line_ids = (t.to(args.device) for t in batch)
        with torch.no_grad():
            loss, g_logits, n_logits = model(adj, adj_mask, adj_feat,
                                    graph_labels=g_lbl,
                                    node_labels =n_lbl,
                                    node_mask   =n_msk)
        eval_loss += loss.item()
        nb_steps  += 1
        all_logits.append(g_logits.cpu().numpy())
        all_labels.append(g_lbl.cpu().numpy())

    eval_loss /= nb_steps
    logits  = np.concatenate(all_logits, 0)
    labels  = np.concatenate(all_labels, 0)
    preds   = logits > 0.5
    # nothing was processed – just return zeros to avoid /0
    if nb_steps == 0:
        return {k: 0 for k in [
            "eval_loss","eval_acc","eval_precision","eval_recall",
            "eval_f1","eval_auc","eval_FPR","eval_FNR"]}

    eval_loss /= nb_steps

    precision, recall, f1, fpr, fnr = p_r_f1(labels, preds)
    auc = roc_auc_score(labels, preds)

    return {
        "eval_loss":     round(float(eval_loss), 6),
        "eval_acc":      round(float(np.mean(labels == preds)), 4),
        "eval_precision":round(precision, 4),
        "eval_recall":   round(recall, 4),
        "eval_f1":       round(f1, 4),
        "eval_auc":      round(auc, 4),
        "eval_FPR":      round(fpr, 4),
        "eval_FNR":      round(fnr, 4),
    }

def test(args, test_dataset, model, tokenizer):
    """
    Run test on held-out set and write predictions.txt
    Works with the 5-tensor batches.
    """
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    sampler_cls = SequentialSampler if args.local_rank == -1 else DistributedSampler
    test_sampler = sampler_cls(test_dataset)
    test_loader  = DataLoader(
        test_dataset,
        sampler     = test_sampler,
        batch_size  = args.eval_batch_size,
        num_workers = 4,
        pin_memory  = True,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size   = %d", args.eval_batch_size)

    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(test_loader, desc="Testing", leave=False):
        # adj       = batch[0].to(args.device)
        # adj_mask  = batch[1].to(args.device)
        # adj_feat  = batch[2].to(args.device)

        # with torch.no_grad():
        #     g_logits, _ = model(adj, adj_mask, adj_feat)  # forward **without** labels
        adj, adj_mask, adj_feat, *_ = (t.to(args.device) for t in batch)
        with torch.no_grad():
            g_logits, _ = model(adj, adj_mask, adj_feat)

        all_logits.append(g_logits.cpu().numpy())
        all_labels.append(batch[3].numpy())  # graph labels

    logits = np.concatenate(all_logits, 0)
    labels = np.concatenate(all_labels, 0)
    preds  = (logits > 0.5)

    precision, recall, f1, fpr, fnr = p_r_f1(labels, preds)
    auc = roc_auc_score(labels, preds)

    # --------------------------------------------------
    #  write raw predictions for inspection / leaderboard
    # --------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as fh:
        for ex, p in zip(test_dataset.examples, preds):
            fh.write(f"{ex.idx}\t{int(p)}\n")

    return {
        "test_acc":       round(float(np.mean(labels == preds)), 4),
        "test_precision": round(precision, 4),
        "test_recall":    round(recall, 4),
        "test_f1":        round(f1, 4),
        "test_auc":       round(auc, 4),
        "test_FPR":       round(fpr, 4),
        "test_FNR":       round(fnr, 4),
    }
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="../dataset/my_train.jsonl", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default="../dataset/my_valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../dataset/my_test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


    parser.add_argument("--model", default="GNNs", type=str,help="")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="hidden size.")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percet of training sample")
    parser.add_argument("--alpha_weight", default=1., type=float, help="percet of training sample")

    input_argument = [
        "--output_dir", "./saved_models",
        "--model_type", "roberta",
        "--tokenizer_name", "microsoft/graphcodebert-base",  
        "--model_name_or_path", "microsoft/graphcodebert-base",
        "--do_eval",
        "--do_test",
        "--do_train",
        # "--train_data_file", "../dataset/NVD/my_train.jsonl",
        # "--eval_data_file", "../dataset/NVD/my_valid.jsonl",
        # "--test_data_file", "../dataset/NVD/my_test.jsonl",
        "--train_data_file", "../dataset/GITA/openssl/my_train.jsonl",
        "--eval_data_file", "../dataset/GITA/openssl/my_valid.jsonl",
        "--test_data_file", "../dataset/GITA/openssl/my_test.jsonl",
        # "--block_size", "400",
        # "--block_size", "256",
        # "--block_size", "128",
        "--block_size", "64",
        "--fp16",
        # "--train_batch_size", "32",
        "--train_batch_size", "8",
        # "--eval_batch_size", "32",
        "--eval_batch_size", "8",
        "--max_grad_norm", "1.0",
        "--evaluate_during_training",
        "--gnn", "ReGCN",
        "--learning_rate", "5e-4",
        # "--epoch", "200",
        "--epoch", "10",
        "--hidden_size", "128",
        "--num_GNN_layers", "2",
        "--seed", "123456",
    ]

    args = parser.parse_args(args=input_argument)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//max(args.n_gpu,1)
    args.per_gpu_eval_batch_size=args.eval_batch_size//max(args.n_gpu,1)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        pretrained_model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        pretrained_model = model_class(config)

    model = GNNReGVD(pretrained_model, config, tokenizer, args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = TextDataset(pretrained_model, tokenizer, args, args.train_data_file, args.training_percent)
    eval_dataset = TextDataset(pretrained_model, tokenizer, args, args.eval_data_file)
    test_dataset = TextDataset(pretrained_model, tokenizer, args, args.test_data_file)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()
        train(args, train_dataset, eval_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            result=evaluate(args, eval_dataset, model, tokenizer)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            test_result = test(args, test_dataset, model, tokenizer)

            logger.info("***** Test results *****")
            for key in sorted(test_result.keys()):
                logger.info("  %s = %s", key, str(round(test_result[key],4)))

    return results


if __name__ == "__main__":
    main()


