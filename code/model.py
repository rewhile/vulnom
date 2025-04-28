# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from modelGNN_updates import *
from utils import preprocess_features, preprocess_adj
from utils import *
from torch.autograd import Variable
from operator import itemgetter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(args.hidden_size, args.num_classes)
        self.out_proj = nn.Linear(args.hidden_size, 1)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class GNNReGVD(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer = tokenizer

        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(feature_dim_size=args.feature_dim_size,
                                hidden_size=args.hidden_size,
                                num_GNN_layers=args.num_GNN_layers,
                                dropout=config.hidden_dropout_prob,
                                residual=not args.remove_residual,
                                att_op=args.att_op)
        else:
            self.gnn = ReGCN(feature_dim_size=args.feature_dim_size,
                               hidden_size=args.hidden_size,
                               num_GNN_layers=args.num_GNN_layers,
                               dropout=config.hidden_dropout_prob,
                               residual=not args.remove_residual,
                               att_op=args.att_op)
        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)
        # ★ node-level classifier for localisation
        self.node_cls  = nn.Linear(args.hidden_size, 1)

        # optional weighting between tasks
        self.lambda_node = getattr(args, "lambda_node", 0.2)


    def forward(
        self, adj=None, adj_mask=None, adj_feature=None,
        graph_labels=None, node_labels=None, node_mask=None
    ):
        adj = torch.squeeze(adj)  # [B,R,N,N]

        # --- GNN encoding ------------------------------------------------
        graph_embs = self.gnn(
            adj_feature.to(device).float(),
            adj.to(device).float(),
            adj_mask.to(device).float()
        )                                   # [B, hidden]
        graph_logits = self.classifier(graph_embs).squeeze(-1)   # [B]
        node_logits  = self.node_cls(self.gnn.last_node_vecs).squeeze(-1) # [B,N]

        if graph_labels is not None and node_labels is not None:
            # crit = torch.nn.BCEWithLogitsLoss(reduction='none')
            # use the same positive weight we computed for the sampler
            pos_w = getattr(self.args, "pos_weight", 1.0)
            crit  = torch.nn.BCEWithLogitsLoss(
                        reduction='none',
                        pos_weight=torch.tensor(pos_w, device=graph_logits.device)
                    )

            # ---- graph-level --------------------------------------------------
            g_loss = crit(graph_logits, graph_labels.float()).mean()

            # ---- node-level (ignore padded spots) ----------------------------
            n_loss_full = crit(node_logits, node_labels.float())          # [B,N]
            n_loss      = (n_loss_full * node_mask).sum() / node_mask.sum()

            loss = g_loss + self.lambda_node * n_loss
            return (
                loss,
                torch.sigmoid(graph_logits),   # returned probabilities
                torch.sigmoid(node_logits)
            )
