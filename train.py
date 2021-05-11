import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RAGA
from data import MyData
from loss import MyLoss
from utils import add_inverse_rels, get_train_batch, get_hits

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", choices=["WN18RR", "FB15k-237"])
    parser.add_argument("--e_hidden", type=int, default=300)
    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--test_epoch", type=int, default=5)
    args = parser.parse_args()
    return args

def init_data(args, device):
    data = MyData(args.data, e_hidden=args.e_hidden)[0]
    data.x = F.normalize(data.x, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all, data.rel_all = add_inverse_rels(data.edge_index, data.rel)
    return data

def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        ent_emb, rel_emb = model(data.x, data.edge_index, data.rel, data.edge_index_all, data.rel_all)
    return ent_emb, rel_emb

def train(model, criterion, optimizer, data, train_batch):
    model.train()
    x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
    x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    loss = criterion(x1, x2, data.train_set, train_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def test(model, data):
    ent_emb, rel_emb = get_emb(model, data)
    print('-'*16+'Train_set'+'-'*16)
    get_hits(ent_emb, rel_emb, data.train_set)
    print('-'*16+'Valid_set'+'-'*16)
    get_hits(ent_emb, rel_emb, data.valid_set)
    print('-'*16+'Test_set'+'-'*16)
    get_hits(ent_emb, rel_emb, data.test_set)
    print()

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)
    model = RAGA(data.rel_num, args.e_hidden, args.r_hidden).to(device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x])))
    criterion = MyLoss()
    for epoch in range(args.epoch):
        loss = train(model, criterion, optimizer, data, train_batch)
        print('Epoch:', epoch+1, '/', args.epoch, '\tLoss: %.3f'%loss, '\r', end='')
        if (epoch+1)%args.test_epoch == 0:
            print()
            test(model, data)

if __name__ == '__main__':
    args = parse_args()
    main(args)
