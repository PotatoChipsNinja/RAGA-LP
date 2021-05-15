import math
import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RAGA, ConvE
from data import MyData
from utils import get_hits

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", choices=["WN18RR", "FB15k-237"])
    parser.add_argument("--e_hidden", type=int, default=120)
    parser.add_argument("--r_hidden", type=int, default=40)
    parser.add_argument('--embedding-shape1', type=int, default=20)
    parser.add_argument('--hidden-drop', type=float, default=0.3)
    parser.add_argument('--input-drop', type=float, default=0.2)
    parser.add_argument('--feat-drop', type=float, default=0.2)
    parser.add_argument('--hidden-size', type=int, default=21888) # 注意这个随嵌入维度变化，需要按照报错信息改
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    return args

def train(model, criterion, optimizer, data, train_batch):
    model.train()
    ent_emb, rel_emb = model(data.x, data.edge_index, data.rel, data.edge_index_all, data.rel_all)
    loss = criterion(ent_emb, rel_emb, train_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = MyData(args.data)[0].to(device)

    encoder = RAGA(args, data.ent_num).to(device)
    decoder = ConvE(args, data.ent_num, data.rel_num).to(device)
    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()))
    criterion = nn.BCELoss()

    '''
    encoder.train()
    decoder.train()
    batch = data.train_set[0:10]
    e1 = batch[:, 0]
    rel = batch[:, 2]
    e2 = batch[:, 1]
    e2_multi = torch.zeros(e2.size(0), data.ent_num).to(device).scatter_(1, e2.view(-1, 1), 1)
    # label smoothing
    e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))

    emb_ent = encoder()
    pred = decoder(emb_ent, e1, rel)
    print(pred)
    print(pred.shape)
    '''
    get_hits(encoder, decoder, data.test_set, data.triple_dict)

if __name__ == '__main__':
    args = parse_args()
    main(args)
