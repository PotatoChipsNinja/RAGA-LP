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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--e-hidden", type=int, default=120)
    parser.add_argument("--r-hidden", type=int, default=40)
    parser.add_argument('--embedding-shape1', type=int, default=20)
    parser.add_argument('--hidden-drop', type=float, default=0.3)
    parser.add_argument('--input-drop', type=float, default=0.2)
    parser.add_argument('--feat-drop', type=float, default=0.2)
    parser.add_argument('--hidden-size', type=int, default=21888) # 注意这个随嵌入维度变化, 需要按照报错信息改
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--test-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--test-batch-size", type=int, default=128)
    args = parser.parse_args()
    return args

def train(encoder, decoder, criterion, optimizer, args, data, train_batch):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    e1, e2, rel = train_batch.t()
    e2_multi = torch.zeros(e2.size(0), data.ent_num).to(train_batch.device).scatter_(1, e2.view(-1, 1), 1)
    # label smoothing
    e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))

    emb_ent = encoder()
    pred = decoder(emb_ent, e1, rel)
    loss = criterion(pred, e2_multi)
    loss.backward()
    optimizer.step()
    return loss.item()

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = MyData(args.data)[0].to(device)
    encoder = RAGA(args, data.ent_num).to(device)
    decoder = ConvE(args, data.ent_num, data.rel_num).to(device)
    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr)
    criterion = nn.BCELoss()

    batch_num = math.ceil(data.train_set.size(0) / args.batch_size)
    avg_loss = 0
    for epoch in range(args.epoch):
        losses = []
        train_set = data.train_set[torch.randperm(data.train_set.size(0))] # 随机打乱训练集
        for iteration in range(batch_num):
            train_batch = train_set[iteration*args.batch_size : (iteration+1)*args.batch_size]
            loss = train(encoder, decoder, criterion, optimizer, args, data, train_batch)
            losses.append(loss)
            print('Epoch: %d / %d, Iteration: %d / %d, Loss: %.3f, Avg_Loss: %.3f\r'
                % (epoch+1, args.epoch, iteration+1, batch_num, loss, avg_loss), end='')
        avg_loss = torch.tensor(losses).mean().item()
        if (epoch+1)%args.test_epoch == 0:
            print()
            get_hits(encoder, decoder, data.test_set, data.triple_dict, batch_size=args.test_batch_size)

if __name__ == '__main__':
    args = parse_args()
    main(args)
