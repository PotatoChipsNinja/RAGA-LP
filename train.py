import math
import argparse
import itertools

import torch
import torch.nn.functional as F

from model import RAGA
from data import MyData
from loss import MyLoss
from utils import add_inverse_rels, get_emb, get_hits, get_train_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", choices=["WN18RR", "FB15k-237"])
    parser.add_argument("--e_hidden", type=int, default=300)
    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--k", type=int, default=5)  # 对每个正样本取 2k 个负样本，其中 k 个随机替换头实体，k 个随机替换尾实体
    args = parser.parse_args()
    return args

def init_data(args, device):
    data = MyData(args.data, e_hidden=args.e_hidden)[0]
    data.x = F.normalize(data.x, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all, data.rel_all = add_inverse_rels(data.edge_index, data.rel)
    return data

def train(model, criterion, optimizer, data, train_batch):
    model.train()
    ent_emb, rel_emb = model(data.x, data.edge_index, data.rel, data.edge_index_all, data.rel_all)
    loss = criterion(ent_emb, rel_emb, train_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def test(model, data):
    ent_emb, rel_emb = get_emb(model, data)
    '''
    print('-'*16+'Train_set'+'-'*16)
    get_hits(ent_emb, rel_emb, data, data.train_set)
    print('-'*16+'Valid_set'+'-'*16)
    get_hits(ent_emb, rel_emb, data, data.valid_set)
    '''
    print('-'*16+'Test_set'+'-'*16)
    get_hits(ent_emb, rel_emb, data, data.test_set)
    print()

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)
    model = RAGA(data.rel_num, args.e_hidden, args.r_hidden).to(device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x])))
    criterion = MyLoss()
    batch_num = math.ceil(data.train_set.size(0) / args.batch_size)
    avg_loss = 0
    for epoch in range(args.epoch):
        losses = []
        train_set = data.train_set[torch.randperm(data.train_set.size(0))] # 随机打乱训练集
        for iteration in range(batch_num):
            batch = train_set[iteration*args.batch_size : (iteration+1)*args.batch_size]
            train_batch = get_train_batch(batch, data.ent_num, args.k)
            loss = train(model, criterion, optimizer, data, train_batch)
            losses.append(loss)
            print('Epoch: %d / %d, Iteration: %d / %d, Loss: %.3f, Avg_Loss: %.3f\r'
                % (epoch+1, args.epoch, iteration+1, batch_num, loss, avg_loss), end='')
        avg_loss = torch.tensor(losses).mean().item()
        if (epoch+1)%args.test_epoch == 0:
            print()
            test(model, data)

if __name__ == '__main__':
    args = parse_args()
    main(args)
