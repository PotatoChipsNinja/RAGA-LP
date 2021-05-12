import torch

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all

def get_candidate(all_triple, triples, ent_num):
    raw = triples.unsqueeze(dim=1)
    alt_sbj = raw.repeat(1, ent_num, 1)
    alt_sbj[:, :, 0] = torch.tensor(range(ent_num))
    alt_obj = raw.repeat(1, ent_num, 1)
    alt_obj[:, :, 1] = torch.tensor(range(ent_num))
    raw = torch.cat((raw, alt_sbj, alt_obj), dim=1).tolist()
    filt = raw[:]
    for i in range(len(raw)):
        raw[i] = {tuple(triple) for triple in raw[i]}  # 转成集合去重
        raw[i].discard(tuple(triples[i].tolist()))     # 删除正例
        filt[i] = raw[i] - all_triple                  # 计算 filt 集
        raw[i] = list(raw[i])
        filt[i] = list(filt[i])
        raw[i].insert(0, tuple(triples[i].tolist()))   # 把正例加到最前面
        filt[i].insert(0, tuple(triples[i].tolist()))  # 把正例加到最前面
    return raw, filt

def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        ent_emb, rel_emb = model(data.x, data.edge_index, data.rel, data.edge_index_all, data.rel_all)
    return ent_emb, rel_emb

def get_train_batch(train_set, ent_num, k=5):
    pos = train_set.unsqueeze(dim=1)

    # 随机替换头实体
    neg_sbj = pos.repeat(1, k, 1)
    neg_sbj[:, :, 0] = (torch.rand(train_set.size(0), k)*ent_num).long()

    # 随机替换尾实体
    neg_obj = pos.repeat(1, k, 1)
    neg_obj[:, :, 1] = (torch.rand(train_set.size(0), k)*ent_num).long()

    train_batch = torch.cat((pos, neg_sbj, neg_obj), dim=1)
    return train_batch

def get_score(ent_emb, rel_emb, batch):
    s = ent_emb[batch[:, :, 0]]
    r = rel_emb[batch[:, :, 2]]
    o = ent_emb[batch[:, :, 1]]
    return torch.sum(s*r*o, dim=2)

def get_score_triples(ent_emb, rel_emb, triples):
    s = ent_emb[triples[:, 0]]
    r = rel_emb[triples[:, 2]]
    o = ent_emb[triples[:, 1]]
    return torch.sum(s*r*o, dim=1)

def get_hits(ent_emb, rel_emb, data, train_set=False, valid_set=False, test_set=False, hits=(1, 3, 10)):
    if train_set:
        raw = data.raw_train
        filt = data.filt_train
    elif valid_set:
        raw = data.raw_valid
        filt = data.filt_valid
    elif test_set:
        raw = data.raw_test
        filt = data.filt_test

    # raw
    print('Raw:\t', end='')
    score = get_score(ent_emb, rel_emb, torch.tensor(raw))
    _, idx = score.sort(descending=True)
    _, rank = idx.sort()
    rank = rank[:, 0] + 1
    for k in hits:
        print('Hits@%d: %.4f    ' % (k, (rank<=k).sum().item()/rank.size(0)), end='')
    print('MRR: %.4f' % (1/rank).mean().item())

    # filt.
    print('Filt.:\t', end='')
    rank_list = []
    for triples in filt:
        score = get_score_triples(ent_emb, rel_emb, triples)
        _, idx = score.sort(descending=True)
        _, rank = idx.sort()
        rank_list.append(rank[0] + 1)
    rank = torch.tensor(rank_list)
    for k in hits:
        print('Hits@%d: %.4f    ' % (k, (rank<=k).sum().item()/rank.size(0)), end='')
    print('MRR: %.4f' % (1/rank).mean().item())
