import torch

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all

# 产生所有头实体替换和尾实体替换的三元组，其中正例放在第一个
def get_candidate(triple, ent_num, all_triple):
    raw = triple.unsqueeze(dim=0)
    alt_sbj = raw.repeat(ent_num, 1)
    alt_sbj[:, 0] = torch.tensor(range(ent_num))
    alt_obj = raw.repeat(ent_num, 1)
    alt_obj[:, 1] = torch.tensor(range(ent_num))
    raw = torch.cat((alt_sbj, alt_obj), dim=0).tolist()
    filt = raw[:]

    raw = {tuple(tri) for tri in raw} # 转成集合去重
    raw.discard(tuple(triple.tolist()))     # 删除正例
    filt = raw - all_triple                 # 计算 filt 集
    raw = list(raw)
    filt = list(filt)
    raw.insert(0, tuple(triple.tolist()))   # 把正例加到最前面
    filt.insert(0, tuple(triple.tolist()))  # 把正例加到最前面

    return torch.tensor(raw).to(triple.device), torch.tensor(filt).to(triple.device)

# 产生所有头实体替换的三元组，其中正例放在第一个
def get_candidate_s(triple, ent_num, all_triple):
    triple_2_dim = triple.unsqueeze(dim=0)
    alt_sbj = triple_2_dim.repeat(ent_num, 1)
    alt_sbj[:, 0] = torch.tensor(range(ent_num))
    raw = alt_sbj.tolist()
    filt = raw[:]

    raw = {tuple(tri) for tri in raw}       # 转成集合去重
    raw.discard(tuple(triple.tolist()))     # 删除正例
    filt = raw - all_triple                 # 计算 filt 集

    # 转换回 tensor
    raw = torch.tensor(list(raw)).to(triple.device)
    filt = torch.tensor(list(filt)).to(triple.device)

    # 把正例加到最前面
    raw = torch.cat((triple_2_dim, raw), dim=0)
    filt = torch.cat((triple_2_dim, filt), dim=0)

    return raw, filt

# 产生所有尾实体替换的三元组，其中正例放在第一个
def get_candidate_o(triple, ent_num, all_triple):
    triple_2_dim = triple.unsqueeze(dim=0)
    alt_obj = triple_2_dim.repeat(ent_num, 1)
    alt_obj[:, 1] = torch.tensor(range(ent_num))
    raw = alt_obj.tolist()
    filt = raw[:]

    raw = {tuple(tri) for tri in raw}       # 转成集合去重
    raw.discard(tuple(triple.tolist()))     # 删除正例
    filt = raw - all_triple                 # 计算 filt 集

    # 转换回 tensor
    raw = torch.tensor(list(raw)).to(triple.device)
    filt = torch.tensor(list(filt)).to(triple.device)

    # 把正例加到最前面
    raw = torch.cat((triple_2_dim, raw), dim=0)
    filt = torch.cat((triple_2_dim, filt), dim=0)

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
    neg_sbj[:, :, 0] = (torch.rand(train_set.size(0), k).to(train_set.device)*ent_num).long()

    # 随机替换尾实体
    neg_obj = pos.repeat(1, k, 1)
    neg_obj[:, :, 1] = (torch.rand(train_set.size(0), k).to(train_set.device)*ent_num).long()

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

def get_hits(ent_emb, rel_emb, data, triples, hits=(1, 3, 10)):
    rank_raw = []
    rank_filt = []

    cnt = 0
    total = len(triples)
    for triple in triples:
        raw, filt = get_candidate_s(triple, data.ent_num, data.all_triple)

        score = get_score_triples(ent_emb, rel_emb, raw)
        _, idx = score.sort(descending=True)
        _, rank = idx.sort()
        rank_raw.append(rank[0] + 1)

        score = get_score_triples(ent_emb, rel_emb, filt)
        _, idx = score.sort(descending=True)
        _, rank = idx.sort()
        rank_filt.append(rank[0] + 1)


        raw, filt = get_candidate_o(triple, data.ent_num, data.all_triple)

        score = get_score_triples(ent_emb, rel_emb, raw)
        _, idx = score.sort(descending=True)
        _, rank = idx.sort()
        rank_raw.append(rank[0] + 1)

        score = get_score_triples(ent_emb, rel_emb, filt)
        _, idx = score.sort(descending=True)
        _, rank = idx.sort()
        rank_filt.append(rank[0] + 1)

        cnt = cnt + 1
        print('Get rank: %d / %d (%.2f%%)\r' % (cnt, total, cnt*100/total), end='')
    print()

    # raw
    print('Raw:\t', end='')
    rank = torch.tensor(rank_raw)
    for k in hits:
        print('Hits@%d: %.4f    ' % (k, (rank<=k).sum().item()/rank.size(0)), end='')
    print('MRR: %.4f' % (1/rank).mean().item())

    # filt.
    print('Filt.:\t', end='')
    rank = torch.tensor(rank_filt)
    for k in hits:
        print('Hits@%d: %.4f    ' % (k, (rank<=k).sum().item()/rank.size(0)), end='')
    print('MRR: %.4f' % (1/rank).mean().item())
