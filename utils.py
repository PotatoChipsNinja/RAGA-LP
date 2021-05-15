import torch

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all

def get_hits(encoder, decoder, triples, triple_dict, hits=(1, 3, 10)):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        s, o, r = triples.t()
        emb_ent = encoder()
        pred = decoder(emb_ent, s, r)

        # raw
        _, idx = pred.sort(descending=True)
        _, rank = idx.sort()
        rank = rank.gather(dim=1, index=o.view(-1, 1)) + 1
        print('Raw:\t', end='')
        for k in hits:
            print('Hits@%d: %.4f    ' % (k, (rank<=k).sum().item()/rank.size(0)), end='')
        print('MRR: %.4f' % (1/rank).mean().item())

        # filt.
        for i, triple in enumerate(triples.tolist()):
            if (triple[0], triple[2]) in triple_dict:
                temp = pred[i][triple[1]]
                pred[i][triple_dict[(triple[0], triple[2])]] = 0
                pred[i][triple[1]] = temp
        _, idx = pred.sort(descending=True)
        _, rank = idx.sort()
        rank = rank.gather(dim=1, index=o.view(-1, 1)) + 1
        print('Filt.:\t', end='')
        for k in hits:
            print('Hits@%d: %.4f    ' % (k, (rank<=k).sum().item()/rank.size(0)), end='')
        print('MRR: %.4f' % (1/rank).mean().item())
