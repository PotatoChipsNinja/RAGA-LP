import torch
import math

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all

def get_hits(encoder, decoder, triples, triple_dict, hits=(1, 3, 10), batch_size=128):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        s, o, r = triples.t()
        emb_ent = encoder()

        batch_num = math.ceil(triples.size(0) / batch_size)
        rank_raw = torch.tensor([], dtype=torch.long)
        rank_filt = torch.tensor([], dtype=torch.long)
        for batch_id in range(batch_num):
            pred = decoder(emb_ent, s[batch_id*batch_size : (batch_id+1)*batch_size], r[batch_id*batch_size : (batch_id+1)*batch_size])
            # raw
            _, idx = pred.sort(descending=True)
            _, rank = idx.sort()
            rank_raw = torch.cat([rank_raw, rank.gather(dim=1, index=o[batch_id*batch_size : (batch_id+1)*batch_size].view(-1, 1))])

            # filt.
            for i, triple in enumerate(triples.tolist()):
                if (triple[0], triple[2]) in triple_dict:
                    temp = pred[i][triple[1]].item()
                    pred[i][triple_dict[(triple[0], triple[2])]] = 0
                    pred[i][triple[1]] = temp
            _, idx = pred.sort(descending=True)
            _, rank = idx.sort()
            rank_filt = torch.cat([rank_filt, rank.gather(dim=1, index=o[batch_id*batch_size : (batch_id+1)*batch_size].view(-1, 1))])

        rank_raw = rank_raw + 1
        rank_filt = rank_filt + 1

        print('Raw:\t', end='')
        for k in hits:
            print('Hits@%d: %.4f    ' % (k, (rank_raw<=k).sum().item()/rank_raw.size(0)), end='')
        print('MRR: %.4f' % (1/rank_raw).mean().item())

        print('Filt.:\t', end='')
        for k in hits:
            print('Hits@%d: %.4f    ' % (k, (rank_filt<=k).sum().item()/rank_filt.size(0)), end='')
        print('MRR: %.4f' % (1/rank_filt).mean().item())
