import torch
import torch.nn as nn

from utils import get_score

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ent_emb, rel_emb, train_batch):
        score = get_score(ent_emb, rel_emb, train_batch)
        lossFunc = nn.CrossEntropyLoss()
        loss = lossFunc(score, torch.zeros(score.size(0), dtype=torch.long))
        return loss
