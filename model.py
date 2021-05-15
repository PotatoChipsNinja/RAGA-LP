import torch
import torch.nn as nn
import torch.nn.functional as F

class RAGA(nn.Module):
    def __init__(self, args, ent_num):
        super(RAGA, self).__init__()
        self.emb_ent = nn.Embedding(ent_num, 2*args.e_hidden+4*args.r_hidden)
        nn.init.xavier_normal_(self.emb_ent.weight.data)

    def forward(self):
        return self.emb_ent.weight

class ConvE(nn.Module):
    def __init__(self, args, ent_num, rel_num):
        super(ConvE, self).__init__()
        self.emb_rel = nn.Embedding(rel_num, 2*args.e_hidden+4*args.r_hidden)
        nn.init.xavier_normal_(self.emb_rel.weight.data)
        self.inp_drop = nn.Dropout(args.input_drop)
        self.hidden_drop = nn.Dropout(args.hidden_drop)
        self.feature_map_drop = nn.Dropout2d(args.feat_drop)
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = (2*args.e_hidden+4*args.r_hidden) // self.emb_dim1

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(2*args.e_hidden+4*args.r_hidden)
        self.register_parameter('b', nn.Parameter(torch.zeros(ent_num)))
        self.fc = nn.Linear(args.hidden_size, 2*args.e_hidden+4*args.r_hidden)

    def forward(self, emb_ent, e1, rel):
        e1_embedded = emb_ent[e1].view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        #print(x.shape) # 查看 hidden_size
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_ent.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
