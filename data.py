import os
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index

from utils import get_candidate

class MyData(InMemoryDataset):
    def __init__(self, dataset, e_hidden=300, seed=1):
        self.dataset = dataset
        self.e_hidden = e_hidden
        self.seed = seed
        root = "data/" + dataset
        torch.manual_seed(seed)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return '%s_%d_%d.pt' % (self.dataset, self.e_hidden, self.seed)

    def process(self):
        ent_path = os.path.join(self.root, "entity2id.txt")
        rel_path = os.path.join(self.root, "relation2id.txt")
        train_path = os.path.join(self.root, "train2id.txt")
        valid_path = os.path.join(self.root, "valid2id.txt")
        test_path = os.path.join(self.root, "test2id.txt")

        train_set = read_txt_array(train_path, sep=' ', dtype=torch.long)
        valid_set = read_txt_array(valid_path, sep=' ', dtype=torch.long)
        test_set = read_txt_array(test_path, sep=' ', dtype=torch.long)

        # 随机初始化嵌入
        ent_num = int(open(ent_path, 'r', encoding='utf-8').readline())
        x = torch.nn.Embedding(ent_num, self.e_hidden).weight.data

        rel_num = int(open(rel_path, 'r', encoding='utf-8').readline())

        # 利用训练集三元组生成图
        sbj, obj, rel = train_set.t()
        edge_index = torch.stack([sbj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)

        # 计算评测的候选三元组，将正确的放在第一个
        all_triple = torch.cat([train_set, valid_set, test_set]).tolist()
        all_triple = {tuple(triple) for triple in all_triple}
        raw_train, filt_train = get_candidate(all_triple, train_set, ent_num)
        raw_valid, filt_valid = get_candidate(all_triple, valid_set, ent_num)
        raw_test, filt_test = get_candidate(all_triple, test_set, ent_num)

        '''
        raw_train = train_set.unsqueeze(dim=1)
        alt_sbj = raw_train.repeat(1, ent_num, 1)
        alt_sbj[:, :, 0] = torch.tensor(range(ent_num))
        alt_obj = raw_train.repeat(1, ent_num, 1)
        alt_obj[:, :, 1] = torch.tensor(range(ent_num))
        raw_train = torch.cat((raw_train, alt_sbj, alt_obj), dim=1)
        raw_train = raw_train.tolist()
        filt_train = raw_train[:]
        for i in range(len(raw_train)):
            raw_train[i] = {tuple(triple) for triple in raw_train[i]}
            raw_train[i].discard(tuple(train_set[i].tolist()))
            filt_train[i] = raw_train[i] - all_triple
            raw_train[i] = list(raw_train[i])
            filt_train[i] = list(filt_train[i])
            raw_train[i].insert(0, tuple(train_set[i].tolist()))
            filt_train[i].insert(0, tuple(train_set[i].tolist()))
        '''

        data = Data(x = x, ent_num=ent_num, rel_num=rel_num, edge_index=edge_index, rel=rel,
                    train_set=train_set, valid_set=valid_set, test_set=test_set,
                    raw_train=raw_train, raw_valid=raw_valid, raw_test=raw_test,
                    filt_train=filt_train, filt_valid=filt_valid, filt_test=filt_test)
        torch.save(self.collate([data]), self.processed_paths[0])
