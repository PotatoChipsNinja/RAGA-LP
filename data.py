import os
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index

from utils import add_inverse_rels

class MyData(InMemoryDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        root = "data/" + dataset
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return '%s.pt' % self.dataset

    def process(self):
        ent_path = os.path.join(self.root, "entity2id.txt")
        rel_path = os.path.join(self.root, "relation2id.txt")
        train_path = os.path.join(self.root, "train2id.txt")
        valid_path = os.path.join(self.root, "valid2id.txt")
        test_path = os.path.join(self.root, "test2id.txt")

        train_set = read_txt_array(train_path, sep=' ', dtype=torch.long)
        valid_set = read_txt_array(valid_path, sep=' ', dtype=torch.long)
        test_set = read_txt_array(test_path, sep=' ', dtype=torch.long)

        ent_num = int(open(ent_path, 'r', encoding='utf-8').readline())
        rel_num = int(open(rel_path, 'r', encoding='utf-8').readline())

        # 利用训练集三元组生成图
        sbj, obj, rel = train_set.t()
        edge_index = torch.stack([sbj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)

        # 加入反关系
        edge_index_all, rel_all = add_inverse_rels(edge_index, rel)

        # 建立 map: (s, r) => o, 用于 filt. 评测
        all_triples = torch.cat([train_set, valid_set, test_set]).tolist()
        triple_dict = {}
        for triple in all_triples:
            if (triple[0], triple[2]) in triple_dict:
                triple_dict[(triple[0], triple[2])].append(triple[1])
            else:
                triple_dict[(triple[0], triple[2])] = [triple[1]]

        data = Data(ent_num=ent_num, rel_num=rel_num, triple_dict=triple_dict,
                    edge_index=edge_index, rel=rel,
                    edge_index_all=edge_index_all, rel_all=rel_all,
                    train_set=train_set, valid_set=valid_set, test_set=test_set)
        torch.save(self.collate([data]), self.processed_paths[0])
