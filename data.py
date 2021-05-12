import os
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index

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
        sbj, rel, obj = train_set.t()
        edge_index = torch.stack([sbj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)

        data = Data(x = x, ent_num=ent_num, rel_num=rel_num, edge_index=edge_index, rel=rel,
                    train_set=train_set, valid_set=valid_set, test_set=test_set)
        torch.save(self.collate([data]), self.processed_paths[0])
