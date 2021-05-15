import os
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset

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

        # 建立 map: (s, r) => o, 用于 filt. 评测
        all_triples = torch.cat([train_set, valid_set, test_set]).tolist()
        triple_dict = {}
        for triple in all_triples:
            if (triple[0], triple[2]) in triple_dict:
                triple_dict[(triple[0], triple[2])].append(triple[1])
            else:
                triple_dict[(triple[0], triple[2])] = [triple[1]]

        data = Data(ent_num=ent_num, rel_num=rel_num, triple_dict=triple_dict,
                    train_set=train_set, valid_set=valid_set, test_set=test_set)
        torch.save(self.collate([data]), self.processed_paths[0])
