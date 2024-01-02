import torch
from torch_geometric.data import Data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_dataset, edge_index, max_nodes, min_nodes):
        self.encoded_dataset = encoded_dataset
        self.edge_index = edge_index
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes

        # 过滤节点数过低过高的图
        self.filtered_indices = [
            i for i, data in enumerate(self.encoded_dataset)
            if self.min_nodes <= len(data['input_ids']) <= self.max_nodes
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        real_idx = self.filtered_indices[idx]
        label = torch.tensor(self.encoded_dataset[real_idx]['label'], dtype=torch.long)
        input_ids = torch.tensor(self.encoded_dataset[real_idx]['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(self.encoded_dataset[real_idx]['attention_mask'], dtype=torch.long)

        # 将全图关系索引映射到子图关系索引
        m = min(self.edge_index[real_idx][0])
        for i in range(len(self.edge_index[real_idx])):
            for j in range(len(self.edge_index[real_idx][i])):
                self.edge_index[real_idx][i][j] -= m
        edge_index = torch.tensor(self.edge_index[real_idx])
        edge_index = edge_index.t()

        return Data(x=input_ids, edge_index=edge_index, y=label, mask=attention_mask)
