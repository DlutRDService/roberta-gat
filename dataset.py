import torch
from torch_geometric.data import Data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_dataset, edge_index):
        self.encoded_dataset = encoded_dataset
        self.edge_index = edge_index

    def __len__(self):
        return len(self.encoded_dataset)

    def __getitem__(self, idx):
        label = torch.tensor(self.encoded_dataset[idx]['label'], dtype=torch.long)
        input_ids = torch.tensor(self.encoded_dataset[idx]['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(self.encoded_dataset[idx]['attention_mask'], dtype=torch.long)
        # 将全图关系索引映射到子图关系索引
        m = min(self.edge_index[idx][0])
        for i in range(len(self.edge_index[idx])):
            for j in range(len(self.edge_index[idx][i])):
                self.edge_index[idx][i][j] -= m
        edge_index = torch.tensor(self.edge_index[idx])
        edge_index = edge_index.t()

        return Data(x=input_ids, edge_index=edge_index, y=label, mask=attention_mask)
        # return {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'labels': label,
        #     'edge_index': edge_index
        # }
