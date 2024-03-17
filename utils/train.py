import torch
from datasets import DatasetDict, load_dataset
from torch import nn
from torch_geometric.loader import DataLoader

from utils.dataset import CustomDataset
from utils import spilt_node

from model import RobertaGAT


def train_model(file_path, train_rel, batch_size, num_epochs, learning_rate, weights, weight_decay):
    # 加载数据
    data = load_dataset('csv', data_files=file_path, encoding='utf-8')

    # 构建数据集
    dataset = DatasetDict({'train': data})
    # 划分子图
    train_data = spilt_node(dataset['train']['train'])

    train_dataset = CustomDataset(train_data, train_rel, 20, 8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = RobertaGAT("roberta-base", num_classes=4)

    criterion = nn.CrossEntropyLoss(ignore_index=4, weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch[0]['x']
            attention_mask = batch[0]['mask']
            labels = batch[0]['y']
            edge_index = batch[0]['edge_index']
            num_nodes_graph = 0

            for i in range(1, len(batch)):
                edge_index_tmp = (batch[i]['edge_index'] + num_nodes_graph)
                edge_index = torch.cat((edge_index, edge_index_tmp), dim=1)
                num_nodes_graph += batch[i]['x'].size(0)
                input_ids = torch.cat((input_ids, batch[i]['x']), dim=0)
                attention_mask = torch.cat((attention_mask, batch[i]['mask']), dim=0)
                labels = torch.cat((labels, batch[i]['y']), dim=0)

            input_ids = input_ids.to('cuda:0')
            attention_mask = attention_mask.to('cuda:0')
            edge_index = edge_index.to('cuda:1')
            labels = labels.to('cuda:1')

            optimizer.zero_grad()
            output, weight1 = model(input_ids, attention_mask, edge_index)
            loss = criterion(output.to("cuda:1"), labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch +1}/{num_epochs}, Training Loss: {avg_loss}")


# if __name__ == "__main__":
#     train_model(sys.argv)
