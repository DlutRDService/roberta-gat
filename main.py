import torch
from model import RobertaGAT
from dataset import CustomDataset
from torch_geometric.loader import DataLoader
from transformers import RobertaTokenizer


def encode_batch(abstract):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return tokenizer(abstract, padding='max_length', truncation=True, max_length=96, return_tensors="pt")


def abstract_reprocess(abstract):
    dataset = {split: dataset[split].map(encode_batch, batched=True) for split in dataset.keys()}
    token_result = token(abstract)
    train_dataset = CustomDataset(token_result, rel, 20, 8)
    return DataLoader(train_dataset, batch_size=1, shuffle=True)


def sequence_classification(data):
    abstract_reprocess(data)
    model = RobertaGAT(roberta_model_name="roberta-base", num_classes=4)
    model.load_state_dict(torch.load('./model/model.pth'))

    model.eval()

    for batch in data:
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

            output, weight1 = model(input_ids, attention_mask, edge_index)

            preds = output.argmax(dim=1)
    return preds


if __name__ == "__main__":
    sequence_classification(data)


