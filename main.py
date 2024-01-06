import torch
from model import RobertaGAT


def sequence_classification(dataset):
    preds = 0
    model = RobertaGAT(roberta_model_name="roberta-base", num_classes=4)
    model.load_state_dict(torch.load('./model/model.pth'))

    model.eval()

    for batch in dataset:
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
    print(preds)


if __name__ == "__main__":
    sequence_classification(data)


