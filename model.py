import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from torch_geometric.nn import GATConv


class GATConvWithAttention(GATConv):
    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=True):
        out, attention_weights = super().forward(x, edge_index, edge_attr, size, return_attention_weights)
        return out, attention_weights


class RobertaPartialEncoder(nn.Module):
    def __init__(self, roberta_model_name, start_layer, end_layer):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.encoder_layers = self.roberta.encoder.layer[start_layer:end_layer]

    def forward(self, hidden_states, attention_mask):

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)  # 转换为与 hidden_states 相同的类型
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer in self.encoder_layers:
            layer_outputs = layer(hidden_states, attention_mask=extended_attention_mask)
            hidden_states = layer_outputs[0]

        return hidden_states


class RobertaGAT(nn.Module):
    def __init__(self, roberta_model_name, num_classes):
        super(RobertaGAT, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_embeddings = self.roberta.embeddings.to('cuda:0')
        self.roberta_first_half = RobertaPartialEncoder('roberta-base', 0, 6).to('cuda:0')
        self.roberta_second_half = RobertaPartialEncoder('roberta-base', 6, 12).to('cuda:1')
        self.gat = GATConvWithAttention(self.roberta.config.hidden_size, num_classes).to('cuda:1')

    def forward(self, input_ids, attention_mask, edge_index):
        outputs = self.roberta_embeddings(input_ids=input_ids)
        outputs = self.roberta_first_half(outputs, attention_mask=attention_mask)

        # 将中间输出移动到 GPU 1 并进行后续处理
        outputs = outputs.to('cuda:1')
        attention_mask = attention_mask.to('cuda:1')
        outputs = self.roberta_second_half(outputs, attention_mask=attention_mask)

        # outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs[:, 0, :].to('cuda:1')
        # print(edge_index, sentence_embeddings, self.gat)

        gat_output, attention_weights = self.gat(sentence_embeddings, edge_index)
        return gat_output, attention_weights
    #
    # def __init__(self, roberta_model_name, num_classes):
    #     super(RobertaGAT, self).__init__()
    #     self.roberta = RobertaModel.from_pretrained(roberta_model_name).to("cuda:0")
    #
    #     # 将 RoBERTa 模型的一半层放在一个 GPU 上
    #     self.roberta_half = nn.Sequential(*list(self.roberta.encoder.layer.children())[:6]).to('cuda:0')
    #     self.roberta_other_half = nn.Sequential(*list(self.roberta.encoder.layer.children())[6:]).to('cuda:1')
    #
    #     self.gat = GATConvWithAttention(self.roberta.config.hidden_size, num_classes).to('cuda:1')
    #
    # def forward(self, input_ids, attention_mask, edge_index):
    #     # 将输入传递到 cuda:0
    #     input_ids = input_ids[0].to('cuda:0')
    #     attention_mask = attention_mask[0].to('cuda:0')
    #
    #     # 通过 RoBERTa 的第一半
    #     outputs = self.roberta.embeddings(input_ids=input_ids)
    #     for layer in self.roberta_half:
    #         outputs = layer(outputs, attention_mask)
    #
    #     # 将中间结果传递到 cuda:1
    #     outputs = outputs.to('cuda:1')
    #
    #     # 通过 RoBERTa 的另一半
    #     for layer in self.roberta_other_half:
    #         outputs = layer(outputs, attention_mask.to('cuda:1'))
    #
    #     sentence_embeddings = outputs[0][:, 0, :]
    #
    #     # GAT 部分
    #     gat_output, attention_weights = self.gat(sentence_embeddings, edge_index[0].to('cuda:1').t())
    #     return gat_output, attention_weights
