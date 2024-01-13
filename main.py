import json
import sys

import torch
from model import RobertaGAT
from utils import encode_batch
import pandas as pd


def get_result(data_path):
    # 加载数据
    data = pd.read_csv(data_path, encoding='utf-8')
    data = data["paper"]
    data = data.apply(json.loads)
    # data 是一个列表存的是每个子图的信息。
    results = []
    for i in range(len(data)):
        tokenors = encode_batch(data[i]['seq'])
        model = RobertaGAT(roberta_model_name="roberta-base", num_classes=4)
        model.load_state_dict(torch.load('model/model.pth'))
        model.eval()
        output = model(tokenors['input_ids'], tokenors['attention_mask'], torch.tensor(data[i]['rel']))[0]
        result = output.argmax(dim=1).tolist()
        results.append({"seq": data[i]['seq'][1:], "label": result[1:]})
    return results


if __name__ == "__main__":
    results = get_result(sys.argv[1])
    print(results)
