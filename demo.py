import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import pandas as pd
import numpy as np
import os
import csv



# 图关系
# 训练集(71251,4519)
# 测试集合(15250,965)
# 验证集 (16073,1028)
def get_sentence_rel(path, num):
    """
    以文章为单位，构建关系（abs_sentence-title）
    :param path:
    :param num:
    :return:
    """
    df = pd.read_csv(path, encoding='GB2312')
    relationship = []
    for i in range(0, len(df['label'])):
        if df["label"][i] != 5:
            relationship.append([i, num])
        if df['label'][i] == 4 and df['label'][i + 1] == 0:
            num += 1
            continue
    return relationship


def get_abstract_embedding(path, start):
    """
    Llama编码获取摘要embedding。处理结果为[[][]]
    :param path:
    :param start:
    """
    df = pd.read_csv(path, encoding='utf-8')
    abstract = ''
    for i in range(start, len(df['label'])):
        abstract += df['text'][i]
        if df['label'][i] == 4 and df['label'][i + 1] == 0:
            abstract_embedding = llama.create_embedding(input=abstract).get('data')[0].get('embedding')
            np.save(f"./temp/abstract_embedding{i}.npy", abstract_embedding)
            with open('./data/abstract_embedding_validation.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([abstract, f'abstract_embedding{i}.npy'])
            abstract = ''
    tmp = []
    files = os.listdir("./temp", )
    # 获取每个文件的完整路径
    full_paths = [os.path.join("./temp", file) for file in files]
    # 按创建时间对文件进行排序
    sorted_files = sorted(full_paths, key=os.path.getctime)
    for file in sorted_files:
        if file.endswith('.npy'):
            tmp.append(np.load(f'{file}', allow_pickle=True))
    np.array(tmp)
    np.save(f'./data/abstract_embedding_validation.npy', tmp)


def cos_sim(a, b):
    return cosine_similarity([a, b])[0][1]


def get_paper_rel(array):
    """
    获取文章直接的关系（title-title）
    :param array:
    :return:
    """
    rel = []
    for i, j in itertools.combinations(range(len(array)), 2):
        cos = cos_sim(array[i], array[j])
        if cos >= 0.95:
            rel.append([i, j])
    return rel


def get_edge_index(sen_rel, abs_rel):
    """
    构建图关系
    """
    return torch.tensor(sen_rel + abs_rel)

data = []
data.append(
    {
        "object": "embedding",
        "embedding": ["a","b"],
        "index": 0,
    }
)
print(data)


llama = Llama(model_path='./llama-2-7b.Q4_K_M.gguf', embedding=True, n_ctx=2048, n_gpu_layers=30)
get_abstract_embedding(path='data/validation.csv', start=0)
