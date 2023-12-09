import py2neo
import itertools
import pandas as pd
import numpy as np
import os
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import torch

# llama = Llama(model_path='./llama-2-7b.Q4_K_M.gguf', embedding=True, n_ctx=4096)

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
    df = pd.read_csv(path)
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
    df = pd.read_csv(path)
    abstract = ''
    abstracts_embedding = []
    for i in range(start, len(df['label'])):
        abstract += df['text'][i]
        if df['label'][i] == 4 and df['label'][i + 1] == 0:
            abstract_embedding = llama.create_embedding(input=abstract).get('data')[0].get('embedding')
            abstracts_embedding.append([i, abstract_embedding])
            np.save(f"./temp/abstract_embedding{i}.npy", abstract_embedding)
            abstract = ''
    tmp = []
    files = os.listdir("./temp")
    for file in files:
        if file.endswith('.npy'):
            tmp.append(np.load(f'./temp/{file}', allow_pickle=True))
    np.array(tmp)
    np.save(f'./temp/abstract_embedding_test.npy', tmp)

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
        if cos >= 0.88:
            rel.append([i, j])
    return rel

def get_edge_index(sen_rel, abs_rel):
    """
    构建图关系
    """
    return torch.tensor(sen_rel + abs_rel)


print(len(np.load('./temp/abstract_embedding_test.npy')))
a = np.load('./temp/abstract_embedding_test.npy')

# test_sen_rel = get_sentence_rel(path='data/test.csv', num=0)
# get_abstract_embedding(path='data/test.csv', start=0)
test_paper_rel = get_paper_rel(a)
print(test_paper_rel)
print(len(test_paper_rel))
# test_rel = get_edge_index(test_sen_rel, test_paper_rel)
