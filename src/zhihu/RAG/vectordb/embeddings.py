""" 用于获取文本的embedding, 以及计算embedding之间的相似度 """
import numpy as np
from zhihu.common.api import client


def strip_whitespace(text_list: list[str]) -> list[str]:
    """ 去掉text_list中的空字符串, 填入get_embeddings的texts参数时, 不能有空字符串 """
    return list(filter(lambda x: x.strip(), text_list))


def get_embeddings(texts, model="text-embedding-ada-002"):
    """ OpenAI提供的文本embedding服务 """
    # 注意:
    # 参数texts是list, 每个元素是一个text sentence，对应一个n维embedding;
    # texts的元素不能是空字符串，或包含whitespace的空字符串;
    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]


def cos_sim(a, b):
    """余弦距离, 1表示0度夹角, a,b完全相同; 0表示90度夹角, a,b完全不相关"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def l2(a, b):
    """L2距离, 0表示完全相同, 越大表示越不相同"""
    x = np.asarray(a) - np.asarray(b)
    return np.linalg.norm(x)
