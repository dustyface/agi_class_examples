""" Test the deployed local model """
from zhihu.RAG.deploy.local_model import query_vec, doc_vec
from zhihu.RAG.vectordb.embeddings import cos_sim

# 执行这个测试，会导致下载模型, 由于huggingface被墙，如果没有梯子的情况下
# 可以从如下链接下载model, 使用cache_folder参数指定下载的模型路径
# 链接: https://pan.baidu.com/s/1X0kfNKasvWqCLUEEyAvO-Q?pwd=3v6y 提取码: 3v6y


def test_local_model():
    """ Test the deployed local model """
    for i, vec in enumerate(doc_vec):
        cos_sim_distance = cos_sim(query_vec, vec)
        print(f"cos_sim(query_vec, doc_vec[{i}])={cos_sim_distance}")
