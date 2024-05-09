""" Test the deployed local model """
from zhihu.RAG.deploy.local_model import query_vec, doc_vec
from zhihu.RAG.vectordb.embeddings import cos_sim


def test_local_model():
    """ Test the deployed local model """
    for i, vec in enumerate(doc_vec):
        cos_sim_distance = cos_sim(query_vec, vec)
        print(f"cos_sim(query_vec, doc_vec[{i}])={cos_sim_distance}")
