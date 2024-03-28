import qianfan
import logging

logger = logging.getLogger(__name__)

em = qianfan.Embedding()

def get_embeddings(texts: list[str], model="Embedding-V1", **kwargs):
    data = em.do(model=model, texts=texts, **kwargs)
    return [x["embedding"] for x in data["data"]]