from typing import Union
import chromadb
from chromadb.config import Settings
from zhihu.RAG.vectordb.embeddings import strip_whitespace


class ChromaDBWrapper:
    def __init__(self, collection_name, embeddings_fn):
        # 连接内存中的vectordb
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 非实验不需要每次reset
        chroma_client.reset()
        # 创建collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embeddings_fn = embeddings_fn

    def add_documenet(self, document, metadatas={}):
        """把text embeddings和文档灌入到collection中"""
        filtered_doc = strip_whitespace(document)
        embeddings = self.embeddings_fn(filtered_doc)
        self.collection.add(
            embeddings=embeddings,
            documents=filtered_doc,
            ids=[f"id{i}" for i in range(len(filtered_doc))],
        )
    
    def search(self, query:Union[str, list[str]], top_n=3):
        """检索向量数据库"""
        if isinstance(query, str):
            query = [query]
        return self.collection.query(
            query_embeddings=self.embeddings_fn(query), n_results=top_n
        )