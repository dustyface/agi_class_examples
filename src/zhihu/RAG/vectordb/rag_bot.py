"""
Base class for RAG bot, RAG Bot for ERNIE, 
RAG Bot with CascadeDocParser, RAG Bot with CrossEncoder
"""
from typing import Union
from sentence_transformers import CrossEncoder
from zhihu.RAG.common import parse_paragraph_from_pdf
from zhihu.RAG.common import cascade_split_text
from zhihu.RAG.vectordb.chrome_wrapper import ChromaDBWrapper
from zhihu.RAG.vectordb.embeddings import get_embeddings
from zhihu.common.api import Session
from zhihu.RAG.prompt import build_prompt, PROMPT_TEMPLATE


class RAGBot:
    """ The RAG Pipeline Class"""
    prompt_template = PROMPT_TEMPLATE

    def __init__(self, bot_name: str, *, _prompt_template=None):
        self.chromadb_wrapper = ChromaDBWrapper(bot_name, get_embeddings)
        if _prompt_template is not None:
            self.propmt_template = _prompt_template
        self.session = Session()

    def _parse_document(self, in_file, *, page_numbers=None):
        return parse_paragraph_from_pdf(in_file, page_numbers=page_numbers)

    def prepare(self, in_file):
        """加载和拆分文档, 灌入ChromaDB"""
        paras = self._parse_document(in_file)
        self.chromadb_wrapper.add_documenet(paras)

    # chromadb的search()返回的dict结构如下：
    # { ids: [[], []..., []], distances: [[0.289, 0.31xxx, ...],[],...[]]},
    #   "embeddings": None, "documents": [["", "", ..., ""], []]
    def search(self, query: Union[str, list[str]], *, top_n=3):
        """ Search the ChromaDB """
        return self.chromadb_wrapper.search(query, top_n=top_n)

    def chat(self, user_query: str, *, top_n=5):
        """ Chat with the OpenAI LLM """
        result = self.search(user_query, top_n=top_n)
        result_list = [s for s in result["documents"][0]]
        prompt = build_prompt(
            self.prompt_template,
            info=result_list,
            query=user_query)
        print("user_query=", user_query, "\nprompt=", prompt)
        return self.session.get_completion(prompt)


class RAGBotWithCascadeDocParser(RAGBot):
    """ RAG Bot with CascadeDocParser """

    def _parse_document(self, in_file, *, page_numbers=None):
        paras = super()._parse_document(in_file, page_numbers=page_numbers)
        return cascade_split_text(paras)


class RAGBotwithCrossEncoder(RAGBot):
    """ RAG Bot with CrossEncoder, re-sorting the search result """

    def resort_search_result(self, user_query, search_doc_list) -> list[tuple]:
        """ Re-sort the search result using CrossEncoder """
        cross_encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        scores = cross_encoder_model.predict(
            [(user_query, doc) for doc in search_doc_list])
        return sorted(
            zip(scores, search_doc_list),
            key=lambda x: x[0],
            reverse=True
        )

    def chat(self, user_query: str, *, top_n=5):
        result = self.search(user_query, top_n=top_n)
        # 把搜索到的结果, 使用CrossEncoder重新排序
        sorted_search_list = self.resort_search_result(
            user_query, result["documents"][0])
        prompt = build_prompt(
            self.prompt_template,
            info=list(map(lambda x: x[1], sorted_search_list)),
            query=user_query
        )
        print("user_query=", user_query, "\nprompt=", prompt)
        return self.session.get_completion(prompt)
