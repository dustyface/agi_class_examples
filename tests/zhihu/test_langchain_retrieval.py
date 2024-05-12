""" Test langchain document connector """
import logging
from typing import List
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.documents import Document
from zhihu.LangChain.retrieval import DocumentHandler

logger = logging.getLogger(__name__)


def test_documenthandler():
    """ Test DocumentHandler """
    test_file_path = "src/zhihu/RAG/llama2-test-1-4.pdf"
    doc_handler = DocumentHandler(test_file_path)
    # pylint: disable=unexpected-keyword-arg
    doc_handler.bulk_document(use_timer=True)
    search_docs = doc_handler.retrieve("llama2 chat有多少参数", use_timer=True)
    logger.info(type(search_docs), search_docs)


def test_documenthandler_qianfan():
    """ Test DocumentHandler """
    test_file_path = "src/zhihu/RAG/llama2-test-1-4.pdf"
    doc_handler = DocumentHandler(
        test_file_path, embeddings_constructor=QianfanEmbeddingsEndpoint)
    # pylint: disable=unexpected-keyword-arg
    doc_handler.bulk_document(use_timer=True)
    search_docs: List[Document] = doc_handler.retrieve(
        "llama2 chat有多少参数", use_timer=True)
    logger.info("find result=%s", search_docs[0].page_content)
