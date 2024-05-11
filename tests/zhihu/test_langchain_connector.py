""" Test langchain document connector """
import logging
from zhihu.LangChain.data_connection import DocumentHandler

logger = logging.getLogger(__name__)


def test_documenthandler():
    """ Test DocumentHandler """
    test_file_path = "src/zhihu/RAG/llama2-test-1-4.pdf"
    doc_handler = DocumentHandler(test_file_path)
    # pylint: disable=unexpected-keyword-arg
    doc_handler.bulk_document(use_timer=True)
    search_docs = doc_handler.retrieve("llama2 chat有多少参数", use_timer=True)
    logger.info(type(search_docs), search_docs)
