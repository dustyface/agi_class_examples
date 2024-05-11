""" Test langchain data connection """
import timeit
import logging
from functools import wraps
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

logger = logging.getLogger(__name__)


def timer(target_func):
    """ timeit decorator definition """
    @wraps(target_func)
    def func_wrapper(*args, use_timer=False, **kwargs):
        if use_timer is True:
            start_time = timeit.default_timer()
            target_func(*args, **kwargs)
            end_time = timeit.default_timer()
            run_time = end_time - start_time
            log_msg = "function {name} executed in {run_time} seconds".format(**{
                "name": f"{func_wrapper.__name__!r}", "run_time": run_time})
            logger.info(log_msg)
        else:
            target_func(*args, **kwargs)

    return func_wrapper


class DocumentHandler:
    """ Langchain PDF DocumentLoader & Text Splitter & Vectordb query """

    def __init__(self, file_path: str):
        self.doc_loader = PyPDFLoader(file_path)
        self.doc_list: List[Document] = self.doc_loader.load_and_split()
        self.embeddings = OpenAIEmbeddings()
        self.db = None
        self.text_splitter = None

    def _create_text_splitter(
        self, chunk_size=200, chunk_overlap=100,
        length_function=len, add_start_index=True
    ):
        """ create text_splitter instance """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            add_start_index=add_start_index
        )

    def _text_split(self, doc_list: List[Document]):
        """ create the paragraphs from a page """
        page_content_list = [doc.page_content for doc in doc_list]
        doc_list = self.text_splitter.create_documents(page_content_list)
        return doc_list

    @timer
    def bulk_document(self):
        """ bulk vector db """
        self._create_text_splitter()
        paras: List[Document] = self._text_split(self.doc_list)
        print(f"injecting document list...{len(paras) = !r}")
        self.db = Chroma.from_documents(
            paras, self.embeddings, persist_directory="vectordb/chroma/")

    @timer
    def retrieve(self, search_text, k=1):
        """ retrieve the  search text related doc """
        print("search_text=", search_text)
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(search_text)
        logger.info(docs)
        return docs
