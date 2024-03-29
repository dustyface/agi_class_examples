""" RAGBot class for the RAG pipeline with ElasticSearch"""
import logging
from zhihu.RAG.common import parse_paragraph_from_pdf, parse_paragraph_from_pdf_v2
from zhihu.RAG.es.es_wrapper import ESWrapper
from zhihu.RAG.prompt import build_prompt, PROMPT_TEMPLATE
from zhihu.common.api import Session

logger = logging.getLogger(__name__)

# see test case:
# tests/zhihu/test_es_rag.py::test_es_rag_bot()
# tests/zhihu/test_es_rag.py::test_es_search()


class RAGBot:
    """ The RAG Pipeline Class with ElasticSearch """
    prompt_template = PROMPT_TEMPLATE

    def __init__(self, *,  index_name="student_test_rag", prompt_template=None):
        self.es = ESWrapper()
        self.es.create_indice(index_name)
        if prompt_template is not None:
            self.prompt_template = prompt_template
        self.session = Session()

     # 直接用pdfminer.six的LTTextContainer作为paragraph
    def _parse_document(self, in_file, *, page_numbers=None):
        return parse_paragraph_from_pdf(in_file, page_numbers=page_numbers)

    # 知乎课件的使用last_line_length来分段
    def _parse_document_v2(self, in_file, *, page_numbers=None):
        return parse_paragraph_from_pdf_v2(in_file, page_numbers=page_numbers)

    def _bulk_es(self, paragraphs):
        self.es.bulk_es(paragraphs)

    def search(self, query_string, *, top_n=3) -> list:
        """ Search the ElasticSearch """
        hits = self.es.search(query_string, top_n=top_n)
        print("search result:")
        for h in hits:
            print(f"{h['_score']}\t{h['_source']['text'][0:70]}")
        return [h["_source"]["text"] for h in hits]

    def _build_prompt(self, prompt, **kwargs):
        return build_prompt(prompt, info=kwargs["info"], query=kwargs["query"])

    def prepare(self, in_file):
        """ Load and parse the document, then bulk insert into ES """
        paras = self._parse_document(in_file)
        self._bulk_es(paras)

    def chat(self, user_query, *, top_n=5):
        """ Chat with the OpenAI LLM """
        result_list = self.search(user_query, top_n=top_n)
        prompt = self._build_prompt(
            self.prompt_template,
            info=result_list,
            query=user_query)
        return self.session.get_completion(prompt)
