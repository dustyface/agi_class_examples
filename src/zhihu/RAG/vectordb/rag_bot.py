from typing import Union
from zhihu.RAG.vectordb.chrome_wrapper import ChromaDBWrapper
from zhihu.RAG.vectordb.embeddings import get_embeddings
from zhihu.common.api import Session
from zhihu.RAG.common import parse_paragraph_from_pdf
from zhihu.RAG.prompt import build_prompt, prompt_template
from zhihu.RAG.common import cascade_split_text


class RAG_Bot:
    prompt_template = prompt_template

    def __init__(self, bot_name:str, *, prompt_template=None):
        self.chromadb_wrapper = ChromaDBWrapper(bot_name, get_embeddings)
        if prompt_template is not None:
            self.propmt_template = prompt_template
        self.session = Session()

    def _parse_document(self, in_file, *, page_numbers=None):
        return parse_paragraph_from_pdf(in_file, page_numbers=page_numbers)

    def prepare(self, in_file):
        paras = self._parse_document(in_file)
        self.chromadb_wrapper.add_documenet(paras)

    # chromadb的search()返回的dict结构如下：
    # { ids: [[], []..., []], distances: [[0.289, 0.31xxx, ...],[],...[]]}, "embeddings": None, "documents": [["", "", ..., ""], []]
    def _search(self, query: Union[str, list[str]], *, top_n=3):
        return self.chromadb_wrapper.search(query, top_n=top_n)   
       
    def chat(self, user_query:str, *, top_n=5):
        result = self._search(user_query, top_n=top_n)
        result_list = [s for s in result["documents"][0]]
        prompt = build_prompt(
            self.prompt_template,
            info=result_list,
            query=user_query)
        print("user_query=", user_query, "\nprompt=", prompt)
        return self.session.get_completion(prompt)

class RAG_Bot_with_CascadeDocParser(RAG_Bot):
    def _parse_document(self, in_file, *, page_numbers=None):
        paras = super()._parse_document(in_file, page_numbers=page_numbers)
        return cascade_split_text(paras)
    
    