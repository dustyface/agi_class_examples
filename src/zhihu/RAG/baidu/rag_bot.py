""" Baidu RAG Bot """
from zhihu.RAG.vectordb.rag_bot import RAGBot
from zhihu.common.api_ernie import Session
from zhihu.RAG.prompt import build_prompt


class ERNIERAGBot(RAGBot):
    """ RAG Bot for ERNIE """

    def __init__(self, bot_name: str, *, _prompt_template=None):
        super().__init__(bot_name, _prompt_template=_prompt_template)
        self.session = Session()

    def chat(self, user_query: str, *, top_n=5):
        result = self.search(user_query, top_n=top_n)
        result_list = [s for s in result["documents"][0]]
        prompt = build_prompt(
            self.prompt_template,
            info=result_list,
            query=user_query)
        print("user_query=", user_query, "\nprompt=", prompt)
        return self.session.chat(prompt, model="ERNIE-4.0-8K")
