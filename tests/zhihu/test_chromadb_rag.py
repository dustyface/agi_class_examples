""" Test the RAG bot pipelien of using chromadb """
from zhihu.RAG.vectordb.rag_bot import RAGBot, RAGBotWithCascadeDocParser
from zhihu.RAG.baidu.rag_bot import ERNIERAGBot

PDF_FILE = "src/zhihu/RAG/llama2-test-1-4.pdf"


def test_chromadb_search():
    """ Test the search function of RAGBot """
    bot = RAGBot("chromadb_test")
    bot.prepare(PDF_FILE)
    search_topic = "how many parameters does llama 2 have?"
    bot.search(search_topic, top_n=5)


def test_chromadb_rag():
    """ Test the RAGBot with ChromaDB """
    user_query = [
        # "how many parameters does llama 2 have?",
        # "can llama 2 be used for comercial purpose? ",  # 这个英文问题，却可以回答出正确答案~
        "Llama 2 有可商用的版本吗?",  # OpenAI 基于基本的切分段落, 没有有效的回答这个问题
        # "how safe is llama 2?",
        # "Llama 2 有对话的版本吗?"
    ]
    bot = RAGBot("chromadb_test")
    bot.prepare(PDF_FILE)
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp.choices[0].message.content)


def test_chromadb_rag_ernie():
    """ Test the RAGBot with ERNIE """
    bot = ERNIERAGBot("chromadb_test")
    bot.prepare(PDF_FILE)
    user_query = [
        # "how many parameters does llama 2 have?",
        # "can llama 2 be used for comercial purpose? ",
        "Llama 2 有可商用的版本吗?",  # 在没有对段落切分采用重叠切分的时候, ernie-4回答此问题是错误的
        # "how safe is llama 2?",
        # "Llama 2 有对话的版本吗?"
    ]
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp["body"]["result"])


def test_chromadb_rag_with_cascade():
    """ Test the RAGBot with CascadeDocParser """
    bot = RAGBotWithCascadeDocParser("chromadb_test")
    bot.prepare(PDF_FILE)
    user_query = [
        "Llama 2 有可商用的版本吗?",  # OpenAI针对使用了Cascade_split_text切分doc的RAG，可以回答
    ]
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp.choices[0].message.content)
