from zhihu.RAG.vectordb.rag_bot import RAG_Bot, RAG_Bot_with_CascadeDocParser
from zhihu.RAG.baidu.rag_bot import ERNIE_RAG_Bot

pdf_file = "src/zhihu/RAG/llama2-test-1-4.pdf"

def test_chromadb_search():
    bot = RAG_Bot("chromadb_test")
    bot.prepare(pdf_file)
    search_topic = "how many parameters does llama 2 have?"
    bot._search(search_topic, top_n=5)

def test_chromadb_rag():
    user_query = [
        # "how many parameters does llama 2 have?",
        # "can llama 2 be used for comercial purpose? ",  # 这个英文问题，却可以回答出正确答案~
        "Llama 2 有可商用的版本吗?",  # OpenAI 基于基本的切分段落, 没有有效的回答这个问题
        # "how safe is llama 2?",
        # "Llama 2 有对话的版本吗?"
    ]
    bot = RAG_Bot("chromadb_test")
    bot.prepare(pdf_file)
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp.choices[0].message.content)

def test_chromadb_rag_ernie():
    bot = ERNIE_RAG_Bot("chromadb_test")
    bot.prepare(pdf_file)
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
    bot = RAG_Bot_with_CascadeDocParser("chromadb_test")
    bot.prepare(pdf_file)
    user_query = [
        "Llama 2 有可商用的版本吗?",  # OpenAI针对使用了Cascade_split_text切分doc的RAG，可以回答
    ]
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp.choices[0].message.content)