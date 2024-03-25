from zhihu.RAG.vectordb.rag_bot import RAG_Bot

pdf_file = "src/zhihu/RAG/llama2-test-1-4.pdf"

def test_chromadb_search():
    bot = RAG_Bot("chromadb_test")
    bot.prepare(pdf_file)
    search_topic = "how many parameters does llama 2 have?"
    bot._search(search_topic, top_n=5)

def test_chromadb_rag():
    user_query = [
        "how many parameters does llama 2 have?",
        "can llama 2 be used for comercial purpose? ",
        "Llama 2 有可商用的版本吗?",
        "how safe is llama 2?",
        "Llama 2 有对话的版本吗?"
    ]
    bot = RAG_Bot("chromadb_test")
    bot.prepare(pdf_file)
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp.choices[0].message.content)
