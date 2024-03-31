""" Test the RAG bot pipelien of using chromadb """
import logging
from zhihu.RAG.vectordb.rag_bot import RAGBot, RAGBotWithCascadeDocParser, RAGBotwithCrossEncoder
from zhihu.RAG.common import rrf
from zhihu.RAG.baidu.rag_bot import ERNIERAGBot
from tests.zhihu.test_es_rag import test_es_search_parse_result

logger = logging.getLogger(__name__)
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
        "Llama 2 有可商用的版本吗?",  # OpenAI针对使用了Cascade_split_text切分doc的RAG，可以回答正确答案
    ]
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp.choices[0].message.content)


def test_chromadb_rag_with_crossencoder():
    """ Test the RAGBot with CrossEncoder """
    bot = RAGBotwithCrossEncoder("chromadb_test")
    bot.prepare(PDF_FILE)
    user_query = [
        "how safe is llama 2"
    ]
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        print(rsp.choices[0].message.content)


def test_chromadb_search_parse_result():
    """ Test the RAGBot search with customized parse callback output """
    documents = [
        "李某患有肺癌，癌细胞已转移",
        "刘某肺癌I期",
        "张某经诊断为非小细胞肺癌III期",
        "小细胞肺癌是肺癌的一种"
    ]

    def parse_search_result(result):
        output_result = list()
        for i, (_id, dis, doc) in enumerate(
                zip(result["ids"][0], result["distances"][0], result["documents"][0])):
            print(f"{_id}\t{dis}\t{doc}")
            output_result.append({
                f"id_{_id}": {
                    "text": doc,
                    "rank": i
                }
            })
        logger.info("output_result=%s", output_result)
        return output_result

    bot = RAGBot("chromadb_with_parse_search_result")
    bot.prepare(docs=documents)
    search_topic = "非小细胞肺癌的患者"
    result = bot.search(search_topic, top_n=5)
    logger.info("result=%s", result)
    return parse_search_result(result)


def test_hybrid_search_rrf():
    """ Test the hybrid search RRF algorithm """
    es_result = test_es_search_parse_result()
    chroma_result = test_chromadb_search_parse_result()
    # es_result = [
    #     {'hRk-lI4BXQtMEZJf-fbp': {'text': '张某经诊断为非小细胞肺癌III期', 'rank': 0}},
    #     {'gxk-lI4BXQtMEZJf-fbp': {'text': '李某患有肺癌，癌细胞已转移', 'rank': 1}},
    #     {'hhk-lI4BXQtMEZJf-fbp': {'text': '小细胞肺癌是肺癌的一种', 'rank': 2}},
    #     {'hBk-lI4BXQtMEZJf-fbp': {'text': '刘某肺癌I期', 'rank': 3}}
    # ]
    # chroma_result = [
    #     {'id_id3': {'text': '小细胞肺癌是肺癌的一种', 'rank': 0}},
    #     {'id_id0': {'text': '李某患有肺癌，癌细胞已转移', 'rank': 1}},
    #     {'id_id2': {'text': '张某经诊断为非小细胞肺癌III期', 'rank': 2}},
    #     {'id_id1': {'text': '刘某肺癌I期', 'rank': 3}}
    # ]
    rrf_result = rrf(es_result + chroma_result)
    # json.dumps会导致中文被输出成unicode
    # logger.info("rrf_result=%s", json.dumps(rrf_result))
    logger.info("rrf_result=%s", rrf_result)
