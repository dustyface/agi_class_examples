
""" Test RAG pipeline using ES """
import logging
from zhihu.RAG.common import parse_paragraph_from_pdf, parse_paragraph_from_pdf_v2, to_keyword_cn
from zhihu.RAG.es.rag_bot import RAGBot
from zhihu.RAG.es.es_wrapper import ESWrapper
from zhihu.common.util import write_log_file

logger = logging.getLogger(__name__)

PDF_FILE = "src/zhihu/RAG/llama2-test-1-4.pdf"


def test_parse_doc_v1():
    """ Test parsing the pdf """
    paras = parse_paragraph_from_pdf(PDF_FILE, page_numbers=[0, 1, 2])
    for p in paras[-4:-1]:
        print("\n")
        print(p)
    logger.info("writing log file %s", "llama2_test")
    write_log_file("llama2_test", paras)


# 这两个方法比较，可以看出硬编码parse_paragraph_from_pdf_v2输出的分段结果是不准确的
def test_parse_doc_v2():
    """ Test parsing the pdf using 2nd method """
    paras = parse_paragraph_from_pdf_v2(PDF_FILE, page_numbers=[0, 1, 2])
    for p in paras[-4:-1]:
        print("\n")
        print(p)
    write_log_file("llama2_test_v2", paras)


def test_es_list_indices():
    """ Listing ES indices for zhihu service """
    eswrapper = ESWrapper()
    all_indices = eswrapper.list_all_indices()
    for k, v in all_indices.items():
        print(k, v)


def test_es_delete_unused_indices():
    """ Test delete the zhihu ES service indices """
    eswrapper = ESWrapper()
    eswrapper.delete_indice("student*")


def test_es_search():
    """ Test search funtionality of ES """
    bot = RAGBot()
    bot.prepare(PDF_FILE)
    search_topic = "how many parameters does llama 2 have?"
    bot.search(search_topic, top_n=5)


def test_es_search_parse_result():
    """ Test ES search and parse result as needed """
    def parse_search_result(hits):
        result = [
            {
                h["_id"]: {
                    "text": h["_source"]["text"],
                    "rank": i
                }
            }
            for i, h in enumerate(hits)
        ]
        print("result=", result)
        for res in result:
            item = list(res.items())[0]
            k = item[0]
            v = item[1]
            print(f"{k}\t{v['rank']}\t{v['text']}")
        return result

    documents = [
        "李某患有肺癌，癌细胞已转移",
        "刘某肺癌I期",
        "张某经诊断为非小细胞肺癌III期",
        "小细胞肺癌是肺癌的一种"
    ]
    bot = RAGBot(to_keyword=to_keyword_cn)
    bot.prepare(docs=documents)
    search_topic = "非小细胞肺癌的患者"
    return bot.search(
        search_topic,
        top_n=5,
        parse_search_result=parse_search_result
    )


def test_es_rag_bot():
    """ Test RAG bot pipeline for ES """
    user_query = [
        "how many parameters does llama 2 have?",
        "can llama 2 be used for comercial purpose? "
    ]
    bot = RAGBot()
    bot.prepare(PDF_FILE)
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        logger.info(rsp.choices[0].message.content)
