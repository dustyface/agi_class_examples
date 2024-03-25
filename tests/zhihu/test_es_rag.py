
from zhihu.RAG.common import parse_paragraph_from_pdf, parse_paragraph_from_pdf_v2
from zhihu.RAG.es.rag_bot import RAG_Bot
from zhihu.RAG.es.es_wrapper import ESWrapper
import logging
logger = logging.getLogger(__name__)

pdf_file = "src/zhihu/RAG/llama2-test-1-4.pdf"

def test_parse_doc_v1():
    paras = parse_paragraph_from_pdf(pdf_file, page_numbers=[0, 1, 2])
    for p in paras[-4:-1]:
        print("\n")
        print(p)

# 这两个方法比较，可以看出硬编码parse_paragraph_from_pdf_v2输出的分段结果是不准确的
def test_parse_doc_v2():
    paras = parse_paragraph_from_pdf_v2(pdf_file, page_numbers=[0, 1, 2])
    for p in paras[-4:-1]:
        print("\n")
        print(p)


def test_es_list_indices():
    eswrapper = ESWrapper()
    all_indices = eswrapper.list_all_indices()
    for k,v in all_indices.items():
        print(k, v)

def test_es_delete_unused_indices():
    eswrapper = ESWrapper()
    eswrapper.delete_indice("student*")

def test_es_search():
    bot = RAG_Bot()
    bot.prepare(pdf_file)
    search_topic = "how many parameters does llama 2 have?"
    bot._search(search_topic, top_n=5)


def test_es_rag_bot():
    user_query = [
        "how many parameters does llama 2 have?",
        "can llama 2 be used for comercial purpose? "
    ]
    bot = RAG_Bot()
    bot.prepare(pdf_file)
    for q in user_query:
        rsp = bot.chat(q, top_n=5)
        logger.info(rsp.choices[0].message.content)





    

