""" Test embedding service """
import logging
from zhihu.RAG.vectordb.embeddings import get_embeddings, cos_sim, l2
from zhihu.RAG.baidu.embeddings import get_embeddings as get_embeddings_ernie

logger = logging.getLogger(__name__)

texts = [
    "测试文本",
    "北京的天气怎么样?",
    "上海的生煎特别好吃, 我很久没有吃到地道的上海生煎了"
]


def test_embeddings():
    """ Test embedding model for OpenAI """
    vectors = get_embeddings(texts)
    # 可以看到，不同长度的文本，返回的embedding的长度是一样的，model text-embedding-ada-002的维度是1536
    for embedding in vectors:
        logger.info("embedding=%s, len(embedding)=%s",
                    embedding[0:5], len(embedding))


def test_comparison_embeddings():
    """ Compare the embdding's distance """
    text = "Global conflicts"
    documents = [
        "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
        "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
        "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
        "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
        "我国首次在空间站开展舱外辐射生物学暴露实验",
    ]
    vectors = get_embeddings([text] + documents)
    logger.info("check cos_sim:")
    for i, embedding in enumerate(vectors):
        cos_sim_distance = cos_sim(vectors[0], embedding)
        logger.info("cos_sim(text, vectors[%s])=%s", i, cos_sim_distance)
    logger.info("check l2:")
    for i, embedding in enumerate(vectors):
        l2_dist = l2(vectors[0], embedding)
        logger.info("l2_dist(text, vectors[%s])=%s", i, l2_dist)


def test_comparison_embeddings_v2():
    """ 
    Test the embeddings for RAG example 6,
    matching the large, specifical vocab
    """
    query = "非小细胞肺癌的患者"
    documents = [
        "李某患有肺癌，癌细胞已转移",       # 0.9104978086098463
        "刘某肺癌I期",                   # 0.8897648918974235
        "张某经诊断为非小细胞肺癌III期",    # 0.9040803406710728, 这个应该最匹配的，但向量搜索却没有得分最高
        "小细胞肺癌是肺癌的一种"           # 0.9132102982983269
    ]
    vectors = get_embeddings([query] + documents)
    for i, embedding in enumerate(vectors):
        cos_sim_distance = cos_sim(vectors[0], embedding)
        logger.info("cos_sim(text, vectors[%s])=%s", i, cos_sim_distance)


def test_get_embeddings_ernie():
    """ Test embeddings for Baidu ERNIE """
    vectors = get_embeddings_ernie(texts)
    for embedding in vectors:
        logger.info("embedding=%s, len(embedding)=%s",
                    embedding[0:5], len(embedding))
