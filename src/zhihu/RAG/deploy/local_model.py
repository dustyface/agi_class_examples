""" 本地模型部署 """
import logging
import os
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

_ = load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)

MODEL_CACHED_DIR = os.getenv("MODEL_CACHED_DIR")
if MODEL_CACHED_DIR is None:
    raise ValueError("请配置环境变量 MODEL_CACHED_DIR")
logger.info("MODEL_CACHED_DIR=%s", MODEL_CACHED_DIR)

# 可以支持中英文
EMBEDDING_MODEL = "moka-ai/m3e-base"
# 可以支持中文
# EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHED_DIR)

QUERY = "Global conflicts"
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = model.encode(QUERY, normalize_embeddings=True)
doc_vec = [
    model.encode(doc, normalize_embeddings=True) for doc in documents
]
