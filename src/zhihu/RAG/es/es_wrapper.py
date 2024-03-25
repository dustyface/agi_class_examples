from dotenv import load_dotenv, find_dotenv
from elasticsearch7 import Elasticsearch, helpers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import os
import warnings
import logging
logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())

ES_HOST = f"http://{os.getenv('ZHIHU_ELASTICSEARCH_URL')}"
ES_PWD = os.getenv("ZHIHU_ELASTICSEARCH_PWD")

warnings.simplefilter("ignore")     # 屏蔽 ES 的一些Warnings
nltk.download("punkt")              # 英文切词、词根、切句等方法
nltk.download("stopwords")          # 英文停用词库

class ESWrapper:
    def __init__(self, *, es_host=ES_HOST, es_pwd=ES_PWD):
        self.es_host = es_host
        self.es_pwd = es_pwd
        self.es = Elasticsearch(hosts=[self.es_host], http_auth=("elastic", self.es_pwd))

    def list_all_indices(self):
        return self.es.indices.get_alias("*")

    def delete_indice(self, index_name):
        self.es.indices.delete(index=index_name, ignore=[400, 404])
    
    def create_indice(self, index_name):
        if self.es.indices.exists(index=index_name):
            self.delete_indice(index_name)
        self.es.indices.create(index=index_name)
        self.index_name = index_name
    
    def bulk_es(self, paragraphs):
        if self.index_name is None:
            raise ValueError("bulk_es(): index_name is not set; you should call create_indice() first.")
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "keywords": self.to_keyword(para),
                    "text": para
                },
            }
            for para in paragraphs
        ]
        # 注意: 有关bulk refresh参数
        # https://www.elastic.co/guide/en/elasticsearch/reference/master/docs-refresh.html
        # 必须要使用refresh参数, 否则立即进行es search操作得不到搜索的数据
        helpers.bulk(self.es, actions, refresh='wait_for')
    
    def to_keyword(self, input_string):
        # 使用正则表达式替换所有非字母数字的字符为空格
        no_symblos = re.sub(r"[^a-zA-Z0-9\s]", " ", input_string)
        word_tokens = word_tokenize(no_symblos)
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        # 去停用词，取词根
        filtered_sentence = [ps.stem(w) for w in word_tokens if not w.lower() in stop_words]
        return " ".join(filtered_sentence)
    
    def search(self, query_string, *, top_n=3):
        search_query = {
            "match": {
                "keywords": self.to_keyword(query_string)
            }
        }
        res = self.es.search(index=self.index_name, query=search_query, size=top_n)
        return [hit for hit in res["hits"]["hits"]]
