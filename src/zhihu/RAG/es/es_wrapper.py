""" Elasticsearch Wrapper """
import os
import warnings
import logging
from dotenv import load_dotenv, find_dotenv
from elasticsearch7 import Elasticsearch, helpers
from zhihu.RAG.common import to_keywrod_en

logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())

ES_HOST = f"http://{os.getenv('ZHIHU_ELASTICSEARCH_URL')}"
ES_PWD = os.getenv("ZHIHU_ELASTICSEARCH_PWD")

warnings.simplefilter("ignore")     # 屏蔽 ES 的一些Warnings


class ESWrapper:
    """ Elasticsearch Wrapper """

    def __init__(self, *, to_keyword=to_keywrod_en, es_host=ES_HOST, es_pwd=ES_PWD):
        self.es_host = es_host
        self.es_pwd = es_pwd
        self.es = Elasticsearch(
            hosts=[self.es_host], http_auth=("elastic", self.es_pwd))
        self.index_name = None
        self.to_keyword = to_keyword

    def list_all_indices(self):
        """ List all ES indices """
        return self.es.indices.get_alias("*")

    def delete_indice(self, index_name):
        """ Delete an ES index """
        self.es.indices.delete(index=index_name)

    def create_indice(self, index_name):
        """ Create an ES index """
        if self.es.indices.exists(index=index_name):
            self.delete_indice(index_name)
        self.es.indices.create(index=index_name)
        self.index_name = index_name

    def bulk_es(self, paragraphs):
        """ Bulk insert paragraphs into ES """
        if self.index_name is None:
            raise ValueError(
                "bulk_es(): index_name is not set; you should call create_indice() first.")
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
        # important note: 有关bulk refresh参数
        # https://www.elastic.co/guide/en/elasticsearch/reference/master/docs-refresh.html
        # 必须要使用refresh参数, 否则立即进行es search操作得不到搜索的数据
        helpers.bulk(self.es, actions, refresh='wait_for')

    def search(self, query_string, *, top_n=3):
        """ Search the ES index """
        search_query = {
            "match": {
                "keywords": self.to_keyword(query_string)
            }
        }
        logger.debug("search query=%s", search_query)
        # self.es.search的可接受参数是通过decorator定义的，但pylint无法检测到, 似乎是一个pylint的bug
        # pylint: disable=unexpected-keyword-arg
        res = self.es.search(index=self.index_name,
                             query=search_query, size=top_n)
        return [hit for hit in res["hits"]["hits"]]
