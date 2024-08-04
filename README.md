
# Zhihu AGIClass Campaign Examples

- 这个project是知乎AI训练营的examples。对课件中的例子进行了重构, 全部改为基于pytest unittest的方式进行测试

## 运行程序

```
pytest tests/zhihu/test_api.py
pytest tests/zhihu/test_api.py::test_model_list
```

# .Env 设定

```
PYTHONPATH="src"
LOG_LEVEL="INFO"    # INFO, DEBUG, etc.
OPENAI_API_KEY="your openai key"
OPENAI_BASE_URL="your openai proxy"
AMAP_KEY="xxx"     # API Key for 高德地图, you can get public one from zhihu
ZHIHU_ELASTICSEARCH_URL = "xxx"  # Zhihu Elastic Host URL
ZHIHU_ELASTICSEARCH_PWD = "xxx"    # Zhihu Elastic Host PWD
MODEL_CACHED_DIR="path_to_store_your_local_model"
ERNIE_CLIENT_ID="BaiDu Qianfan AK"
ERNIE_CLIENT_SECRET="BaiDu Qianfan SK"
QIANFAN_AK="BaiDu QianFan AK"
QIANFAN_SK="BaiDu QianFan SK"
QIANFAN_ACCESS_KEY="BaiDu QianFan Access Key"
QIANFAN_SECRET_KEY="BaiDu QianFan Access Key"
LANGCHAIN_API_KEY="LangChain API Key"
LANGCHAIN_TRACING_V2="true"
TAVILY_API_KEY="Tavily API Key"
SERPAPI_API_KEY= "Google Search API Key"
ZHIPUAI_API_KEY="360 Zhipu API Key"
LANGFUSE_SECRET_KEY="LangFuse SK"
LANGFUSE_PUBLIC_KEY="LangFuse PK"
LANGFUSE_HOST="LangFuse host url"   # https://cloud.langfuse.com

```