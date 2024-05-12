
""" Test LCEL """
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_models import QianfanChatEndpoint

logger = logging.getLogger(__name__)
chat_model = QianfanChatEndpoint()


def simple_chain():
    """ make a simple chain """
    prompt = ChatPromptTemplate.from_template(
        "tell me a joke about {subject}")
    chain = prompt | chat_model
    return chain


def check_chain_schema():
    """ check input schema """
    # chain的input schema是chain的第一个runnable element - prompt
    # 的input_schema
    # schema()是input_schema/output_schema pydantic model的JSON表示形式
    # see: https://python.langchain.com/v0.1/docs/expression_language/interface/#input-schema
    chain = simple_chain()

    def get_schema_keys(schema_json):
        return [
            key for i, (key, value) in enumerate(schema_json.items()) if not key == "properties"
        ]

    prompt_schema = chain.input_schema.schema()
    keys = get_schema_keys(prompt_schema)
    logger.info("prompt schema(): %s", keys)

    chat_model_schema = chat_model.input_schema.schema()
    keys = get_schema_keys(chat_model_schema)
    logger.info("qianfan model schema(): %s", keys)

    chat_model_output_schema = chain.output_schema.schema()
    keys = get_schema_keys(chat_model_output_schema)
    logger.info("chain.output_schema.schema(): %s", keys)
