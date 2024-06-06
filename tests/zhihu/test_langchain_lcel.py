""" Test LCEL """
import logging
from zhihu.LangChain.lcel import (
    check_chain_schema, search_rag_chain2, search_rag_chain,
    search_semantcs_package, rag_lcel
)
from zhihu.LangChain.lcel_func_call import llm_bind_tools, llm_exec_tools
from zhihu.LangChain.lcel_factory import lcel_with_config

logger = logging.getLogger(__name__)


def test_schema():
    """ test chain runnable's input_schema / output_schema """
    check_chain_schema()


def test_search_rag_chain():
    """
    test search rag chain
    pytest tests/zhihu/test_langchain_lcel.py::test_search_rag_chain
    """
    logger.info(search_rag_chain())
    logger.info(search_rag_chain2())


def test_search_semantcs_package():
    """
    test search semantcs package
    pytest tests/zhihu/test_langchain_lcel.py::test_search_semantcs_package
    """
    search_semantcs_package()


def test_rag_lcel():
    """
    test rag lcel
    pytest tests/zhihu/test_langchain_lcel.py::test_rag_lcel
    """
    logger.info(rag_lcel())


def test_bind_tools():
    """
    test bind tools
    pytest tests/zhihu/test_langchain_lcel.py::test_bind_tools
    """
    llm_bind_tools()


def test_exec_tools():
    """
    test exec bind tools,
    let the chain execute the calling definition returned by former Runnable
    pytest tests/zhihu/test_langchain_lcel.py::test_exec_tools
    """
    logger.info(llm_exec_tools("1024的16倍是多少?"))
    logger.info(llm_exec_tools("1024加上16等于多少"))


def test_lcel_with_config():
    """
    test lcel with config
    pytest tests/zhihu/test_langchain_lcel.py::test_lcel_with_config
    """
    logger.info(lcel_with_config("介绍你自己, 包括你的生产商?"))
    logger.info(lcel_with_config("介绍你自己, 包括你的生产商?", llm_name="ernie"))
    logger.info(lcel_with_config("介绍你自己, 包括你的生产商?", llm_name="gpt"))
