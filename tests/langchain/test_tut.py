""" Test the cases in LangChain tutorials """
import logging
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_samples.tutorial.translator import simple_translate_with_chain
from langchain_samples.tutorial.chatbot import chatbot_chat, limited_length_chat
from langchain_samples.tutorial.agent import stream_tokens
from langchain_samples.tutorial.vectorstore import retriever1 as retriever, rag_chain

logger = logging.getLogger(__name__)


def test_simple_traslator_with_chain():
    """ test simple translator """
    simple_translate_with_chain()


def test_chatbot():
    """
    test chatbot with message history
    pytest tests/LangChain/test_tut.py::test_chatbot
    """
    chatbot_chat("qst")


def test_limit_length_chat():
    """ pytest tests/LangChain/test_tut.py::test_limit_length_chat """
    # filter_message设定最新10条, 超过10条messages, 仍然不记得;
    messages = [
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]
    limited_length_chat("abc", messages)

# https://stackoverflow.com/questions/55893235/pytest-skips-test-saying-asyncio-not-installed


@pytest.mark.asyncio
async def test_agent_stream_tokens():
    """
    pytest tests/LangChain/test_tut.py::test_agent_stream_tokens
    """
    await stream_tokens()


def test_rag_chain():
    """
    pytest tests/LangChain/test_tut.py::test_rag_chain
    """
    logger.info(rag_chain(retriever).content)
