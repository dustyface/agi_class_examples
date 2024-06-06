""" A testing chatbot app """
import logging
from typing import List
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

_ = load_dotenv(find_dotenv())
logger = logging.getLogger("chatbot")
message_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """ get history for session """
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


def chatbot_chat(session_id="xyz"):
    """ a chatbot with message history """
    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer all questions \
            to the best of your ability in {language}"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | model
    config = {
        "configurable": {
            "session_id": session_id
        }
    }
    with_history_runnable = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages")

    response = with_history_runnable.invoke({
        "messages": [HumanMessage(content="hi, I'm todd")],
        "language": "English",
    }, config=config)
    logger.info("response for 1st question=%s", response.content)

    response = with_history_runnable.invoke({
        "messages": [HumanMessage(content="what's my name?")],
        "language": "English",
    }, config=config)
    logger.info("response for 2nd question=%s", response.content)


def filter_message(messages, k=10):
    """ fetch the last 10 messags """
    return messages[-k:]

# pylint: disable=dangerous-default-value


def limited_length_chat(session_id: str = "xyz", messages: List[BaseMessage] = []):
    """ use RunnablePassthrough """
    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer all questions \
            to the best of your ability in {language}"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = (
        RunnablePassthrough.assign(
            messages=lambda x: filter_message(x["messages"]))
        | prompt
        | model
    )
    config = {
        "configurable": {
            "session_id": session_id,
        }
    }
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )
    res = runnable_with_history.invoke({
        "messages": messages + [HumanMessage(content="what is my name")],
        "language": "English",
    }, config=config)
    logger.info("res=%s", res.content)
