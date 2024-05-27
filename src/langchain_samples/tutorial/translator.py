""" A application translating one language to another languge """
import sys
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


def simple_translate():
    """ A simple translator """
    chat_model = ChatOpenAI()
    messages = [
        SystemMessage(
            content="Translate the following from English into Italian"),
        HumanMessage(content="hi"),
    ]
    res = chat_model.invoke(messages)
    parser = StrOutputParser()
    return parser.invoke(res)


def simple_translate_with_chain():
    """ A simple translator with chain """
    model = ChatOpenAI()
    parser = StrOutputParser()
    system_template = "Translate the following into {language}: "
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template), ("user", "{text}")
    ])
    chain = prompt_template | model | parser
    return chain
    # res = chain.invoke({
    #     "language": "Italian",
    #     "text": "hi",
    # })
    # logger.info("tranlator result=%s", res)


li = ['hello', 'world']
li.insert(0, "haha")
print(sys.path)
