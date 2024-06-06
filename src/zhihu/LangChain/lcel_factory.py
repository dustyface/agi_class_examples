""" Test choose lcel factory """
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables.utils import ConfigurableField
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv(find_dotenv())


def lcel_with_config(query: str, llm_name: str = "gpt"):
    """ lcel with config """
    ernie_model = QianfanChatEndpoint()
    gpt_model = ChatOpenAI()
    model = gpt_model.configurable_alternatives(
        ConfigurableField(id="llm"),
        default_key="gpt",
        ernie=ernie_model,
    )
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{query}")
    ])
    chain = ({
        "query": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    ret = chain.with_config(configurable={"llm": llm_name}).invoke(query)
    return ret
