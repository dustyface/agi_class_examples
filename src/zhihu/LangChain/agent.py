""" the 2 example of agent from AGI class """
import logging
import calendar
import re
import dateutil.parser as parser
from dotenv import load_dotenv, find_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.tools import Tool, tool
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    create_self_ask_with_search_agent,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper


load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


def get_react_template():
    """ check react template """
    prompt = hub.pull("hwchase17/react")
    print(prompt.pretty_print())


@tool("weekday")
def weekday(date_str: str) -> str:
    """ Convert date to weekday name """
    def fix_possible_err():
        return re.sub(r"[\n\"\']", "", date_str)

    logger.info("date_str=%s", date_str)
    try:
        d = parser.parse(date_str)
        return calendar.day_name[d.weekday()]
    except ValueError:
        date_str = fix_possible_err()
        d = parser.parse(date_str)
        return calendar.day_name[d.weekday()]


def agent_react_with_search(query: str):
    """ create agen with search """
    search = SerpAPIWrapper()
    tools = [
        Tool.from_function(
            func=search.run,
            name="search",
            description="useful for when you need to answer questions about current events"
        ),
        weekday
    ]
    # 选用能力强的model非常重要, 针对用户输入的prompt, 不同的model理解力不同, prompt map Function Calling的能力不同
    # e.g. 本例prompt "2024年周杰伦的演唱会是星期几?", gpt-3.5-turbo在很多次无法理解查询步骤的次序;
    model = "gpt-4o"
    # model = "gpt-3.5-turbo"
    model = ChatOpenAI(model=model, temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True)
    result = ""
    for chunk in agent_executor.stream({"input": query}):
        print(chunk)
        if "output" in chunk:
            result = chunk["output"]
    return result


class WeekDay(BaseModel):
    """ Pydantic model """
    date: str = Field(description="the weekday name of a date")


def with_structured_result(result: str):
    """ parse the result to JSON structure"""
    output_parser = PydanticOutputParser(pydantic_object=WeekDay)
    template = """
    Abstract information from the user query base on the format instruction below.
    {format_instructions}

    {query}
    """
    model = ChatOpenAI()
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        }
    )
    chain = prompt | model | output_parser
    res = chain.invoke({"query": result})
    logger.info("result=%s", res)


def agent_selfask_with_search():
    """ create agent self-ask with search """
    prompt = hub.pull("hwchase17/self-ask-with-search")
    logger.info("prompt=%s", prompt.pretty_print())
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search"
        )
    ]
    model = ChatOpenAI(model="gpt-4o")
    agent = create_self_ask_with_search_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_error=True)
    for chunk in agent_executor.stream({"input": "冯小刚的老婆演过哪些电影?"}):
        print(chunk)
