""" Test Custimzed agent """
import logging
# from langchain.globals import set_debug
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# set_debug(True)
logger = logging.getLogger(__name__)


@tool
def get_word_length(word: str) -> int:
    """ get the length of a word """
    return len(word)


async def test_custom_agent(query: str):
    """ create a custom agent """
    tools = [get_word_length]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are very powerful assistant, but don't know current events"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    model = ChatOpenAI()
    llm_with_tools = model.bind_tools(tools=tools)
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # for step in agent_executor.stream({"input": query}):
    # print("step=", step)
    # 按步骤iterator, 可以输出每一步信息, 但甚至可以不用print step
    # for step in agent_executor.stream({"input": query}):
    #     pass
    # async for s_ in agent_executor.astream_log({"input": query}):
    async for chunk in agent_executor.astream({"input": query}):
        logger.info("chunk=%s", chunk)


def test_custom_agent_with_memory():
    """ create a custom agent, with memory"""
    tools = [get_word_length]
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    model = ChatOpenAI()
    llm_with_tools = model.bind_tools(tools=tools)
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    chat_history = []
    input1 = "how many letters in the word educa?"
    result = agent_executor.invoke({
        "input": input1,
        "chat_history": chat_history,
    })
    chat_history.extend([
        HumanMessage(content=input1),
        AIMessage(content=result["output"])
    ])
    agent_executor.invoke({
        "input": "is that a real word",
        "chat_history": chat_history,
    })
