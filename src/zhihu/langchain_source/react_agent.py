""" Test React Agent """
# from langchain import hub
from langchain_core.prompts import load_prompt
from langchain.agents import AgentExecutor, create_react_agent
from langchain_zhipu import ChatZhipuAI
from zhihu.langchain_source.zhipu_model import ask_neighber
from zhihu.langchain_source.custom_react_output_parser import ReActMultipleInputOutputParser


class BgColor:
    """ Color for print """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# pylint: disable=dangerous-default-value


def create_custom_react_executor(model=ChatZhipuAI(), tools=[ask_neighber]):
    """ Test the ReAct agent"""
    # prompt = hub.pull("hwchase17/react")
    prompt = load_prompt(
        "src/zhihu/langchain_source/prompts/custom_react.yaml")
    # print("Check Prompt=\n",
    #       f"{BgColor.OKBLUE}{prompt.pretty_repr()}{BgColor.ENDC}")
    agent = create_react_agent(
        model, tools, prompt, ReActMultipleInputOutputParser())
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


async def use_neighber_react_agent():
    """ Test react neighber agent"""
    executor = create_custom_react_executor()
    async for e in executor.astream_events({"input": "马冬梅住哪里?"}, version="v1"):
        kind = e["event"]
        if kind == "on_chat_model_stream":
            print(e["data"]["chunk"].content, end="_")
        else:
            print(kind)
