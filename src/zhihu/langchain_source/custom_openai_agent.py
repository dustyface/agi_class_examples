""" Test custom agent """
from typing import Sequence, List, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentAction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.agent import AgentExecutor
from langchain import hub
from zhihu.langchain_source.zhipu_model import (
    MiniZhipuAI, ask_neighber
)


class BgColor:
    """ Color for print """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_intermedate_step(arg: List[Tuple[AgentAction, str]]):
    """ Parse intermediate steps """
    print("Intermedate Steps info:")
    for i, (action, output) in enumerate(arg):
        print(
            f"{BgColor.OKBLUE}第{i}个action:\ntool={action.tool!r}, tool_input={action.tool_input!r}{BgColor.ENDC}")
        print(
            f"""{BgColor.OKBLUE}message_log[0]:
content={action.message_log[0].content!r},
additional_kwargs={action.message_log[0].additional_kwargs!r}{BgColor.ENDC}""")
        print(f"{BgColor.OKGREEN}第{i}个output:\noutput={output}{BgColor.ENDC}")


def create_custom_openai_tools_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate


) -> Runnable:
    """ Create a openai tools agent of myself """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    # 确保openai tool calling这类的agent必须包含agent_scratchpad prompt input variable
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(
        tools=[convert_to_openai_tool(tool) for tool in tools])

    # agent本质上只是一个chained runnable
    # RunnablePassthrough的作用是将input dict不变的穿过，.assgin()将传入的参数，作为output dict的key-value pair，合并入input varialb dict中, 作为这个Runnable环节的输出;
    # x["intermediate_steps"], List[Tuple(ActionAgent, Any)], 是Agnet之前动作的结果；是填入agent_sratchpad内容；
    # 见观测中间步骤Intermediate Steps
    def add_agent_scratchpad(x):
        # print intermediate steps info in every round of running
        print_intermedate_step(x["intermediate_steps"])
        return format_to_openai_tool_messages(x["intermediate_steps"])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=add_agent_scratchpad
        )
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return agent

# pylint: disable=dangerous-default-value


def create_openai_agent_executor(model=MiniZhipuAI(), tools=[ask_neighber]) -> AgentExecutor:
    """ test custom openai style agent """
    prompt = hub.pull("hwchase17/openai-tools-agent")
    print("Check Prompt=\n",
          f"{BgColor.OKBLUE}{prompt.pretty_repr()}{BgColor.ENDC}")
    agent = create_custom_openai_tools_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


async def use_agent_with_events():
    """ test agent with astream_events """
    executor = create_openai_agent_executor(MiniZhipuAI(), [ask_neighber])
    async for event in executor.astream_events({"input": "马冬梅住在哪里?"}, version="v1"):
        # print(event["name"], event["tags"], event["event"])
        if event["event"] in ["on_chat_model_end", "on_tool_end"]:
            if "input" in event["data"]:
                print("\n", "-"*10, event["name"], "-"*2, event["event"])
                print("INPUT:")
                print(event["data"]["input"])
            if "output" in event["data"]:
                print("\n", "-"*10, event["name"], "-"*2, event["event"])
                print("OUTPUT:")
                print(event["data"]["output"])
