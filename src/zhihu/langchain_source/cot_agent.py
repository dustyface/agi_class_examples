""" 实现一个思维链的agent，和手撕autoGPT的 """
from typing import Optional, Dict, Any, Union, List, Callable
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import load_prompt, PromptTemplate
from langchain.agents.agent import AgentOutputParser, AgentAction, AgentFinish, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import tool
from langchain.tools.render import render_text_description
from langchain_zhipu import ChatZhipuAI
from zhihu.langchain_source.zhipu_model import ask_neighber


@tool
def finish(output: str) -> str:
    """
    输出最终的结果
    @param
    args: output 想要得到的答案
    """
    return output


def load_prompt_from_file():
    """ load prompt """
    prompt_path = "src/zhihu/langchain_source/prompts/"
    system_constraint = load_prompt(prompt_path + "system_constraint.yaml")
    task_tools_hist = load_prompt(prompt_path + "task_tools_hist.yaml")
    output_process = load_prompt(prompt_path + "output_process.yaml")
    result = system_constraint.template + "\n" + \
        task_tools_hist.template + "\n" + output_process.template
    return result


class Action(BaseModel):
    """ Action """
    name: str = Field(
        description="The name of tool or action: FINISH or Other tool names."
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters of tool or action are "
    )


_action_outputparser = PydanticOutputParser(pydantic_object=Action)
# _action_parser_format = _action_outputparser.get_format_instructions()


class ReasonOutputParser(AgentOutputParser):
    """ 解析单个动作的智能体action和输入参数 """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        action: Action = _action_outputparser.invoke(text)
        name: Optional[str] = action.name
        args: Optional[Dict[str, Any]
                       ] = action.args if text is not None else "No args"
        log: str = text if text is not None else ""
        if name.upper() == "FINISH":
            return AgentFinish(args, log)
        elif name is not None:
            return AgentAction(name, args, log)

    @property
    def _type(self) -> str:
        return "Chain-of-Thought"


def prompt_creator() -> Callable[[List[str]], str]:
    """ create prompt """
    def creator(tools: List[str]) -> str:
        prompt = load_prompt_from_file()
        tools_format = render_text_description(tools)
        template = PromptTemplate.from_template(prompt)

        return template.partial(
            tools=tools_format,
            action_format_instructions=_action_outputparser.get_format_instructions()
        )

    return creator


def create_reason_agent(llm, tools):
    """ create reason agent """
    prompt_creator_ = prompt_creator()
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(
                x["intermediate_steps"])
        )
        | prompt_creator_(tools)
        | llm
        | ReasonOutputParser()
    )
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


async def use_create_reason_agent(query: str):
    """ test create reason agent """
    llm = ChatZhipuAI()
    tools = [ask_neighber, finish]
    executor = create_reason_agent(llm, tools)
    async for e in executor.astream_events({"input": query}, version="v1"):
        if e["event"] in ["on_chat_model_end", "on_tool_end"]:
            if "input" in e["data"]:
                print("\n", "-"*10, e["name"], "-"*2, e["event"])
                print("INPUT:")
                print(e["data"]["input"])
            if "output" in e["data"]:
                print("\n", "-"*10, e["name"], "-"*2, e["event"])
                print("OUTPUT:")
                print(e["data"]["output"])
