""" test for LCEL to implement function calling """
from typing import Union
from operator import itemgetter
from langchain_core.globals import set_debug
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain.output_parsers import JsonOutputToolsParser
from langchain_openai import ChatOpenAI

set_debug(True)


@tool
def multiply(a: int, b: int) -> int:
    """ multiply two numbers """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """ add two numbers """
    return a + b


@tool
def exponentiate(base: int, exponent: int) -> int:
    """ Exponentiate the base to the exponent power """
    return base ** exponent


def llm_bind_tools():
    """ let the model return function calling definition """
    tools = [multiply, add, exponentiate]
    model = ChatOpenAI(model="gpt-3.5-turbo")

    # 1. model.bind_tools(tools): 这是告知model app所使用的function call的定义的方式
    # 2. runnable | dict, 可以参考RunnableParallel的注释; dict的部分是被coerce_to_runnable函数
    # 转成RunnableParallel
    # 3. dict的key, 是由用户自选的, e.g. 以下的functions可以是tools
    # 4. JsonOutputToolsParser()是专门解析OpenAI的output的tools的内容的parser
    llm_with_tools = model.bind_tools(tools) | {
        "functions": JsonOutputToolsParser(),
        "text": StrOutputParser()
    }
    # output sample:
    # {'functions': [{'args': {'a': 1024, 'b': 16}, 'type': 'multiply'}], 'text': ''}
    res = llm_with_tools.invoke("1024的16倍是多少?")
    print(res)
    res = llm_with_tools.invoke("你是谁?")
    print(res)


def llm_exec_tools(query: str):
    """ let the model execute the function calling """
    tools = [multiply, add, exponentiate]
    # make dict of tools
    tool_map = {tool.name: tool for tool in tools}
    model = ChatOpenAI()

    def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
        """ 根据model选择的tool, 动态创建LCEL Runnable """

        func_tool = tool_map[tool_invocation["type"]]

        # 返回RunnableAssign(RunnableParallel(kwargs))
        # `output=itemgetter("args") | func_tool` 被转成了RunnableParallel
        # 它实际代表着对function call的执行
        # output 这个参数是在最终的结果中, 显示的字段, 它是由用户自定义的, e.g.
        # {"args": {"a": 1024, "b": 16}, "type": "multiply", output: 16384}
        # RunnablePassthrough.assign() 是把output key-value pair assign到input json中作为最终结果
        return RunnablePassthrough.assign(
            output=itemgetter("args") | func_tool
        )
    # Runnable的map(), 是把input list和output list对应起来
    call_tool_list = RunnableLambda(call_tool).map()
    # functions的这部分output 交给 `JsonOutputToolsParser()|call_tool_list` runnable 处理
    llm_with_tools = model.bind_tools(tools) | {
        "functions": JsonOutputToolsParser() | call_tool_list,
        "text": StrOutputParser()
    } | RunnableLambda(lambda res: (
        res["functions"] if len(res["functions"]) > 0 else res["text"]
    ))
    return llm_with_tools.invoke(query)
