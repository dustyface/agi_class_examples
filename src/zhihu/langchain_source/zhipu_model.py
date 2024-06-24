""" Test implementing zhihpu model """
import re
from typing import Optional, List, Any
from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI
from langchain_core.runnables import chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
from langchain_community.adapters.openai import convert_message_to_dict, convert_dict_to_message

load_dotenv(find_dotenv(), override=True)


@chain
def ask_zhipu_model(prompt_value: ChatPromptValue) -> AIMessage:
    """ ask zhipu model """
    # chain decorator 把这个函数转成RunnableLambda
    # 函数的入参可以是str, List[BaseMessage], PromptValue
    # 返回值是ChatMessage
    client = ZhipuAI()
    response = client.chat.completions.create(
        model="glm-4",
        messages=[convert_message_to_dict(m)
                  for m in prompt_value.to_messages()]
    )
    return AIMessage(response.choices[0].message.content)


def test_zhipu_chain_runnable():
    """ test chain decorated model """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译机器人，我说中文你就直接翻译成英文，我说英文你就直接翻译为中文。不要输出其他，不要啰嗦。"),
        ("human", "你好"),
        ("ai", "hello"),
        ("human", "{question}"),
    ])
    chain_ = prompt | ask_zhipu_model | StrOutputParser()
    return chain_.invoke({"question": "你叫什么名字?"})


class MiniZhipuAI(BaseChatModel):
    """ Support the zhipu model """
    client: Optional[ZhipuAI] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = ZhipuAI()

    @property
    def _llm_type(self) -> str:
        """ Return the type of chat model """
        return "zhipuai"

    def _ask_remote(self, messages, stream=False, **kwargs):
        """ access zhihpu API """
        # 把langchain的message格式, 转换成zhipu的message 格式
        dict_zhipu = [convert_message_to_dict(m) for m in messages]
        # print("messages=", dict_zhipu, "stream=", stream, "kwargs=", kwargs)
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=dict_zhipu,
            stream=stream,
            **kwargs
        )
        # 需要把 zhipu的response转成langchain的message格式
        if not isinstance(response, dict):
            response = response.model_dump()
        return [convert_dict_to_message(c["message"]) for c in response["choices"]]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        generate the chat result with zhipu
        """
        response = self._ask_remote(messages, stream=False, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=m) for m in response])


@tool
def ask_neighber(query: str) -> str:
    """ 我是马冬梅的邻居老大爷, 关于她的事情你可以问我 """
    if re.search("马冬梅", query):
        return "楼上322"
    else:
        return "我不清楚"


def use_binded_model():
    """ 可以用这个方法来测试绑定tool的model """
    model = MiniZhipuAI().bind(tools=[convert_to_openai_tool(ask_neighber)])
    # 可以查看面对这个query, 是否返回正确的tool calling的识别message(content=""为空, tool存在定义的message)

    # 特别夸张的是, 这个query zhipu的model会直接报500 internal error
    # return model.invoke("马冬梅是谁?")
    # 但是以下的query, zhipu model可以正常返回结果
    return model.invoke("告诉我马冬梅在那个房间?")


def use_neighber_openai_agent():
    """ Test using neighber agent """
    # 把自定义的custom model用于agent
    model = MiniZhipuAI()
    # model = ChatZhipuAI()
    tools = [ask_neighber]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # agent_executor.invoke({"input": "马冬梅是谁?"})
    for action in agent_executor.stream({"input": "马冬梅是谁?"}):
        print(action)
