""" Test implementing zhihpu model """
from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI
from langchain_core.runnables import chain
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.adapters.openai import convert_message_to_dict

load_dotenv(find_dotenv())


@chain
def ask_zhipu_model(prompt_value: ChatPromptValue) -> AIMessage:
    """ ask zhipu model """
    # chain decorator 把这个函数转成RunnableLambda
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
