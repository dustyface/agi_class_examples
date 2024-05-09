"""
test langchain IO moduel

api update:
from langchain.chat_model import ChatOpenAI, ErnieBotChat
is about to be deprecated

`from langchain_community.chat_models import ChatOpenAI`.
`from langchain_community.chat_models import ErnieBotChat

"""
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import (
    QianfanChatEndpoint,
    ChatOpenAI as ChatOpenAI_Community
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    load_prompt
)

_ = load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)

chat_model_openai = ChatOpenAI(model="gpt-3.5-turbo")
chat_model_openai2 = ChatOpenAI_Community(model="gpt-3.5-turbo")

# ErnieBotChat 从langchain v0.0.13 废弃了, 推荐使用QianfanChatEndpoint;
# chat_model_erniebot = ErnieBotChat()

# ErnieBotChat使用API Key/Secret Key换取access token的鉴权方式(第一种鉴权方式),
# API Key name: ERNIE_CLIENT_ID, Secret Key name: ERNIE_CLIENT_SECRET
chat_model_qianfan = QianfanChatEndpoint(model="ERNIE-Bot-turbo")

chat_models = [
    chat_model_openai,
    chat_model_openai2,
    chat_model_qianfan
]


def simple_talk():
    """ simple talk """
    for m in chat_models:
        response = m.invoke("你是谁?")
        print(response.content)


def multiround_talk():
    """ multiple round talk """
    messages = [
        SystemMessage(content="你是AGIClass的课程助理。"),
        HumanMessage(content="我是学员，我叫王卓然。"),
        AIMessage(content="欢迎！"),
        HumanMessage(content="我是谁")
    ]
    for m in chat_models:
        res = m.invoke(messages)
    logger.info(res.content)


def prompt_template_talk():
    """ test PromptTemplate """
    # 使用PromptTemplate, 处理单一的template
    template = PromptTemplate.from_template("讲个关于{subject}的笑话")
    prompt_singleround = template.format(subject="程序员")
    logger.info("Testing PropmtTemplate")
    res = chat_model_openai.invoke(prompt_singleround)
    logger.info(res.content)

    # 使用 ChatPromptTemplate 是把多轮的对话，汇总在一个template object中
    template2 = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "你是一个{product}专业的代码助手, 你的名字是{name}"),
        HumanMessagePromptTemplate.from_template("{query}")
    ])
    prompt_multiround = template2.format_messages(
        product="前端开发", name="skywalker", query="你是谁?")
    logger.info("Testing ChatPromptTemplate")
    res = chat_model_openai.invoke(prompt_multiround)
    logger.info(res.content)


def messageholder_talk():
    """ test messageholder """
    human_prompt = "Translate your answer to {language}"
    human_prompt_template = HumanMessagePromptTemplate.from_template(
        human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="conversation"),
        human_prompt_template
    ])
    human_msg = HumanMessage(content="Who is Elon Musk?")
    ai_msg = AIMessage(
        content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer")
    # MessagePlaceholder可以将多个Message对象，插入到该placeholder插桩的位置
    messages = chat_prompt.format_prompt(
        # conversion的位置，可以被替换为多个message
        conversation=[human_msg, ai_msg], language="日文"
    )
    res = chat_model_openai.invoke(messages)
    logger.info(res.content)


def load_prompt_from_file():
    """ test load prompt file """
    prompt = load_prompt("./src/zhihu/LangChain/prompt.yaml")
    logger.info(prompt.format(adjective="funny", content="fox"))
    prompt = load_prompt("./src/zhihu/LangChain/prompt.json")
    logger.info(prompt.format(adjective="funny", content="programmer"))
