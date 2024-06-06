""" A simple agent """
import logging
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import chat_agent_executor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

# create 2 tools: TavilySearch, WebBaseLoader Retriever
tavily_search = TavilySearchResults(max_result=2)
# print(tavily_search.invoke("what is  the weather in SF"))
# print(tavily_search.invoke("what is  the weather in Beijing"))

WEB_URL = "https://docs.smith.langchain.com/overview"
loader = WebBaseLoader(WEB_URL)
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever
# turn retriever into tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about langsmith"
)

tools = [tavily_search, retriever_tool]
model = ChatOpenAI()


def invoke_with_tools(query):
    """ invoke with tools """
    model_with_tools = model.bind_tools(tools)
    # res = model_with_tools.invoke([HumanMessage(content="Hi")])
    # print("res.content", res.content)
    # print("res.tool_calls", res.tool_calls)

    # Note:
    # 默认的gpt-3.5-turbo也可以完成function calling
    # 这是否是langchain加入的能力，即针对function calling的大模型的限制被放宽了
    res = model_with_tools.invoke(
        [HumanMessage(content=query)])
    logger.info("res.content=%s", res.content)
    logger.info("res.tool_calls=%s", res.tool_calls)


async def stream_tokens():
    """
    create agent using langgraph, stream tokens
    这个agent可以支持2类查询，查询天气，查询langsmith的内容，都是仅依靠用户query即可自动执行
    """
    agent = chat_agent_executor.create_tool_calling_executor(
        model, tools)
    # 注意, .astream_events only support with python 3.11 or higher
    async for event in agent.astream_events({
        "messages": [HumanMessage(content="what is the weather in Beijing?")]
    }, version="v1"):
        kind = event["event"]

        if kind == "on_chain_start":
            # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            if event["name"] == "Agent":
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}")
        elif kind == "on_chain_end":
            # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            if event["name"] == "Agent":
                print()
                print("---")
                print(
                    f"""
                    Done agent: {event['name']} with output: {event['data'].get('output')['output']}
                    """
                )

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # empty content means starting to ask tools
                print(content, end="|")
        elif kind == "on_tool_start":
            print("---")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}")
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("---")
