""" tutorial for langgraph """
import logging
from typing import Annotated, List
from pydantic import BaseModel
from typing_extensions import TypedDict
from dotenv import load_dotenv, find_dotenv

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, ToolMessage
# from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

# set_debug(True)


class State(TypedDict):
    """
    This is State class, which is the input param of graph
    It defines the exchanged message between graph nodes
    """
    messages: Annotated[List, add_messages]
    ask_human: bool


class RequestAssistant(BaseModel):
    """ request from user """
    request: str


llm = ChatOpenAI()

# Tools
tavily_search = TavilySearchResults(max_result=2)
tools = [tavily_search, RequestAssistant]
llm_with_tools = llm.bind_tools(tools)

# node: chatbot


def chatbot(state: State):
    """ This is graph node: chatbot """
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistant.__name__:
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


# node: tools_node
tools_node = ToolNode(tools=tools)


def create_response(response: str, ai_message: AIMessage):
    """ create a ToolMessage response """
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"]
    )

# node: human_node, provide default human behaviour


def human_default_node(state: State):
    """ This is graph node: human """
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(
            create_response("No response from human", state["messages"][-1])
        )
    return {
        "messages": new_messages,
        "ask_human": False
    }


def select_next_node(state: State):
    """ router to suited nodes """
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


def human_interrupt_graph():
    """ simulate human action """
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("human", human_default_node)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", "__end__": "__end__"}
    )
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["human"],
    )
    return graph


def use_human_interrupt_graph():
    """ call human_interrupt graph """
    graph = human_interrupt_graph()
    config = {
        "configurable": {"thread_id": "1"}
    }
    interrupted = False
    while True:
        user_tip = "Do you wanna use a true human action?(Y/N)" if interrupted is True else "User: "
        user_input = input(f"{user_tip}")
        use_human_action = False

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye")
            break
        elif interrupted and user_input.lower() in ["yes", "y"]:
            user_input = input("Input your instruction as human action: ")
            use_human_action = True
        elif interrupted and user_input.lower() in ["no", "n"]:
            use_human_action = False

        if interrupted:
            snapshot = graph.get_state(config)
            if use_human_action:
                last_message = snapshot.values["messages"][-1]
                tool_message = create_response(user_input, last_message)
                # overwrite the last message in the state
                graph.update_state(config, {
                    "messages": [tool_message]
                })
                use_human_action = False
            interrupted = False
            events = graph.stream(None, config, stream_mode="values")
            for event in events:
                # print("Assistant: ", value["messages"][-1].content)
                if "messages" in event:
                    print(event["messages"])
        else:
            events = graph.stream(
                {"messages": ("user", user_input)},
                config,
                stream_mode="values")
            for event in events:
                # print("Assistant: ", value["messages"][-1].content)
                if "messages" in event:
                    print(event["messages"])
            snapshot = graph.get_state(config)
            next_state = snapshot.next
            interrupted = True if next_state[0] == "human" else False


def basic_graph_demo():
    """ make a langgraph basic demo """
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye")
            break
        for event in graph.stream({"messages": ("user", user_input)}):
            for value in event.values():
                print("Assistant: ", value["messages"][-1].content)
