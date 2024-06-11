""" Test agent """
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from zhihu.LangChain.agent import (
    get_react_template, agent_react_with_search,
    with_structured_result,
    agent_selfask_with_search
)
from zhihu.langchain_source.chat_model_older import ChatModelOlder
from zhihu.langchain_source.zhipu_model import test_zhipu_chain_runnable


def test_react_template():
    """ test hub react prompt template
    pytest tests/zhihu/test_langchain_agent.py::test_react_template
    """
    get_react_template()


def test_agent_react_with_search():
    """ test agent react with search
    pytest tests/zhihu/test_langchain_agent.py::test_agent_react_with_search
    """
    result = agent_react_with_search("2024年周杰伦的演唱会是星期几?")
    # agent_react_with_search("帮我查出2024年周杰伦演唱会是哪一天, 我要知道那一天是星期几?")
    with_structured_result(result)


def test_agent_self_with_search():
    """
    test agent self with search
    pytest tests/zhihu/test_langchain_agent.py::test_agent_self_with_search
    """
    agent_selfask_with_search()


def test_chat_with_older():
    """
    test chat with customized model
    pytest tests/zhihu/test_langchain_agent.py::test_chat_with_older
    """
    model = ChatModelOlder()
    questions = [
        "楼上322住的是马冬梅家吗？",
        "马冬梅啊",
        "马冬梅！",
        "我是说马冬梅！",
        "大爷您歇着吧..."
    ]
    for question in questions:
        print(f"\n\n夏洛: {question}")
        print("大爷: ", end="")
        # print(model.invoke(question).content, end="")
        for chunk in model.stream(question):
            print(chunk.content, end="|")


def test_chat_with_older_with_chain():
    """
    use a simple chain to test chat_with_older
    pytest tests/zhihu/test_langchain_agent.py:test_chat_with_older_with_chain
    """
    model = ChatModelOlder()
    questions = [
        "楼上322住的是马冬梅家吗？",
        "马冬梅啊",
        "马冬梅！",
        "我是说马冬梅！",
        "大爷您歇着吧..."
    ]
    prompt = ChatPromptTemplate.from_template("大爷: {question}")
    chain = prompt | model | StrOutputParser()
    print("\n\n")
    print("chain structure:")
    chain.get_graph().print_ascii()
    for question in questions:
        print(f"\n\n夏洛: {question}")
        print("大爷: ", end="")
        for chunk in chain.stream({"question": question}):
            print(chunk, end="|")


def test_zhipu_custom_llm():
    """
    test zhipu custom llm
    pytest tests/zhihu/test_langchain_agent.py::test_zhipu_custom_llm
    """
    print(test_zhipu_chain_runnable())
