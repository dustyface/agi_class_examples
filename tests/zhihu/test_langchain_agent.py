""" Test agent """
from zhihu.LangChain.agent import (
    get_react_template, agent_react_with_search,
    with_structured_result,
    agent_selfask_with_search
)


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
