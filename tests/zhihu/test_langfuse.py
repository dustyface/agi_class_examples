""" Test LangFuse test case """
from zhihu.llm_tools.hello import run


def test_hello():
    """ A most basic test case
    pytest tests/zhihu/test_langfuse.py::test_hello
    """
    print(run())
