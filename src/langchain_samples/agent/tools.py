""" Test Tools """
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


def test_wikipedia_tool():
    """ Test basic info for Wikipedia Tool """
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1, doc_content_chars_max=100)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    print("name=", tool.name)
    print("description=", tool.description)
    print("args=", tool.args)
    print("return_direct=", tool.return_direct)    # 是否返回给用户
    # 当args是只有一个key时，以下2种调用方式都可以; 但当存在多个key时, 需要使用dict
    res = tool.run("langchain")
    res2 = tool.run({
        "query": "langchain"
    })
    print("res=", res)
    print("res2=", res2)
