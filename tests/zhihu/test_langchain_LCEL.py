""" Test LCEL """
from zhihu.LangChain.LCEL import check_chain_schema


def test_schema():
    """ test chain runnable's input_schema / output_schema """
    check_chain_schema()
