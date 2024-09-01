""" Test langgraph Code Assistant """
import logging
import pytest
from langgraph_samples.code_assistant.lcel import (
    setup_code_assistant_graph,
    generate_lcel
)

logger = logging.getLogger(__name__)


# @pytest.mark.asyncio
def test_generate_lcel():
    """
    pytest tests/langgraph/test_code_assistant.py::test_generate_lcel
    """
    # 这是一个普通的lcel case;
    # 是根据用户input question，生成相应的LCEL expression code
    generate_lcel("How do I build a RAG chain in LCEL?")


def test_code_assistant():
    """
    pytest tests/langgraph/test_code_assistant.py::test_code_assistant
    """
    app = setup_code_assistant_graph()
    question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
    app.invoke({"messages": [("user", question)], "iterations": 0})
