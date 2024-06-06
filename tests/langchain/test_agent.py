""" Test agent """
import pytest
from langchain_samples.agent.tools import test_wikipedia_tool
from langchain_samples.agent.custom_agent import test_custom_agent, test_custom_agent_with_memory


def test_tool_properties():
    """
    Test tool properties
    pytest tests/langchain/test_agent.py::test_tool_properties
    """
    test_wikipedia_tool()


@pytest.mark.asyncio
async def test_customized_agent():
    """
    pytest tests/langchain/test_agent.py::test_customized_agent
    """
    await test_custom_agent("How many letters in the word educa")


def test_customized_agent_with_memory():
    """
    pytest tests/langchain/test_agent.py::test_customized_agent_with_memory
    """
    test_custom_agent_with_memory()
