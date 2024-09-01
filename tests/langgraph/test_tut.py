""" Test langgraph Tutorial """
import logging
from langgraph_samples.tutorial.quick_start import (
    basic_graph_demo,
    use_human_interrupt_graph
)


logger = logging.getLogger(__name__)


def test_basic_demo():
    """
    Test graph basic demo
    pytest tests/langgraph/test_tut.py::test_basic_demo
    """
    basic_graph_demo()


def test_use_human_interrupt():
    """
    pytest tests/langgraph/test_tut.py::test_use_human_interrupt
    """
    use_human_interrupt_graph()
