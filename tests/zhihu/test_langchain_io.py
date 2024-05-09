""" test langchain"""
import logging
from zhihu.LangChain.basic_io import (
    simple_talk, multiround_talk,
    prompt_template_talk,
    messageholder_talk,
    load_prompt_from_file,
)

logger = logging.getLogger(__name__)


def test_basic_io_models():
    """ test single round & multiround talk with LLM """
    simple_talk()
    multiround_talk()


def test_basic_io_prompttemplate():
    """ test PromptTemplate """
    prompt_template_talk()


def test_basic_io_messageholder():
    """ test PromptTemplate """
    messageholder_talk()


def test_basic_io_loadpromptfile():
    """ test load_prompt() """
    load_prompt_from_file()
