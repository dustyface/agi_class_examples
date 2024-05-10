""" Test langchain IO outputparser """
import logging
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# defin a pydantic model class


class Joke(BaseModel):
    """ pydantic model """
    setup: str = Field(description="question to setup a joke")
    punchline: str = Field(descripton="answer to resolve the joke")

    # Add custom validation logic
    @validator("setup")
    # pydanti 要求不能用self当作第一个参数
    # pylint: disable=no-self-argument
    def question_ends_with_question_mark(cls, field):
        """ validate setup class attribute """
        if field[-1] != "?":
            raise ValueError("Badly formed question")
        return field


def test_pydanticparser():
    """ test pydantic outputparser """
    # 1. create
    parser = PydanticOutputParser(pydantic_object=Joke)
    template = "Anwser the user query.\n{format_instructions}\n{query}"
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )
    print("prompt=", prompt)
    # 以下chain相当于：
    #
    # res = chat_model.invoke(prompt.to_messages())
    # output = parser.parse(res.content)
    #
    # 有关`|` overloaded opeartor
    # see: langchain_core/runnables/base.py Runnable class __or__, __ror__ method
    #
    # pylint: disable=unsupported-binary-operation
    chain = prompt | chat_model | parser
    res = chain.invoke({"query": "Tell me a joke"})
    logger.info(res)
