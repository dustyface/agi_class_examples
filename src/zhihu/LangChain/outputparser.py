""" Test langchain IO outputparser """
import logging
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# defin a pydantic model class


class Joke(BaseModel):
    """ Pydantic model """
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

# 课堂例子 pydantic parser


class Date(BaseModel):
    """ Pydantic model """
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")

    @validator("month")
    def valid_month(cls, field):
        """ validate month field """
        if field <= 0 or field > 12:
            raise ValueError("月份必须在1-12之间")
        return field

    @validator("day")
    def valid_day(cls, field):
        """ validate day field """
        if field <= 0 or field > 31:
            raise ValueError("日期必须在1-31日之间")
        return field

    # pre: 这个vaidator是否在standard validator之前调用
    @validator("day", pre=True, always=True)
    def valid_date(cls, day, values):
        """ validate day before 'valid_day' checker """
        year = values.get("year")
        month = values.get("month")

        if year is None or month is None:
            return day

        if month == 2:
            if cls.is_leap_year(year) and day > 29:
                raise ValueError("闰年2月最多有29天")
            elif not cls.is_leap_year(year) and day > 28:
                raise ValueError("非闰年2月最多有28天")
        elif month in [4, 6, 9, 11] and day > 30:
            raise ValueError(f"{month}月最多有30天")

        return day

    @staticmethod
    def is_leap_year(year):
        """ util is_leap_year """
        return True if (year % 400 == 0 or (
            year % 4 == 0 and year % 100 != 0
        )) else False


def test_pydanticparser_2():
    """ Test pydanticparser 2 """
    parser = PydanticOutputParser(pydantic_object=Date)
    template = """
    提取用户输入中的日期.
    {format_instructions}
    用户输入:
    {query}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )
    chain = prompt | chat_model | parser
    res = chain.invoke({
        "query": "2023年四月6日天气晴..."
    })
    logger.info(res)

# Test auto-fix output parser


def test_auto_fixparser():
    """ Test autofix parser """
    parser = PydanticOutputParser(pydantic_object=Date)
    fix_parser = OutputFixingParser.from_llm(
        parser=parser,
        llm=chat_model
    )
    error_output = """
    {
        "year": 2023,
        "month": 四月,
        "day": 3
    }
    """
    data = fix_parser.parse(error_output)
    logger.info("重新解析结果 %s", data)
