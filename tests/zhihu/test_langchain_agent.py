""" Test agent """
import pytest
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from zhihu.LangChain.agent import (
    get_react_template, agent_react_with_search,
    with_structured_result,
    agent_selfask_with_search
)
from zhihu.langchain_source.chat_model_older import ChatModelOlder
from zhihu.langchain_source.zhipu_model import (
    test_zhipu_chain_runnable, MiniZhipuAI, use_neighber_openai_agent,
    use_binded_model, ask_neighber
)
from zhihu.langchain_source.custom_openai_agent import (
    use_agent_with_events,
    create_openai_agent_executor
)
from zhihu.langchain_source.react_agent import use_neighber_react_agent
from zhihu.langchain_source.cot_agent import use_create_reason_agent
from zhihu.langchain_source.custom_auto_gpt import (
    ask_document, inspect_excel, load_excel_analyzer_prompt, excel_analyse,
    use_openai_executor, use_react_executor, use_cot_executor
)
from zhihu.langchain_source.zhipu_model import MiniZhipuAI

# set_debug(True)


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


def test_chat_with_older():
    """
    test chat with customized model
    pytest tests/zhihu/test_langchain_agent.py::test_chat_with_older
    """
    model = ChatModelOlder()
    questions = [
        "楼上322住的是马冬梅家吗？",
        "马冬梅啊",
        "马冬梅！",
        "我是说马冬梅！",
        "大爷您歇着吧..."
    ]
    for question in questions:
        print(f"\n\n夏洛: {question}")
        print("大爷: ", end="")
        # print(model.invoke(question).content, end="")
        for chunk in model.stream(question):
            print(chunk.content, end="|")


def test_chat_with_older_with_chain():
    """
    use a simple chain to test chat_with_older
    pytest tests/zhihu/test_langchain_agent.py:test_chat_with_older_with_chain
    """
    model = ChatModelOlder()
    questions = [
        "楼上322住的是马冬梅家吗？",
        "马冬梅啊",
        "马冬梅！",
        "我是说马冬梅！",
        "大爷您歇着吧..."
    ]
    prompt = ChatPromptTemplate.from_template("大爷: {question}")
    chain = prompt | model | StrOutputParser()
    print("\n\n")
    print("chain structure:")
    chain.get_graph().print_ascii()
    for question in questions:
        print(f"\n\n夏洛: {question}")
        print("大爷: ", end="")
        for chunk in chain.stream({"question": question}):
            print(chunk, end="|")


def test_zhipu_custom_llm():
    """
    test zhipu custom llm
    pytest tests/zhihu/test_langchain_agent.py::test_zhipu_custom_llm
    """
    print(test_zhipu_chain_runnable())


def test_zhipu_chain():
    """
    Use custom llm in chain for accessing zhihpu API
    pytest tests/zhihu/test_langchain_agent.py::test_zhipu_chain
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个中英互译机器人，只负责翻译，不要试图对问题做解答。我说中文你就直接翻译成英文，我说英文你就直接翻译为中文。不要输出其他，不要啰嗦。"),
        ("human", "你好"),
        ("ai", "hello"),
        ("human", "{question}"),
    ])
    llm_zhipu = MiniZhipuAI()
    chain = prompt | llm_zhipu | StrOutputParser()
    for chunk in chain.stream({
        "question": "The competition between China and the United States in the AI field is very intense. Can China catch up?"
    }):
        print(chunk, end="|")


def test_agent_with_minizhipu():
    """
    Test using MiniZhipuAI in agent
    pytest tests/zhihu/test_langchain_agent.py::test_agent_with_minizhipu
    """
    print(use_binded_model())
    use_neighber_openai_agent()


def test_use_custom_openai_agent():
    """
    Test intermediate steps in custom openai agent
    pytest tests/zhihu/test_langchain_agent.py::test_use_custom_openai_agent
    """
    # llm = MiniZhipuAI()
    # tools = [ask_neighber]
    executor = create_openai_agent_executor()
    for chunk in executor.stream({"input": "马冬梅是谁?"}):
        print()
        print(chunk, end="|")
    # res = executor.invoke({"input": "马冬梅是谁?"})
    # print(res)


@pytest.mark.asyncios
async def test_use_agent_with_events():
    """
    pytest tests/zhihu/test_langchain_agent.py::test_use_agent_with_events
    """
    await use_agent_with_events()


@pytest.mark.asyncio
async def test_use_neighber_agent():
    """
    pytest tests/zhihu/test_langchain_agent.py::test_use_neighber_agent
    """
    await use_neighber_react_agent()


@pytest.mark.asyncio
async def test_use_create_reason_agent():
    """
    pytest tests/zhihu/test_langchain_agent.py::test_use_create_reason_agent
    """
    # load_prompt_from_file()
    await use_create_reason_agent("马冬梅住在哪里?")


def test_ask_document():
    """
    test the tool querying the content of a document
    pytest tests/zhihu/test_langchain_agent.py::test_ask_document
    """
    ask_document.invoke({"filename": "供应商资格要求.pdf", "query": "供应商达标标准"})


def test_inspect_excel():
    """
    Test the tool inspect excel
    pytest tests/zhihu/test_langchain_agent.py::test_inspect_excel
    """
    # print(inspect_excel.invoke({"filename": "2023年8月-9月销售记录.xlsx", "n": 10}))
    # test load_excel_prompt
    # load_excel_analyzer_prompt()
    result = excel_analyse.invoke({
        "query": "销售总额是多少?",
        "filename": "2023年8月-9月销售记录.xlsx"
    })
    print("result=", result)


@pytest.mark.asyncio
async def test_analyze_excel_pdf_openai_agent():
    """
    Test analyze excel pdf openai agent
    pytest tests/zhihu/test_langchain_agent.py::test_analyze_excel_pdf_openai_agent
    """
    await use_openai_executor("供应商达标的业绩要求是什么")


@pytest.mark.asyncio
async def test_analyze_excel_pdf_react_agent():
    """
    test using react agent to analyze excel and pdf
    pytest tests/zhihu/test_langchain_agent.py::test_analyze_excel_pdf_react_agent
    """
    await use_react_executor("供应商达标的业绩要求是什么", model_tag="openai")


@pytest.mark.asyncio
async def test_cot_agent():
    """
    pytest tests/zhihu/test_langchain_agent.py::test_cot_agent
    """
    await use_cot_executor("供应商达标的标准是什么？")
    # 下一条无法分析成功
    await use_cot_executor("9月份有哪些供应商达标?")
