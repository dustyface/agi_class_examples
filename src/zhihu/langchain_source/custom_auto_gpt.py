""" create a task agent that is autoGPT alike """
import sys
import os
import re
from typing import List, Union
import fnmatch
import pandas as pd
from colorama import Fore, Style
from langchain_core.prompts import ChatPromptTemplate, load_prompt, PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.utilities import PythonREPL
from langchain_community.document_loaders import Docx2txtLoader
from langchain.tools import tool
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_zhipu import ChatZhipuAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from zhihu.langchain_source.custom_openai_agent import create_openai_agent_executor
from zhihu.langchain_source.react_agent import create_custom_react_executor
# from zhihu.langchain_source.zhipu_model import MiniZhipuAI


THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
ROUND_COLOR = Fore.RED
CODE_COLOR = Fore.BLUE


def color_print(text, color=None, end="\n"):
    """ print text with color """
    if color is not None:
        content = color + text + Style.RESET_ALL + end
    else:
        content = text + end
    sys.stdout.write(content)
    sys.stdout.flush()

# Tools definition


WORK_DIR = "./data"


@tool
def list_files() -> str:
    """ 如果需要查询资料，应该先使用这个工具列出本地文件夹中的结构和内容, 展示它的文件名和文件夹名 """
    print("当前目录为:", WORK_DIR)
    patterns = ["*.pdf", "*.xlsx"]
    if os.path.isdir(WORK_DIR):
        all_files = os.listdir(WORK_DIR)
        matching_files = [
            f for f in all_files for p in patterns if fnmatch.fnmatch(f, p)]
        return "\n".join(matching_files)
    else:
        return []


def get_file_extension(filename: str) -> str:
    """ get the file extension """
    return filename.split(".")[-1]


def format_docs(docs: List[str]) -> str:
    """ format the docs into each page """
    return "\n\n".join(d.page_content for d in docs)


def convert_message_to_str(message: Union[BaseMessage, str]) -> str:
    """ convert langchain message to string """
    if isinstance(message, BaseMessage):
        return message.content
    else:
        return message


class FileLoadFactory:
    """ Factory class to get the file loader"""
    @staticmethod
    def get_loader(filename: str):
        """ 根据文件类型加载不同的Document Loader """
        filename = filename.strip()
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return Docx2txtLoader(filename)
        else:
            raise NotImplementedError(f"File extension {ext} not supported")


def load_docs(filename: str) -> List[Document]:
    """ 加载列表中的文档, 分割doc为切片 """
    file_loader = FileLoadFactory.get_loader(filename)
    return file_loader.load_and_split()


@tool
def ask_document(filename: str, query: str) -> str:
    """
    查询Word或PDF文档中的文本内容, 以便回答问题.
    考虑上下文信息, 确保问题对相关概念的定义表述完整.
    """

    path = os.path.join(WORK_DIR, filename)
    if not os.path.exists(path):
        return f"给定的文件路径不存在, 请从工作目录{WORK_DIR}中列举文件，确认其存在"

    chunks = load_docs(path)
    # print("chunks=", chunks)
    if chunks is None or len(chunks) == 0:
        return "无法读取文档内容"

    db = Chroma.from_documents(chunks, OpenAIEmbeddings())

    # pylint: disable=invalid-name
    DEFAULT_QA_CHAIN_PROMPT = """
        你要严格依据如下资料回答问题, 你的回答不能与其冲突, 更不要编造。
        请始终使用中文回答。

        {context}

        问题: {question}
        """
    prompt = ChatPromptTemplate.from_template(DEFAULT_QA_CHAIN_PROMPT)
    qa_chain_prompt = (
        {
            "context": db.as_retriever() | format_docs,
            "question": lambda x: convert_message_to_str(x),
        }
        | prompt
    )
    # res = qa_chain_prompt.invoke(query)
    # print("\nCheck chain prompt input:", res)
    qa_chain = qa_chain_prompt | ChatZhipuAI()

    final_output = ""
    for chunk in qa_chain.stream(query):
        print(chunk.content, end="|")
        final_output += chunk.content
    return final_output


def get_sheet_names(filename: str) -> str:
    """ 获取 Excel 文件的工作表名称 """
    path = filename
    if not os.path.exists(path):
        return f"给定的文件路径不存在, 请从工作目录{WORK_DIR}中列举文件，确认其存在"

    excel_file = pd.ExcelFile(path.strip())
    sheet_names = excel_file.sheet_names
    return f"这是'{path}' 文件的工作表名称: \n\n{sheet_names}"


def get_column_names(filename: str) -> str:
    """ 获取 Excel 文件的列名 """
    path = filename
    if not os.path.exists(path):
        return f"给定的文件路径不存在, 请从工作目录{WORK_DIR}中列举文件，确认其存在"

    df = pd.read_excel(path.strip(), sheet_name=0)  # sheet_name = 0 表示第一个工作表
    column_names = "\n".join(
        df.columns.to_list()
    )
    return f"这是 '{path.strip()}' 文件第一个工作表的列名: \n\n{column_names}"


def get_first_n_rows(filename: str, n: int = 3) -> str:
    """ 获取前n个excel的行的内容 """
    path = os.path.join(WORK_DIR, filename)
    path = path.strip()
    if not os.path.exists(path):
        return f"给定的文件路径不存在, 请从工作目录{WORK_DIR}中列举文件，确认其存在"

    result = get_sheet_names(path) + "\n\n"
    result += get_column_names(path) + "\n\n"

    df = pd.read_excel(path, sheet_name=0)
    n_lines = "\n".join(
        df.head(n).to_string(index=False, header=True).split("\n")
    )
    result += f"这是 '{path}' 文件第一个工作表的前{n}行内容: \n\n{n_lines}"
    return result


@tool
def inspect_excel(filename: str, n: int = 3) -> str:
    """
    探查Excel数据文件的内容和结构, 展示它的列名和前n行, n默认为3，
    注意, 该工具仅使用于探查Excel文件， 不能探查PDF或Word文件

    使用该函数时应当准备提供filename和n两个参数, 其中：

    - filename: 要探查的Excel文件名
    - n: 要显示的行数, 默认为3
    """
    return get_first_n_rows(filename, n)


class PythonCodeParser(BaseOutputParser):
    """ 从大模型返回的文本中提取Python代码 """

    def parse(self, text: str) -> str:
        """ abstract the output into string of python code """
        # find all python code blocks
        python_code_blocks = re.findall(
            r"```python\n(.*?)\n```", text, re.DOTALL)
        # print("python_code_blocks=", python_code_blocks)
        python_code = None
        if len(python_code_blocks) > 0:
            python_code = python_code_blocks[0]
            python_code = self._remove_marked_lines(python_code)
        return python_code

    def _remove_marked_lines(self, input_str: str) -> str:
        lines = input_str.strip().split("\n")
        if lines and lines[0].strip().startswith("```"):
            del lines[0]
        if lines and lines[-1].strip().startswith("```"):
            del lines[-1]

        ans = "\n".join(lines)
        return ans


def load_excel_analyzer_prompt() -> PromptTemplate:
    """ 加载Excel分析器的提示词 """
    prompt_path = "src/zhihu/langchain_source/prompts/"
    excel_analyzer_prompt = load_prompt(prompt_path + "excel_analyzer.yaml")
    print("excel_analyzer_prompt=", type(excel_analyzer_prompt))
    return excel_analyzer_prompt


analyze_prompt = load_excel_analyzer_prompt()
model = ChatZhipuAI()
analyze_chain = analyze_prompt | model | PythonCodeParser()


@tool
def excel_analyse(query: str, filename: str) -> str:
    """
    给定了一个Excel文件, 根据该工具分析其内容
    注意, 使用本工具需提供2个参数:
    - query: 用户要查询的问题
    - filename: 要分析的Excel文件名
    """
    path = os.path.join(WORK_DIR, filename)
    path = path.strip()
    if not os.path.exists(path):
        return f"给定的文件路径不存在, 请从工作目录{WORK_DIR}中列举文件，确认其存在"

    inspections = get_first_n_rows(filename, 3)
    color_print("\n#!/usr/bin/env python", CODE_COLOR, end="\n")
    code = ""
    for c in analyze_chain.stream({
        "query": query,
        "filename": path,
        "inspections": inspections,
        "list_of_library": ",".join(["pandas", "re", "math", "datetime", "openpyxl"])
    }):
        color_print(c, CODE_COLOR, end="")
        code += c

    if code:
        return PythonREPL().run(code)
    else:
        return "没有找到可执行的Python代码"


llm = ChatZhipuAI()
tools = [
    # 列举本地文档
    list_files,
    # RAG 查询文档
    ask_document,
    # 探查 Excel 文件
    inspect_excel,
    # Excel 数据分析
    excel_analyse
]


async def use_executor(executor: AgentExecutor, query: str):
    """ 使用openai agent """
    async for chunk in executor.astream_events({
        "input": query,
    }, version="v1"):
        event_name = chunk["event"]
        if event_name in ["on_cha_model_end", "on_tool_end"]:
            print("\n", "-"*10, event_name, "-"*2, chunk["name"])
            print("chunk=", chunk)
            if "input" in chunk["data"]:
                print("INPUT:")
                print(chunk["data"]["input"])
            if "output" in chunk["data"]:
                print("OUTPUT:")
                print(chunk["data"]["output"])


async def use_react_executor(query: str, model_tag: str = "zhipu"):
    """ 使用react agent """
    llm_ = ChatZhipuAI() if model_tag == "zhipu" else ChatOpenAI()
    glm_react_executor = create_custom_react_executor(llm_, tools)
    await use_executor(glm_react_executor, query)


async def use_openai_executor(query: str):
    """ 使用openai agent """
    glm_openai_executor = create_openai_agent_executor(llm, tools)
    await use_executor(glm_openai_executor, query)
