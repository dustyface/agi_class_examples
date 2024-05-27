""" Test LCEL """
import logging
from enum import Enum
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
chat_model = QianfanChatEndpoint()


def simple_chain():
    """ make a simple chain """
    prompt = ChatPromptTemplate.from_template(
        "tell me a joke about {subject}")
    parser = StrOutputParser()
    chain = prompt | chat_model | parser
    return chain


def check_chain_schema():
    """ check input schema """
    # chain的input schema是chain的第一个runnable element - prompt
    # 的input_schema
    # .schema()是input_schema/output_schema pydantic model的JSON表示形式
    # see: https://python.langchain.com/v0.1/docs/expression_language/interface/#input-schema
    chain = simple_chain()

    def get_schema_keys(schema_json):
        return [
            key for i, (key,) in enumerate(schema_json.items()) if not key == "properties"
        ]

    prompt_schema = chain.input_schema.schema()
    keys = get_schema_keys(prompt_schema)
    logger.info("prompt schema(): %s", keys)

    chat_model_schema = chat_model.input_schema.schema()
    keys = get_schema_keys(chat_model_schema)
    logger.info("qianfan model schema(): %s", keys)

    chat_model_output_schema = chain.output_schema.schema()
    keys = get_schema_keys(chat_model_output_schema)
    logger.info("chain.output_schema.schema(): %s", keys)


def search_rag_chain():
    """ make a chain with RAG """
    model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    vector_store = DocArrayInMemorySearch.from_texts([
        "harrison worked at kensho", "bears like to eat honey"
    ], embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    template = """
    Answer the queston base only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    # RunnableParallel 可以根据prompt template的input variable自动来创建并行任务
    setup_and_retrieval = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
    })
    chain = setup_and_retrieval | prompt | model | output_parser
    res = chain.invoke("where did harrison work?")
    # 以下这种形式是不支持的; from_template这个方法构成promptValue, 在invoke时，只支持一个input
    # res = chain.invoke({
    #     "question1": "where did harrison work?",
    #     "question2": "what do bears like to eat?"
    # })
    return res


def search_rag_chain2():
    """ make a chain with RAG """
    model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    vector_store = DocArrayInMemorySearch.from_texts([
        "harrison worked at kensho", "bears like to eat honey"
    ], embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    template = """
    Answer the queston base only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("user", template)
    ])
    # prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    setup_and_retrieval = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
        # 使用Runnable的方式, 不能直接在属性中传入str，Runnable所能接纳的
        # runnable, callable, dict; 不能接受str
        # "question": "where did harrison work?"
    })

    # 踩坑2: 以下这种形式，虽然可以正确调用，但是通过retriever获取的context并不能正确处理
    # setup_and_retrieval = {
    #     "context": retriever,
    #     # "question2": RunnablePassthrough(),
    #     "question": "where did harrison work?"
    # }
    # chain = prompt | model | output_parser
    # res = chain.invoke(setup_and_retrieval)

    chain = setup_and_retrieval | prompt | model | output_parser
    # 踩坑1: 注意, 通过Runnable构建的chain, invoke()只能接受一个input参数，一般是string
    # Runable 的参数是InputType
    res = chain.invoke("where did harrison work?")
    return res


class SortEnum(str, Enum):
    """ SortEnum """
    data = "data"
    price = "price"


class OrderingEnum(str, Enum):
    """ OrderingEnum """
    asc = "asc"
    desc = "desc"


class Semantics(BaseModel):
    """ Semantic """
    name: Optional[str] = Field(description="流量包的名字", default=None)
    price_upper: Optional[int] = Field(description="价格上限", default=None)
    price_lower: Optional[int] = Field(description="价格下限", default=None)
    data_upper: Optional[int] = Field(description="流量上限", default=None)
    data_lower: Optional[int] = Field(description="流量下限", default=None)
    sort_by: Optional[SortEnum] = Field(description="排序字段", default=None)
    ordering: Optional[OrderingEnum] = Field(
        description="升序或降序排序", default=None)


def search_semantcs_package():
    """ test """
    parser = PydanticOutputParser(pydantic_object=Semantics)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "将用户的输入解析成JSON表示。输出格式如下: \n{format_instructions}\n不要输出未提及字段"),
        ("human", "{text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    model = ChatOpenAI()
    # 注意, 当parser为PydanticOutputParser, 最终返回的是PydanticObject
    # 它无法在for s in chain.stream()中使用, 因为pydantic parser会将stream出来的部分内容进行解析
    # 此时会爆出错误
    # runnable = (
    #     {"text": RunnablePassthrough()} | prompt | model | parser
    # )
    runnable = (
        {"text": RunnablePassthrough()} | prompt | model | StrOutputParser()
    )
    for s in runnable.stream("不超过100元的套餐有哪些?"):
        print(s, end="")  # 让stream的内容在同一行
    # res = runnable.invoke("不超过100元的套餐有哪些?")
    # print(res, type(res))


def rag_lcel():
    """ LCEL simulate RAG """
    loader = PyPDFLoader("src/zhihu/RAG/llama2-test-1-4.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    texts = text_splitter.create_documents([
        page.page_content for page in pages[:4]
    ])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    template = """
    Answer the question based on the following context:
    {context}

    Question: {question}
    """
    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | prompt
        | model
        | StrOutputParser()
    )
    res = rag_chain.invoke("Llama 2有多少参数?")
    return res
