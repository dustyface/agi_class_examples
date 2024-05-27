""" Test a vectorstore """
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

documents = [
    Document(page_content="Dogs are great companions, known for their loyalty and friendliness.",
             metadata={"source": "mammal-pets-doc"}),
    Document(page_content="Cats are independent pets that often enjoy their own space.",
             metadata={"source": "mammal-pets-doc"},),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds that can be trained to speak.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())


def output_similarity():
    """ output similarity """
    logger.info(vectorstore.similarity_search("cat"))
    logger.info(vectorstore.similarity_search_with_score("cat"))


async def output_similarity_async():
    """ output similarity """
    logger.info(await vectorstore.asimilarity_search("cat"))


def output_by_vector(word: str):
    """ output by vector """
    # .embed(): 为text 制造embedding
    embedding_word = OpenAIEmbeddings().embed(word)
    res = vectorstore.similarity_search_by_vector(embedding_word)
    logger.info(res)


# create a retriever
retriever1 = RunnableLambda(vectorstore.similarity_search).bind(k=1)
# retriever.batch(["cat", "shark"])
retriever2 = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


def output_by_retriever(retriever):
    """ output by retriever """
    res = retriever.batch(["cat", "shark"])
    logger.info(res)


def rag_chain(retriever):
    """ test rag_chain """
    message = """
    Answer question using the provided context only
    {question}
    context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])
    model = ChatOpenAI(model="gpt-3.5-turbo")
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        } | prompt | model)
    res = chain.invoke("tell me about cats")
    return res
