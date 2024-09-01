""" Langgraph code assistant, generate LCEL expression """
from typing import TypedDict
from bs4 import BeautifulSoup as Soup
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langgraph.graph import StateGraph, START, END


def make_rag_content(url: str):
    """ Make RAG content from a given URL """
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()
    # builtin sorted 实际实现是在CPython中;
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    print("rag content=", len(concatenated_content))
    return concatenated_content

# Data Model


class Code(BaseModel):
    """ Code output """
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."


RAG_CONTENT = make_rag_content(
    "https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel")


def get_code_generation_chain():
    """ Get code generation chain """
    # setup LLM
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a python code assistant with expertise in LCEL, LangChain expression language.
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n
    Answer the user question based on the above provided documentation.
    Ensure any code you provide can be executed with all required imports and variables defined.
    Structure your answer with a description of the code solution.
    Then list the imports. And finally list the functioning code block. Here is the user question:

    """
            ),
            ("placeholder", "{messages}")
        ]
    )

    expt_llm = "gpt-4-0125-preview"
    llm = ChatOpenAI(temperature=0, model=expt_llm)
    code_gen_chain = code_gen_prompt | llm.with_structured_output(Code)
    return code_gen_chain


def generate_lcel(question: str):
    """ setup code generate for LCEL """
    code_gen_chain = get_code_generation_chain()
    res = code_gen_chain.invoke({
        "context": RAG_CONTENT,
        "messages": [("user", question)]
    })
    print("response=", res)


class GraphState(TypedDict):
    """ State of graph """
    error: str
    messages: str
    generation: str
    iterations: int


MAX_ITERATION = 3
FLAG = "do not reflect"


# Nodes: Generate code

def generate(state: GraphState):
    """ Generate a code solution """

    print("---GENERATING CODE SOLUTION---")
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block: "
            )
        ]

    code_gen_chain = get_code_generation_chain()
    code_solution = code_gen_chain.invoke({
        "context": RAG_CONTENT, "messages": messages
    })
    messages += [
        (
            "assistant",
            f"""
{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"""
        )
    ]
    iterations += 1
    print("generate: generation=", code_solution,
          "\nmessages=", messages, "\niterations=", iterations)
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations
    }


# Node: Code_check

def code_check(state: GraphState):
    """
    Check code
    Args:
        state(dict): The current graph state

    Returns:
        state(dict): New keys added to the state, error
    """
    print("---CHECKING CODE---")
    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    imports = code_solution.imports
    code = code_solution.code

    try:
        exec(imports)
    except Exception:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed at imports")]
        messages = state["messages"]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }

    # Check execution
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [
            ("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

     # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def reflect(state: GraphState):
    """ Reflect on error """
    print("---GENERATING CODE SOLUTION: Reflect ---")
    # state
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    # prompt reflection
    code_gen_chain = get_code_generation_chain()
    reflections = code_gen_chain.invoke({
        "context": RAG_CONTENT, "messages": messages
    })
    messages += [
        ("assistant", f"""
         Here are reflections on the error: {reflections}""")
    ]
    return {
        "generation": code_solution,
        "messages": messages,
        "iteratons": iterations,
    }


# Edges

def decide_to_finish(state: GraphState):
    """
    Determine whether to finish
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == MAX_ITERATION:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        if FLAG == "reflect":
            return "reflect"
        else:
            return "generate"

# Setup Graph


def setup_code_assistant_graph():
    """ setup code assistant graph """
    workflow = StateGraph(GraphState)

    # Define the node
    workflow.add_node("generate", generate)
    workflow.add_node("check_code", code_check)
    workflow.add_node('reflect', reflect)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "reflect": "reflect",
            "generate": "generate"
        }
    )
    workflow.add_edge("reflect", "generate")
    app = workflow.compile()
    return app
