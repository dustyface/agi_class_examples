""" A function calling example of using eval function returned by LLM """
import logging
from zhihu.common.api import Session
from zhihu.function_call.common import MODEL_FUNC_CALL, function_calling_cb, make_func_tool

logger = logging.getLogger(__name__)


def calculate(**args):
    """ Calculate what the args as expression return by LLM """
    expression = args["expression"]
    try:
        # pylint: disable=eval-used
        result = eval(expression)
        return result
    except ValueError:
        logger.error("expression error in eval", exc_info=True)
    except Exception:
        logger.error("calculate error", exc_info=True)


def function_call_eval(user_query: str):
    """ Test function calling with eval"""
    tools = [make_func_tool("calculate", "计算一个数学表达式的值",
                            {
                                "expression": """
                                a mathmatical expression strictly in a valid python grammar syntax
                                """
                            })]
    session = Session(system_prompt="你是一个数学家，你可以计算任何算式")
    rsp = session.get_completion(
        user_query,
        model=MODEL_FUNC_CALL,
        seed=1024,
        tools=tools,
        clear_session=False
    )
    message_assistant = rsp.choices[0].message
    session.add_message(message=message_assistant)
    logger.info("message_assistant=%s", message_assistant)
    return function_calling_cb(session, message_assistant,  __name__, "calculate")


if __name__ == "__main__":
    logger.info("executing eval...")
    logger.info("output=%s", function_call_eval("3的平方根乘以2在开方"))
