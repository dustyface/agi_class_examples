from zhihu.common.api import Session
from zhihu.function_call.common import model_func_call, function_calling_cb, make_func_tool
import logging

logger = logging.getLogger(__name__)


def calculate(**args):
    expression = args["expression"]
    try:
        result = eval(expression)
        return result
    except Exception as e:
        logger.error("calculate error", exc_info=True)


def function_call_eval(prompt:str):
    tools = [make_func_tool("calculate", "计算一个数学表达式的值", 
        {
            "expression": "a mathmatical expression strictly in a valid python grammar syntax"
        })]
    session = Session(system_prompt="你是一个数学家，你可以计算任何算式")
    rsp = session.get_completion(
        prompt,
        model=model_func_call,
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
    prompt = "3的平方根乘以2在开方"
    output = function_call_eval(prompt)
    logger.info("output=%s", output)
        




