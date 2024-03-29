""" A function calling example of sum """
import logging
from zhihu.common.api import Session
from zhihu.function_call.common import MODEL_FUNC_CALL, make_func_tool, function_calling_cb

logger = logging.getLogger(__name__)

# examples:
# prompt = "Tell me the sum of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
# prompt = "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？"
# prompt = "1+2+3...+99+100"


# sum is built-in function, so we use _sum to avoid conflict
def _sum(numbers):
    return sum(numbers)


def function_call_sum(prompt: str):
    """ Test function calling with sum """
    tools = [
        make_func_tool("_sum", "加法器, 计算一组数的和", {
            "numbers": {
                "type": "array",
                "items": {"type": "number"}
            }
        })
    ]
    session = Session()
    session.set_system_prompt("你是一个小学数学老师, 你要教学生加法")
    rsp = session.get_completion(
        prompt,
        model=MODEL_FUNC_CALL,
        temperature=0.7,
        tools=tools,
        clear_session=False   # 调用api接口时, 不清空session message
    )
    message_assistant = rsp.choices[0].message
    session.add_message(message=message_assistant)
    logger.info("message_assistant=%s", message_assistant)

    # sum is built-in function in python
    return function_calling_cb(session, message_assistant, __name__, "_sum")


if __name__ == "__main__":
    logger.info("executing function_call()...")
    function_call_sum("15+16")
