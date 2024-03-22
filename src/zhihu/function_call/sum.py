from zhihu.common.api import Session
from zhihu.function_call.common import model_func_call, make_func_tool, function_calling_cb
import logging

logger = logging.getLogger(__name__)

# prompt = "Tell me the sum of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
# prompt = "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？"
# prompt = "1+2+3...+99+100"

def _sum(numbers):
    return sum(numbers)

def function_call_sum(prompt:str):
    tools = [
        make_func_tool("_sum", "加法器, 计算一组数的和", {
            "numbers": {
                "type": "array",
                "items": { "type": "number"}
            }
        })
    ]
    session = Session()
    session.set_system_prompt("你是一个小学数学老师, 你要教学生加法")
    rsp = session.get_completion(
        prompt, 
        model=model_func_call,
        temperature=0.7, 
        tools= tools,
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