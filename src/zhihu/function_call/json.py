""" json mode function calling """
import logging
import importlib
from zhihu.common.api import Session
from zhihu.common.util import print_json
from zhihu.function_call.common import make_func_tool, MODEL_FUNC_CALL
# A way to resolve the conflict name of this json.py file name
json = importlib.import_module("json")
logger = logging.getLogger(__name__)


def function_call_json(prompt, /, system_prompt, json_desc: dict):
    """ 
    This is to simulate the JSON mode using function calling
    :json_desc, a dict that describes the json structure
    """
    session = Session()
    tools = [
        make_func_tool(
            "json_parser", "这是一个由function calling实现的json mode parser", json_desc)
    ]
    if system_prompt is not None:
        session.set_system_prompt(system_prompt)
    rsp = session.get_completion(prompt, tools=tools, model=MODEL_FUNC_CALL)
    logger.info("rsp=%s", rsp)
    message_assistant = rsp.choices[0].message
    try:
        tool_call = message_assistant.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        print_json(args)
        return args
    except Exception as e:
        print("error", e)
