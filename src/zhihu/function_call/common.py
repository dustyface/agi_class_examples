import json
import sqlite3
import sys
import logging
from zhihu.common.util import print_json

logger = logging.getLogger(__name__)

# latest model that support function calling well
model_func_call = "gpt-3.5-turbo-0125"
# gpt-4 is expensive
# model_func_call = "gpt-4-turbo-preview"


def function_calling_cb(session, message, module_name:str, *callbacks):
    if message.tool_calls is None:
        logger.warn("No tool_calls in assistant message, there is no function calling to invoke; message=%s", message)
        return
    
    session.add_message(message=message)
    tool_calls = message.tool_calls
    for tool_call in tool_calls:
        if len(callbacks) == 0:
            raise ValueError("You should at least provide one callback function name")
        fn_name = tool_call.function.name
        logger.debug("fn_name=%s", fn_name)
        if len(callbacks) > 0 and fn_name not in callbacks:
            continue
        fn_args = json.loads(tool_call.function.arguments)
        logger.debug("fn_args=%s", fn_args)
        module = sys.modules[module_name]
        callback = getattr(module, fn_name)
        if callback:
            result = callback(**fn_args)
            logger.info("callback result=%s", result)
            session.add_message(message={
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": fn_name,
                "content": str(result)
            })
    rsp = session.get_completion(
        model=model_func_call,
        temperature=0.7,
        seed=1024
    )
    return rsp

def make_func_tool(name: str, description:str, /, properties: dict, required:list=None) -> dict:
    tool = {
        "type": "function",
        "function": {
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
    func = tool["function"]
    func["name"] = name
    func["description"] = description
    func_properties = func["parameters"]["properties"]
    for k, v in properties.items():
        if isinstance(v, str):
            func_properties[k] = {
                "type": "string",
                "description": v
            }
        elif isinstance(v, dict):
            func_properties[k] = {
                "type": v["type"],
            }
            del v["type"]
            for k1, v1 in v.items():
                func_properties[k][k1] = v1
    if required is not None and len(required) > 0:
        func["parameters"]["required"] = required
    return tool

# The base class for database operations
class DB:
    def __init__(self):
        self.conn = sqlite3.connect(':memory')
        self.cursor = self.conn.cursor()
    
    def exec_query(self, query:str):
        logger.info("executing query=%s", query)
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def __del__(self):
        self.cursor.close()
        self.conn.close()