""" This module provides common functions for function calling """
import json
import sqlite3
import sys
import logging
from zhihu.common.api import Session

logger = logging.getLogger(__name__)

# latest model that support function calling well
MODEL_FUNC_CALL = "gpt-3.5-turbo-0125"
# gpt-4 is expensive
# MODEL_FUNC_CALL = "gpt-4-turbo-preview"


def function_calling_cb(session, message, module_name: str, *callbacks):
    """ This is the callback for the 2nd round of function calling """
    if message.tool_calls is None:
        logger.warning(
            """
            No tool_calls in assistant message,
            there is no function calling to invoke; message=%s
            """,
            message
        )
        return
    session.add_message(message=message)
    tool_calls = message.tool_calls
    for tool_call in tool_calls:
        if len(callbacks) == 0:
            raise ValueError(
                "You should at least provide one callback function name")
        fn_name = tool_call.function.name
        logger.info("fn_name=%s", fn_name)
        if len(callbacks) > 0 and fn_name not in callbacks:
            continue
        fn_args = json.loads(tool_call.function.arguments)
        logger.info("fn_args=%s", fn_args)
        module = sys.modules[module_name]
        callback = getattr(module, fn_name)
        if callback:
            result = callback(**fn_args)
            logger.info("callback result=%s", result)
            assist_msg = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": fn_name,
                "content": str(result)
            }
            session.add_message(message=assist_msg)
    rsp = session.get_completion(
        model=MODEL_FUNC_CALL,
        temperature=0.7,
        seed=1024,
        clear_session=False
    )
    return rsp


def make_func_tool(name: str, description: str, /, properties: dict, required: list = None) -> dict:
    """ Return the function calling tool structure """
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


class DB:
    """ The base class for SQLite DB operations """

    def __init__(self):
        self.conn = sqlite3.connect(':memory')
        self.cursor = self.conn.cursor()

    def exec_query(self, query: str):
        """ Execute the query and return the result"""
        logger.info("executing query=%s", query)
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def __del__(self):
        self.cursor.close()
        self.conn.close()


class DBAnalyzer:
    """ The class to analyze the SQL query"""

    def __init__(self, **kwargs):
        database_schema_string = kwargs.get("database_schema_string")
        print("DBAnalyzer init; database_schema_string=", database_schema_string)
        self.database_schema_string = database_schema_string

    def analyze(self, prompt: str, system_prompt: str, *, caller_module_name: str = None):
        """ Generate the SQL query by the LLM """
        session = Session()
        session.set_system_prompt(system_prompt)
        tools = [
            make_func_tool(
                "exec_query",
                """
                This is function is to answer user requirement about business.
                The output should be a fully formed SQL query statement.""",
                {
                    "query": f"""
                        SQL query extracting information to answer user's question.
                        SQL should be written using this database schema:
                        {self.database_schema_string}
                        The query should be returned in plain text, not in JSON.
                        The query should only contain grammars supported by SQLite.
                        """
                },
                required=["query"]
            )
        ]
        rsp = session.get_completion(
            prompt,
            model=MODEL_FUNC_CALL,
            tools=tools,
            seed=1024,
            clear_session=False
        )
        message = rsp.choices[0].message
        logger.info("assistant message=%s", message)
        return function_calling_cb(
            session,
            message,
            caller_module_name if caller_module_name is not None else __name__,
            "exec_query"
        )
