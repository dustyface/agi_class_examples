""" Call the function calling using an option stream """
import logging
import sys
from zhihu.common.api import Session
from zhihu.function_call.common import MODEL_FUNC_CALL, make_func_tool

logger = logging.getLogger(__name__)


def collect_stream(response):
    """ handling the stream response """
    function_name, args, text = "", "", ""
    for msg in response:
        delta = msg.choices[0].delta
        if delta.tool_calls:
            function_name = delta.tool_calls[0].function.name
            args_delta = delta.tool_calls[0].function.arguments
            args = args + args_delta
            sys.stdout.write(args_delta)
        elif delta.content:
            text += delta.content
            sys.stdout.write(delta.content)
    # 注意, connet stream to stdout, use sys.stdout.write() instead of print()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return function_name, args, text


def stream_func_call(prompt):
    """ Call the function calling using an option stream """
    session = Session()
    tools = [
        make_func_tool("sum", "计算一组数的和", {
            "numbers": {
                "type": "array",
                "items": {
                    "type": "number"
                }
            }
        })
    ]
    rsp = session.get_completion(
        prompt,
        model=MODEL_FUNC_CALL,
        tools=tools,
        temperature=0.7,
        seed=1024,
        stream=True
    )
    fn_name, args, text = collect_stream(rsp)
    return fn_name, args, text
