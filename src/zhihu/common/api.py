import os
import logging
from typing import Union, Dict, List
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 不知道为何原因, logging的设定针对vscode的右侧topbar的run button不能生效
# 但在vscode Debug mode 调试bar run button可以生效
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_session_messages = []

# 返回_session_messages
def get_session_messages():
    return _session_messages

# 把特定role, content的消息加入session
def add_message_to_session(role:str, content:str) -> list[dict]:
    _session_messages.append({
        "role": role,
        "content": content
    })
    return _session_messages

# 设定role为system的消息
def set_system_prompt(prompt:str) -> list[dict]:
    if len(_session_messages) > 0:
        raise ValueError("When setting system_prompt, session_messages should be empty")
    return add_message_to_session("system", prompt)

# 清空_session_message
def reset_session_message():
    _session_messages.clear()

# 和LLM的会话
def get_completion(prompt, /, model="gpt-3.5-turbo", temperature=0, *, system_prompt:str=None, messages: list[object]=None, clear_session:bool=True,  **kwargs):
    args = {}
    messages = _session_messages if messages is None else messages
    if system_prompt is not None and len(messages) >= 1:
        raise ValueError("system_prompt should not be set, when messages is not empty")
    messages = set_system_prompt(system_prompt) if system_prompt is str and len(messages) == 0 else messages
    messages.append({
        "role": "user",
        "content": prompt
    })
    args = {k: v for k, v in kwargs.items() if v is not None}
    args['model'] = model
    args['temperature'] = temperature
    args['messages'] = messages
    logger.debug("get_completion args=%s", args)
    rsp = client.chat.completions.create(**args)
    logger.debug("get_completion response=%s", rsp)
    if clear_session:
        reset_session_message()
    return rsp