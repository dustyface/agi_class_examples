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

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

def set_system_prompt(prompt) -> list[dict]:
    system_message = []
    system_message.append({
        "role": "system",
        "content": prompt
    });
    return system_message
    

def get_completion(prompt, /, model="gpt-3.5-turbo", temperature=0, *, system_prompt: Union[str, List[Dict]]=None, **kwargs):
    args = {}
    system_prompt = set_system_prompt(system_prompt) if system_prompt is str else system_prompt
    messages = [] if system_prompt is None else (
        [{
        "role": "system",
        "content": system_prompt
        }]
        if isinstance(system_prompt, str) else system_prompt
    )
    messages.append({
        "role": "user",
        "content": prompt
    })
    args = {k: v for k, v in kwargs.items() if v is not None}
    args['model'] = model
    args['temperature'] = temperature
    args['messages'] = messages
    logger.info("get_completion args=%s", args)
    rsp = client.chat.completions.create(**args)
    logger.debug("get_completion response=%s", rsp)
    return rsp