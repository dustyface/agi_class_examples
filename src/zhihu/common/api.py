""" OpenAI API Wrapper """
import os
import logging
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


class Session:
    """ OpenAI Chat Session """

    def __init__(self, messages: list[dict] = None, system_prompt: str = None):
        if system_prompt is not None and messages is not None and len(messages) >= 1:
            raise ValueError(
                """
                Check your messages and system_prompt status,
                you should use either one to start a new session.
                """
            )
        self.system_prompt = system_prompt
        self._session_message = messages if messages is not None else []
        self.set_system_prompt(self.system_prompt)

    def set_system_prompt(self, prompt: str) -> list[dict]:
        """ Set the system prompt as the LLM model's role """
        if len(self._session_message) > 0:
            raise ValueError(
                "When setting system_prompt, session_messages should be empty")
        if prompt is not None:
            self.add_message(role="system", content=prompt)
        return self._session_message

    def add_message(self, *, role: str = None, content: str = None, message=None):
        """ Add message to the session """
        if message is not None and (role is not None or content is not None):
            raise ValueError("you should use either role&content or message")
        if message is not None:
            self._session_message.append(message)
        elif role is not None and content is not None:
            self._session_message.append({
                "role": role,
                "content": content
            })
        return self._session_message

    def get_session_messages(self):
        """ Get the session messages """
        return self._session_message

    def reset_session_message(self):
        """ Reset the session message """
        self._session_message.clear()

    def get_completion(
            self,
            prompt=None,
            /,
            model="gpt-3.5-turbo",
            temperature=0,
            *,
            clear_session: bool = True,
            **kwargs):
        """ Visit the OpenAI API to get the completion """
        args = {}
        args = {k: v for k, v in kwargs.items() if v is not None}
        args['model'] = model
        args['temperature'] = temperature
        if prompt is not None:
            self.add_message(role="user", content=prompt)
        if len(self._session_message) == 0:
            logger.warning(
                "You should provide at least one message to start a session talk to LLM")
            return
        args['messages'] = self._session_message
        logger.debug("get_completion args=%s", args)
        rsp = client.chat.completions.create(**args)
        logger.debug("get_completion response=%s", rsp)
        if clear_session:
            self.reset_session_message()
        return rsp
