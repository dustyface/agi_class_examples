from dotenv import load_dotenv, find_dotenv
import qianfan
import logging

logger = logging.getLogger(__name__)
_ = load_dotenv(find_dotenv())

chat_comp = qianfan.ChatCompletion()

class Session:
    def __init__(self, messages: list[dict]=None, system_prompt:str=None):
        self.system_prompt = system_prompt
        self._dict_args = {}
        if self.system_prompt is not None:
            self.set_system_prompt(self.system_prompt)
        self._session_message = messages if messages is not None else []
    
    def set_system_prompt(self, prompt:str):
        self._dict_args["system"] = prompt
    
    def add_message(self, *, role:str=None, content:str=None, message=None):
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
        return self._session_message
    
    def reset_session_message(self):
        self._session_message.clear()

    def chat(self, prompt:str=None, /, model="ERNIE-Bot-turbo", temperture=0.7, *, clear_session:bool=True, **kwargs):
        args = {k: v for k, v in kwargs.items() if v is not None}
        args["model"] = model
        args["temperature"] = temperture
        for k, v in self._dict_args.items():
            args[k] = v
        if prompt is not None:
            self.add_message(role="user", content=prompt)
        if len(self._session_message) == 0:
            logger.warn("You should provide at least one message to start a session talk to LLM")
            return
        args["messages"] = self._session_message
        logger.debug("chat.do args=%s", args)
        rsp = chat_comp.do(**args)
        logger.debug("chat do response=%s", rsp)
        if clear_session:
            self.reset_session_message()
        return rsp

