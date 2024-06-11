""" Customize your own ChatModel with the old funny guy """
import random
import re
import time
from typing import List, Optional, Iterator, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import (
    ChatResult, ChatGeneration, ChatGenerationChunk
)
from langchain_core.callbacks import CallbackManagerForLLMRun


class ChatModelOlder(BaseChatModel):
    """ ChatModel with the old funny guy """
    @property
    def _llm_type(self):
        """ return the type of the language model """
        return "chat with older man"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        generate the chat resul with old guy.
        the old guy will always return funny answers.
        """
        # ChatGeneration是一个pydantic model类, 它的root_validator
        # decorator会处理List[BaseMessage]的内容，把message中的content提取到ChatGeneration的text属性中
        # 在创建pydantic的子类时, 可以直接将某个field作为参数传入, pydantic的__init__()中, 设定为instance attribute;
        generations = [ChatGeneration(message=res)
                       for res in self._ask_remote(messages)]
        return ChatResult(generations=generations)

    def _ask_remote(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """ give the old guy answer """
        answers = [AIMessage(m) for m in [
            "马什么梅?",
            "什么冬梅?",
            "马东什么???"
        ]]

        if re.search("马冬梅", messages[0].content):
            response = answers[random.randint(0, 2)]
        else:
            response = AIMessage("哦...")
        return [response]

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """ customize the stream method """
        response = self._ask_remote(messages)
        # content的是string, 作为iterator，每次只访问它的一个元素；
        for chunk in response[0].content:
            time.sleep(0.1)
            yield ChatGenerationChunk(message=AIMessageChunk(chunk))
