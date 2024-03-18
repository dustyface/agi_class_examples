import logging
import zhihu.common.api as api
from zhihu.common.api import set_system_prompt, get_completion, get_session_messages, reset_session_message, add_message_to_session

logger = logging.getLogger(__name__)

def test_set_system_prompt():
    sys_msg = set_system_prompt("你叫瓜瓜, 你是一个讲笑话的专家")
    logger.info("sys_msg=%s", sys_msg)
    assert sys_msg == get_session_messages()
    assert sys_msg[0]["role"] == "system"


def test_get_completion():
    # 测试基本功能
    rsp = get_completion("讲一个10个字以内的笑话")
    logger.info("response_1=%s", rsp.choices[0].message.content)
    # 测试使用system_prompt
    # 设定了set_system_prompt(), 再次调用get_completion()时, 会自动使用system_prompt
    set_system_prompt("你叫瓜瓜, 你是一个讲笑话的专家")
    rsp = get_completion("讲一个10个字以内的笑话", clear_session=False)
    session_msg = get_session_messages()
    assert len(session_msg) == 2
    assert session_msg[0]['role'] == "system"
    logger.info("response_2=%s", rsp.choices[0].message.content)

def test_get_completion_with_system_prompt():
    # 测试使用system_prompt字符串 & 其他参数
    # 在调用get_completion(), 如果希望_session_message是空的, 则需要调用reset_session_message()
    reset_session_message()
    rsp = get_completion("讲一个10个字以内的笑话", temperature=0.5, system_prompt="这次你叫嘎嘎, 你是一个相声演员")
    logger.info("response_3=%s", rsp.choices[0].message.content)

def test_get_completion_with_multimessage():
    # 为了确保get_completion()的成功, 应该先清空_session_messages
    reset_session_message()
    rsp = get_completion("北京有7环路吗?", temperature=0.4, system_prompt="你是一个地理通。", clear_session=False)
    rsp_content = rsp.choices[0].message.content
    logger.info("response_4=%s", rsp_content)
    add_message_to_session("assistant", rsp_content)
    rsp = get_completion("我要去泡温泉, 告诉去小汤山温泉怎么走?")
    logger.info("response_5=%s", rsp.choices[0].message.content)
