import logging
# from zhihu.common.api import set_system_prompt, get_completion, get_session_messages, reset_session_message, add_message_to_session
from zhihu.common.api import Session, client

logger = logging.getLogger(__name__)

def test_model_list():
    model_list = client.models.list()
    for m in model_list:
        logger.info("model=%s", m.id)


def test_set_system_prompt():
    session = Session()
    msg = session.set_system_prompt("你叫呱呱, 你是一个讲笑话的专家")
    logger.info("sys_msg=%s", msg)
    assert msg == session.get_session_messages()
    assert msg[0]["role"] == "system"


def test_get_completion():
    # 测试基本功能
    session = Session(system_prompt="你叫呱呱, 你是一个讲笑话的专家")
    rsp = session.get_completion("讲一个10个字以内的笑话")
    logger.info("response_1=%s", rsp.choices[0].message.content)
    # 测试再次对话
    session.set_system_prompt("你叫嘎嘎, 你现在对对联的专家")
    rsp = session.get_completion("不听老人言的上一句是什么?", clear_session=False)
    session_msg = session.get_session_messages()
    assert len(session_msg) == 2
    assert session_msg[0]['role'] == "system"
    logger.info("response_2=%s", rsp.choices[0].message.content)


def test_get_completion_with_multimessage():
    # 为了确保get_completion()的成功, 应该先清空_session_messages
    session = Session()
    session.set_system_prompt("你是一个地理通")
    rsp = session.get_completion("北京有7环路吗?", temperature=0, clear_session=False)
    rsp_content = rsp.choices[0].message.content
    logger.info("response_4=%s", rsp_content)
    session.add_message(role="assistant", content=rsp_content)
    rsp = session.get_completion("我要去泡温泉, 告诉去小汤山温泉做公交车怎么走?")
    logger.info("response_5=%s", rsp.choices[0].message.content)

def test_get_completion_with_only_message_param():
    session = Session(system_prompt="你是一个地理通")
    rsp = session.get_completion("北京有7环路吗?", temperature=0.4,  clear_session=False)
    logger.info("response_6=%s", rsp.choices[0].message.content)
    # 测试支持Function calling，需要直接把openai返回的ChatCompletionMessage作为message加入session
    session.add_message(message=rsp.choices[0].message)
    session.add_message(role="user", content="请简单的告诉我, 我要去小汤山, 怎么坐公交车?")
    rsp = session.get_completion()
    logger.info("response_7=%s", rsp.choices[0].message.content)