import logging
from zhihu.common.api_ernie import Session

logger = logging.getLogger(__name__)

def test_chat_do():
    session = Session()
    session.set_system_prompt("你叫呱呱, 你是一个讲笑话的专家")
    rsp = session.chat("讲一个10个字以内的笑话", model="ERNIE-4.0-8K")
    logger.info("response=%s", rsp.body["result"])

def test_chat_do_with_multimessage():
    session = Session()
    session.set_system_prompt("你是一个地理通")
    # temperature can't be 0 in ERNIE model
    rsp = session.chat("北京有7环路吗?", model="ERNIE-4.0-8K", temperture=0.7, clear_session=False)
    rsp_content = rsp.body["result"]
    logger.info("response=%s", rsp_content)
    session.add_message(role="assistant", content=rsp_content)
    rsp = session.chat("我要去泡温泉, 告诉去小汤山温泉做公交车怎么走?", model="ERNIE-4.0-8K")
    logger.info("response=%s", rsp.body["result"])

def test_chat_do_with_only_message_param():
    session = Session(system_prompt="你是一个地理通")
    rsp = session.chat("北京有7环路吗?", model="ERNIE-4.0-8K", temperture=0.7, clear_session=False)
    logger.info("response=%s", rsp.body["result"])
    session.add_message(message={
        "role": "assistant",
        "content": rsp.body["result"]
    })
    session.add_message(role="user", content="请简单的告诉我, 我要去小汤山, 怎么坐公交车?")
    rsp = session.chat(model="ERNIE-4.0-8K")
    logger.info("response=%s", rsp.body["result"])