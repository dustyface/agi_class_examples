import logging
import zhihu.common.api as api

logger = logging.getLogger(__name__)


def test_set_system_prompt():
    sys_msg = api.set_system_prompt("你叫瓜瓜, 你是一个讲笑话的专家")
    assert isinstance(sys_msg, list)
    

def test_get_completion():
    # 测试基本功能
    rsp = api.get_completion("讲一个10个字以内的笑话")
    logger.info("rsp1=%s", rsp.choices[0].message.content)
    # 测试使用system_prompt
    sys_msg = api.set_system_prompt("你叫瓜瓜, 你是一个讲笑话的专家")
    rsp = api.get_completion("讲一个10个字以内的笑话", system_prompt=sys_msg)
    logger.info("rsp2=%s", rsp.choices[0].message.content)
    # 测试使用system_prompt字符串 & 其他参数
    rsp = api.get_completion("讲一个10个字以内的笑话", temperature=0.5, system_prompt="这次你叫嘎嘎, 你是一个相声演员")
    logger.info("rsp3=%s", rsp.choices[0].message.content)
