from zhihu.common.api import set_system_prompt, get_completion, add_message_to_session 
import logging
import os

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

system_prompt = """
你叫呱呱，是一个手机流量套餐的客服代表。
可以帮助用户选择最合适的流量套餐产品。可以选择的套餐包括:
经济套餐，月费50元，10G流量；
畅游套餐，月费180元，100G流量；
无限套餐，月费300元，1000G流量；
校园套餐，月费150元，200G流量，仅限在校生
"""

# 这个例子, 创造了一个手机流量客服套餐助手的情景, 直接和LLM对话, 
# 从输出可以看出, 这种prompt控制方式和LLM的交互，LLM的回复的内容随机性是非常大的, 在和client端的交互可控性比较差
def multiround_conversation():
    set_system_prompt(system_prompt)
    response = get_completion("没有有土豪套餐?", clear_session=False)
    logger.info("response_0=%s", response.choices[0].message.content)
    add_message_to_session("assistant", response.choices[0].message.content)
    response = get_completion("多少钱?", clear_session=False)
    logger.info("response_1=%s", response.choices[0].message.content)
    add_message_to_session("assistant", response.choices[0].message.content)
    response = get_completion("给我办一个")
    logger.info("reponse_2=%s", response.choices[0].message.content)

if __name__ == "__main__":
    multiround_conversation()