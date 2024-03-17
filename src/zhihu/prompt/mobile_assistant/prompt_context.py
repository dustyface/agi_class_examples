from zhihu.prompt.mobile_assistant.prompt_text import ROLE, INSTRUCTION, EXAMPLES, OUTPUT_FORMAT
from zhihu.common.api import get_completion

CONTEXT_EXAMPLES=f"""
{EXAMPLES}

客服：有什么可以帮您
用户：100G套餐有什么

{{"data":{{"operator":">=","value":100}}}}

客服：有什么可以帮您
用户：100G以上的套餐有什么
客服：我们现在有无限套餐，不限流量，月费300元
用户：太贵了，有200元以内的不

{{"data":{{"operator":">=","value":100}},"price":{{"operator":"<=","value":200}}}}

客服：有什么可以帮您
用户：便宜的套餐有什么
客服：我们现在有经济套餐，每月50元，10G流量
用户：100G以上的有什么

{{"data":{{"operator":">=","value":100}},"sort":{{"ordering": "ascend","value"="price"}}}}

客服：有什么可以帮您
用户：100G以上的套餐有什么
客服：我们现在有畅游套餐，流量100G，月费180元
用户：流量最多的呢

{{"sort":{{"ordering": "descend","value"="data"}},"data":{{"operator":">=","value":100}}}}

"""

CONTEXT = """
客服: 有什么可以帮您
用户: 有什么10G以上的套餐
客服: 我们现在有畅游套餐和无限套餐，您有什么价格倾向？
用户: {user_input}
"""

# 这段Prompt体现了使用设定Prompt的6大要素:
# Role, Instruction, Context, Example, Input, Output
PROMPT_TEMPLATE = f"""
{ROLE}
{INSTRUCTION}

{OUTPUT_FORMAT}

{CONTEXT_EXAMPLES}
"""

def simple_conversation():
    robot_emoji = "\U0001F916"
    user_emoji = "\U0001F42E"
    user_input = input(f"\n{robot_emoji}: 你好, 请回答你的办卡需求, 我可以帮你选择合适的套餐\n{user_emoji}: ")
    context = CONTEXT.format(user_input=user_input)
    prompt = PROMPT_TEMPLATE + "\n\n" + context
    return get_completion(prompt)

# print(simple_conversation().choices[0].message.content)


