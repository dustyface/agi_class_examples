""" Using NLU to control the conversation flow """
import json
import logging
from zhihu.common.api import Session
from zhihu.prompt.prompt_text import ROLE, INSTRUCTION, EXAMPLES, OUTPUT_FORMAT

logger = logging.getLogger(__name__)


def create_prompt_template():
    """ Create prompt template for the second round conversation """
    prompt_templates = {
        "recommand": """
            用户说: __INPUT__\n\n向用户介绍如下产品:__NAME__, 月费__MONTH_PRICE__元, 每月流量__MONTH_DATA__G
        """,
        "not_found": "用户说: __INPUT__\n\n没有找到满足__MONTH_PRICE__元价位__MONTH_DATA__G流量的产品，询问用户是否有其他倾向。"
    }
    extra_constraint = "尽量口语一些，亲切一些；不用说'抱歉'; 直接给出回答，不要在前面加'小瓜说'; NO COMMENT. NO ACKNOWLEDGEMENTS"
    prompt_templates = {k: v + extra_constraint for k,
                        v in prompt_templates.items()}
    return prompt_templates


class NLU:
    """
    NLU: Natural Language Understanding
    - NLU 处理第一轮user_input, 从中抽取出表示用户意图的json结构信息
    - NLU => (json) => DST
    """

    def __init__(self):
        self.prompt_template = f"""
        {ROLE}

        {INSTRUCTION}

        {OUTPUT_FORMAT}

        {EXAMPLES}

        用户输入: __INPUT__
        """
        self.session = Session()

    def get_completion(self, prompt, model="gpt-3.5-turbo"):
        """ Get completion from OpenAI """
        rsp = self.session.get_completion(prompt, model=model, temperature=0)
        rsp_json = json.loads(rsp.choices[0].message.content)
        return {k: v for k, v in rsp_json.items() if v}

    # 和LLM的第一轮对话，获得用户意图中的json结构信息;
    def parse(self, user_input):
        """ Parse user input to json structure """
        prompt = self.prompt_template.replace("__INPUT__", user_input)
        return self.get_completion(prompt)


class DST:
    """
    DST: Dialogue State Tracking
    - 目前DST update只是做一些数据的清洗/转换工作; i.e. 如果name出现在从NLU传过来的json中，则清除原有state中的信息(可能意味着换了一个套餐);
    根据sort信息清洗，etc
    - 目前以上两种清洗，在很多json中都没有用到, 基本上成了透传;

    NLU => (json) => DST => (json) => DB
    """

    def update(self, state, nlu_json):
        """ Update the state with nlu_json """
        def _clear():
            if "name" in state:
                state.clear()
            if "sort" in nlu_json:
                # the sorted field, e.g. month_data, month_price
                slot = nlu_json["sort"]["value"]
                if slot in state:
                    del state[slot]
        _clear()
        for k, v in nlu_json.items():
            state[k] = v
        return state


class MockedDB:
    """
    MockedDB

    - 可以被视为Policy处理的部分；
    - retrieve方法依据DST吐出的json信息，在MockedDB.data中检索符合条件的套餐，把符合的item加进records; 
    - 核心的比较是 `eval(str(r[k]) + v["operator"] + str(v["value"])):`这句, 动态的计算是否满足条件
    """

    def __init__(self):
        # mocked data, 暂时用class的member标识db data
        self.data = [
            {"name": "经济套餐", "month_price": 50,
                "month_data": 10, "requirement": None},
            {"name": "畅游套餐", "month_price": 180,
                "month_data": 100, "requirement": None},
            {"name": "无限套餐", "month_price": 300,
                "month_data": 1000, "requirement": None},
            {"name": "校园套餐", "month_price": 150,
                "month_data": 200, "requirement": "在校生"},
        ]

    def retrieve(self, **kwargs):
        """ Retrieve the records from MockedDB """
        records = []

        def _meet_requirement(item, args):
            # 注意，在目前的instructon,output_format的设定中, output jsond的输出没有设定status字段
            return "status" in args and args["status"] == item["requirement"]

        def _is_not_unlimited_set(item):
            return item["month_data"] != 1000

        def _meet_operator_relation(item, key, value):
            if key == "month_data" and value["value"] == "无上限":
                # pylint: disable=eval-used
                return eval(str(item[key]) + value["operator"] + "1000")
            # pylint: disable=eval-used
            return eval(str(item[key]) + value["operator"] + str(value["value"]))

        def _sort_records():
            key = "month_price"
            reverse = False
            if "sort" in kwargs:
                key = kwargs["sort"]["value"]
                reverse = kwargs["sort"]["ordering"] == "descend"
            return sorted(records, key=lambda x: x[key], reverse=reverse)

        for r in self.data:
            select = True
            # 排除requirements不符合的那一条记录
            if r["requirement"]:
                if not _meet_requirement(r, kwargs):
                    continue
            for k, v in kwargs.items():
                if k == "sort":
                    continue
                if k == "month_data" and v["value"] == "无上限":
                    if _is_not_unlimited_set(r):
                        select = False
                        break
                if "operator" in v:
                    if not _meet_operator_relation(r, k, v):
                        select = False
                        break
                elif str(r[k]) != str(v):
                    select = False
                    break
            if select:
                records.append(r)
        if len(records) <= 1:
            return records
        # 如果找到的信息大于1条, 则根据sort字段进行排序
        return _sort_records()


class DialogManager:
    """
    DialogManager: 管理了2轮用户和LLM的对话
    - constructor接受的参数prompt_templates是第2轮user询问LLM的模板,
      其中需要加入第一轮的用户输入和LLM回答的信息(从MockDB retrieve()得来的)
    - wrap_2nd_prompt, 是将第一轮的信息，加入到第二轮的模板中，形成第二轮的用户输入;
    - run, 执行第2轮对话;
    """

    def __init__(self, prompt_templates):
        self.state = {}
        self.messages = [
            {"role": "system", "content": "你是一个手机流量套餐的客服代表，你叫小瓜。可以帮助用户选择最合适的流量套餐产品。"}
        ]
        self.session = Session()
        self.nlu = NLU()
        self.dst = DST()
        self.db = MockedDB()
        # 第2轮对话的模版
        self.prompt_templates = prompt_templates

    def wrap_2nd_prompt(self, user_input, records):
        """ formulate the prompt for 2nd round """
        if len(records) > 0:
            prompt = self.prompt_templates["recommand"].replace(
                "__INPUT__", user_input)
            # 只取records(LLM回答的dictionary中的第一个元素)
            r = records[0]
            for k, v in r.items():
                prompt = prompt.replace(f"__{k.upper()}__", str(v))
        else:
            prompt = self.prompt_templates["not_found"].replace(
                "__INPUT__", user_input)
            for k, v in self.state.items():
                if "operator" in v:
                    prompt = prompt.replace(
                        f"__{k.upper()}__", v["operator"] + str(v["value"]))
                else:
                    prompt = prompt.replace(f"__{k.upper()}__", str(v))
        return prompt

    def get_completion(self, prompt, model="gpt-3.5-turbo"):
        """ Get completion from OpenAI """
        rsp = self.session.get_completion(
            prompt,
            model=model,
            messages=self.messages,
            temperature=0
        )
        return rsp.choices[0].message.content

    def run(self, user_input) -> str:
        """ Run the dialog """
        # 第一轮user对话LLM, nlu_json是LLM的回答
        nlu_json = self.nlu.parse(user_input)
        logger.info("nlu_json=%s", nlu_json)
        # 从第一轮对话LLM回答中，抽取出json结构信息，更新到state中;
        self.state = self.dst.update(self.state, nlu_json)
        # 从MockedDB中检索出符合条件的records
        records = self.db.retrieve(**self.state)
        # 构造第2轮对话的prompt, 和LLM对话
        prompt = self.wrap_2nd_prompt(user_input, records)
        response = self.get_completion(prompt)
        # 更新 self.session status
        self.session.add_message(role="assistant", content=response)
        return self.session.get_session_messages()


# NLU这个例子，实际上是使用了2个单轮的LLM的对话, 来得到用户需要的推荐结果
# NLU, 作用是处理用户最初的需求输入, 预置了prompt抽取原始需求为json结构的消息;
# DST, 主要是对用户需求对应的json结构进行一些转换和清洗
# MockedDB, 是一个模拟的数据库, 用于存储和检索用户需求对应的产品信息
# DialogManager, 整合2轮流程, 得到用户需求对应的信息json, 在db中检索出符合条件的记录, 和LLM进行第2轮对话，提示结果。
# (注意, 替换recommand prompt template的 wrap_2nd_prompt()时候，已经得到答案了, 第2轮对话甚至是可省略的)
# NLU => DST => DB => DialogManager
# 从这个例子，可以看到, 使用NLU+DST的流程, 用流程和prompt限制LLM的回答，使得数据格式更准确，更可控
def test_dialog(user_input):
    """ Test the dialog flow """
    prompt_templates = create_prompt_template()
    dm = DialogManager(prompt_templates)
    response = dm.run(user_input)
    logger.info("手机助手小瓜: %s", response)
