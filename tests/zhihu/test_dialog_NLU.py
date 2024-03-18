from zhihu.prompt.NLU.dialog_NLU import create_prompt_template, DialogManager, test_dialog
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_prompt_templates():
    data = {
        "month_price": 50,
        "name": "畅游套餐",
        "month_data": 200
    }
    user_request = {
        "month_price": {"operator": ">=", "value": 100}
    }
    input = "给我个土豪套餐"
    dm = DialogManager(create_prompt_template())
    output = dm._wrap(input, [data])
    logger.info("output_1=%s", output)

    dm.state = dm.dst.update({}, user_request)
    output = dm._wrap(input, [])
    logger.info("output_2=%s", output)


def test_dialogmgr_run():
    testcase =[
        "给我个便宜的套餐",
        "有没有不限流量的套餐",
        "我要200元以内的套餐",
        "给我个土豪套餐"
    ]
    for case in testcase:
        test_dialog(case)