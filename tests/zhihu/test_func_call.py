""" Test Function Calling utility """
import logging
from zhihu.common.util import print_json
from zhihu.function_call.common import make_func_tool
from zhihu.function_call.sum import function_call_sum
from zhihu.function_call.eval import function_call_eval
from zhihu.function_call.geo import function_call_geo
from zhihu.function_call.json import function_call_json
from zhihu.function_call.query import analyze
from zhihu.function_call.multi_table_query import analyze as multi_table_analyze
from zhihu.function_call.stream import stream_func_call

logger = logging.getLogger(__name__)


def test_make_func_tool():
    """ Test generating the tool structure for OpenAI """
    tool_json = make_func_tool("search_nearby_pois", "根据给定经纬度坐标，搜索附近的POI", {
        "longitude": "中心点经度",
        "latitude": "中心点纬度",
        "keyword": "目标poi关键词"
    }, required=["longitude", "latitude"])
    print_json(tool_json)
    logger.info("tool_json=%s", tool_json)
    tool_json = make_func_tool("sum", "加法器, 计算一组数的和", {
        "numbers": {
            "type": "array",
            "items": {"type": "number"}
        }
    })
    print_json(tool_json)
    logger.info("tool_json=%s", tool_json)


def test_func_call_sum():
    """ Test for Function Calling sum """
    prompts = [
        "Tell me the sum of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
        "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？",
        "1+2+3...+99+100",
        "4.5乘以4.5是多少?",    # LLM的不靠谱的回答，等于9.0
        "太阳从那边升起?",
    ]
    for p in prompts:
        rsp = function_call_sum(p)
        if rsp is not None:
            logger.info("rsp=%s", rsp.choices[0].message.content)


def test_func_call_eval():
    """ Test Function Calling eval """
    prompts = [
        "1+2+3...+99+100的结果是多少?",
        "1+2+3...+99+100",
        "4.5 * 4.5",
        "3的平方根乘以2"
    ]
    for p in prompts:
        rsp = function_call_eval(p)
        logger.info("rsp=%s", rsp.choices[0].message.content)


def test_func_call_geo():
    """ Test Function Calling geo """
    prompts = [
        "我想在北京五道口附近喝咖啡，给我推荐几个",
        # "我想去小汤山泡温泉，告诉我怎么走",
        # "我想去天坛附近，喝豆汁吃北京小吃, 给我推荐几个地方"
    ]
    for p in prompts:
        final_message = function_call_geo(p)
        logger.info("final message=%s", final_message)
        print(final_message.content)


def test_func_call_json():
    """ Test Function Calling json mode """
    prompts = ["帮我寄给王卓然，地址是北京市朝阳区亮马桥外交办公大楼，电话13012345678。"]
    system_prompts = ["你是一个快递员，你要帮用户寄快递"]
    json_desc_list = [{
        "name": "联系人姓名",
        "address": "联系人地址",
        "tel": "联系人电话"
    }]
    for (i, (k, v)) in enumerate(zip(prompts, system_prompts)):
        j = function_call_json(k, v, json_desc=json_desc_list[i])
        logger.info("json=%s", j)


def test_func_call_dbquery():
    """ Test Function Calling DB query """
    prompts = [
        "10月份的销售额",
        "统计10月份每件商品的销售额"
    ]
    for p in prompts:
        rsp = analyze(p, "你是一个数据分析师，你可以查询数据库, 请基于order表回答用户问题")
        logger.info("rsp=%s", rsp.choices[0].message.content)


def test_func_call_multitable_query():
    """ Test Function Calling multitable query """
    system_prompt = "你是一个数据分析师，你可以查询数据库, 请基于order, product, customer等表回答用户问题"
    prompts = [
        "统计每月每件商品的销售额",
        "这星期消费最高的用户是谁? 他买了哪些商品？",
    ]
    for p in prompts:
        rsp = multi_table_analyze(p, system_prompt)
        if rsp:
            logger.info("rsp=%s", rsp.choices[0].message.content)


def test_stream_func_call():
    """ Test Function Calling stream option """
    prompts = [
        "1+2+3",
        "你是谁"
    ]
    for p in prompts:
        fn, args, text = stream_func_call(p)
        logger.info("fn=%s, args=%s, text=%s", fn, args, text)
