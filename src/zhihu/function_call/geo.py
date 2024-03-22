from dotenv import load_dotenv, find_dotenv
import os
import logging
import requests
from zhihu.function_call.common import make_func_tool, model_func_call, function_calling_cb
from zhihu.common.api import Session
from zhihu.common.util import print_json

_ = load_dotenv(find_dotenv())
amap_key = os.getenv("AMAP_KEY")

logger = logging.getLogger(__name__)

def get_location_coordinate(location="中国", city="北京"):
    print("get_location_coordinate()=", location, city, amap_key)
    if amap_key is None:
        raise ValueError("AMAP_KEY is not set")
    api_url = f"https://restapi.amap.com/v5/place/text?key={amap_key}&keywords={location}&region={city}"
    print("amap api_url=", api_url)
    r = requests.get(api_url)
    result = r.json()
    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None

def search_nearby_pois(longitude, latitude, keyword):
    print("search_nearby_pois()=", longitude, latitude, keyword, amap_key)
    if amap_key is None:
        raise ValueError("AMAP_KEY is not set")
    api_url = f"https://restapi.amap.com/v5/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    print("amap api_url=", api_url)
    r = requests.get(api_url)
    result = r.json()
    ans = ""
    if "pois" in result and result["pois"]:
        for i in range(min(3, len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n{address}\n距离: {distance}米\n\n"
    return ans

def function_call_geo(prompt:str):
    tools = [
        make_func_tool("search_nearby_pois", "根据给定经纬度坐标，搜索附近的POI", {
            "longitude": "中心点经度",
            "latitude": "中心点纬度",
            "keyword": "目标poi关键词"
        }, required=["longitude", "latitude"]),
        make_func_tool("get_location_coordinate", "根据POI名称，获得POI的经纬度坐标", {
            "location": "POI名称, 必须是中文",
            "city": "POI所在城市名称, 必须是中文"
        }, required=["location", "city"])
    ]
    session = Session(system_prompt="你是一个地图通，你可以找到任何地址")
    rsp = session.get_completion(prompt, tools=tools, model=model_func_call, seed=1024, clear_session=False)
    message_assistant = rsp.choices[0].message
    logger.info("message_assistant=%s", message_assistant)

    while message_assistant.tool_calls is not None:
        rsp = function_calling_cb(session, message_assistant, __name__,  "get_location_coordinate", "search_nearby_pois")
        if rsp.choices is not None:
            message_assistant = rsp.choices[0].message
        else:
            break
    
    logger.info("session message:\n\n%s", session.get_session_messages())
    return message_assistant

