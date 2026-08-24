import requests
import pandas as pd
import time
import logging
import os
from typing import List, Dict, Any, Optional

# --- 配置部分 ---
#API_KEY = "c3ae1fc3acd0b8b01df5a82c42c6106c"  # key-zc
#API_KEY = "4407ee6c510c249a6c193db40ff19229"  # key-cy
#API_KEY = "f298283abbc3917b4a0da1aa71525656" #key-cying
API_KEY = "1dea4c078621f0aa4475423e95a55506" #key-zq
AD_CODE = "110119"  # 搜索区域:海淀区(110108) 朝阳区(110105) 东城区(110101) 西城区(110102) 丰台区(110106) 石景山区(110107) 门头沟区(110109) 房山区(110111) 通州区(110112) 顺义区(110113)
                    # 昌平区(110114) 大兴区(110115) 怀柔区(110116) 平谷区(110117) 密云区(110118) 延庆区(110119)
CITY_LIMIT = "true"  # 严格限制在区域内搜索
BASE_URL = "https://restapi.amap.com/v5/place/text"  # 搜索POI 2.0接口地址
PAGE_SIZE = 25  # 每页最大条数，POI 2.0限制为1-25
MAX_RETRIES = 3  # 请求失败时的最大重试次数
REQUEST_INTERVAL = 0.5  # 请求间隔（秒），用于控制QPS

poi_types_file = "data/poi_types_all.xlsx"
output_file = "output/延庆区_POI数据.xlsx"

# 配置日志，便于查看采集状态
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 核心函数 ---

def search_poi_by_type(typecode: str, page_num: int = 1) -> Optional[Dict[str, Any]]:
    """
    按单个最小类型编码搜索POI，并处理错误与重试。
    :param typecode: POI类型编码
    :param page_num: 要请求的页码
    :return: API返回的JSON数据，失败时返回None
    """
    params = {
        "key": API_KEY,
        "types": typecode,  # 使用poi类型编码进行精确搜索
        "region": AD_CODE,
        "city_limit": "true",  # 仅召回海淀区数据
        "page_size": 25,
        "page_num": page_num,
        "output": "json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()  # 非200状态码会抛出异常

            data = response.json()

            # 检查API返回的业务状态码
            if data.get('status') == '1':
                return data
            else:
                # 处理API返回的错误 [citation:9][citation:11]
                info_code = data.get('infocode')
                info_msg = data.get('info', '未知错误')
                logger.warning(f"API错误 (编码: {typecode}, 页: {page_num}): {info_code} - {info_msg}")

                # 限流错误，等待更长时间后重试 [citation:9][citation:10]
                if info_code in ['10044', 'USER_DAILY_QUERY_OVER_LIMIT', '10045', 'QPS_OVER_LIMIT']:
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"触发限流，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                # 其他业务错误，不再重试
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败 (编码: {typecode}, 页: {page_num}): {e}")
            time.sleep(2)  # 网络错误，短暂等待后重试
        except Exception as e:
            logger.error(f"未知错误 (编码: {typecode}, 页: {page_num}): {e}")
            time.sleep(2)

    logger.error(f"达到最大重试次数，放弃请求 (编码: {typecode}, 页: {page_num})")
    return None


def fetch_all_pois_for_type(typecode: str) -> List[Dict[str, Any]]:
    """
    为一个类型编码处理所有分页，直到没有更多数据。
    :param typecode: POI类型编码
    :return: 该类型下所有POI的列表
    """
    all_pois = []
    page = 1
    max_pages = 8
    total_pois = 0

    logger.info(f"开始采集类型编码: {typecode}")

    while page <= max_pages:
        # 限流控制
        time.sleep(REQUEST_INTERVAL)

        result = search_poi_by_type(typecode, page)
        if not result:
            # 如果某页请求失败，停止当前类型的所有采集
            logger.warning(f"类型 {typecode} 在第 {page} 页请求失败，终止采集")
            break

        pois = result.get('pois', [])
        count = int(result.get('count', 0))

        # 如果这一页没有数据，说明已翻完
        if not pois or count == 0:
            break

        total_pois += len(pois)
        all_pois.extend(pois)

        logger.info(f"类型 {typecode} 已获取 {len(pois)} 条 (累计 {total_pois} 条)，当前第 {page} 页")

        # 如果这一页数据小于25，说明已翻完
        if count < 25:
            break

        page += 1

    logger.info(f"类型 {typecode} 采集完成，总计 {len(all_pois)} 条")
    return all_pois


def parse_pois_to_dataframe(pois: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将POI数据解析为Pandas DataFrame，提取关键字段。
    :param pois: POI对象列表
    :return: DataFrame
    """
    if not pois:
        return pd.DataFrame()

    records = []
    for poi in pois:
        # 提取经纬度
        location = poi.get('location', '')
        lon, lat = '', ''
        if location and ',' in location:
            parts = location.split(',')
            if len(parts) == 2:
                lon, lat = parts[0], parts[1]

        record = {
            'POI ID': poi.get('id'),
            '名称': poi.get('name'),
            '类型': poi.get('type'),
            '类型编码': poi.get('typecode'),
            '经度': lon,
            '纬度': lat,
            '地址': poi.get('address'),
            '省份': poi.get('pname'),
            '城市': poi.get('cityname'),
            '区县': poi.get('adname'),
        }
        records.append(record)

    return pd.DataFrame(records)


def crawler_all_types(type_codes: list) -> pd.DataFrame:
    """
    爬取所有子类型的POI数据。
    :param type_codes: POI类型列表
    :return: DataFrame
    """
    all_pois = []
    # 1. 遍历所有类型编码进行采集
    for code in type_codes:
        code_str = code.zfill(6)
        pois_for_one_type = fetch_all_pois_for_type(code_str)
        all_pois.extend(pois_for_one_type)

        # 在大循环之间增加短暂休息，保护API
        time.sleep(REQUEST_INTERVAL)

    # 2. 数据解析与保存
    save_pois_to_excel(all_pois)

def crawler_one_type(type_code: str) -> pd.DataFrame:
    """
    爬取所有子类型的POI数据。
    :param type_code: POI类型
    :return: DataFrame
    """
    all_pois = fetch_all_pois_for_type(type_code)
    save_pois_to_excel(all_pois)


def save_pois_to_excel(all_pois:list):
    """
    保存POI数据到excel文件。
    :param all_pois: POI数据列表
    :return:
    """
    if all_pois:
        final_df = parse_pois_to_dataframe(all_pois)

        # 检查文件是否存在
        file_exists = os.path.exists(output_file)
        if not file_exists:
            final_df.to_excel(output_file, sheet_name='Sheet1', index=False)
        else:
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                final_df.to_excel(writer, sheet_name='Sheet1', index=False, header=False,
                             startrow=writer.sheets['Sheet1'].max_row)
        logger.info(f"数据采集完成！共获取 {len(final_df)} 条POI记录，已保存至 {output_file}")
    else:
        logger.warning("未采集到任何POI数据，请检查API Key和类型编码是否正确。")



# --- 主程序 ---
if __name__ == "__main__":
    # 1. 读取Excel分类编码表，提取"NEW_TYPE"列
    try:
        df_types = pd.read_excel(poi_types_file)
        # 假设"最小类型编码"是列名，请根据实际情况修改
        type_codes = df_types['NEW_TYPE'].astype(str).dropna().unique().tolist()
        logger.info(f"从Excel加载了 {len(type_codes)} 个类型编码")
    except FileNotFoundError:
        logger.error("未找到分类编码表文件，请检查文件路径。")
    except KeyError:
        logger.error("Excel文件缺少 'NEW_TYPE' 列，请检查列名。")
        exit()

    crawler_all_types(type_codes[0:])
    #crawler_one_type("010102")
