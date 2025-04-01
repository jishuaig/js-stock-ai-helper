from ast import main
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import json


# 定义工具函数
def get_stock_data_func(stock_code: str) -> str:
    """
    获取股票实时数据
    :param stock_code: 股票代码，例如：sh600036
    :return: JSON格式的股票数据
    """
    try:
        # 处理股票代码
        if not stock_code.startswith(('sh', 'sz')):
            return json.dumps({"error": "不支持的股票代码格式"}, ensure_ascii=False)
            
        # 获取A股实时数据
        stock_data = ak.stock_zh_a_spot_em()
        if stock_data is None or stock_data.empty:
            return json.dumps({"error": "获取股票数据失败：数据为空"}, ensure_ascii=False)
            
        # 提取股票代码（去掉前缀）
        code = stock_code[2:]
        
        # 过滤对应股票的数据
        filtered_data = stock_data[stock_data['代码'] == code]
        if filtered_data.empty:
            return json.dumps({"error": f"未找到股票代码 {stock_code} 的数据"}, ensure_ascii=False)
            
        # 获取最新一条数据
        latest_data = filtered_data.iloc[0]
        
        # 构建返回数据
        result = {
            "股票代码": stock_code,
            "股票名称": latest_data.get("名称", ""),
            "最新价": float(latest_data.get("最新价", 0)),
            "涨跌幅": float(latest_data.get("涨跌幅", 0)),
            "涨跌额": float(latest_data.get("涨跌额", 0)),
            "成交量": float(latest_data.get("成交量", 0)),
            "成交额": float(latest_data.get("成交额", 0)),
            "振幅": float(latest_data.get("振幅", 0)),
            "最高": float(latest_data.get("最高", 0)),
            "最低": float(latest_data.get("最低", 0)),
            "今开": float(latest_data.get("今开", 0)),
            "昨收": float(latest_data.get("昨收", 0)),
            "量比": float(latest_data.get("量比", 0)),
            "换手率": float(latest_data.get("换手率", 0)),
            "市盈率": float(latest_data.get("市盈率-动态", 0)),
            "市净率": float(latest_data.get("市净率", 0)),
            "总市值": float(latest_data.get("总市值", 0)),
            "流通市值": float(latest_data.get("流通市值", 0)),
            "涨速": float(latest_data.get("涨速", 0)),
            "5分钟涨跌": float(latest_data.get("5分钟涨跌", 0)),
            "60日涨跌幅": float(latest_data.get("60日涨跌幅", 0)),
            "年初至今涨跌幅": float(latest_data.get("年初至今涨跌幅", 0))
        }
        
        return json.dumps(result, ensure_ascii=False)
            
    except Exception as e:
        print(f"获取股票数据出错: {str(e)}")
        return json.dumps({"error": f"获取股票数据失败: {str(e)}"}, ensure_ascii=False)

def get_historical_data_func(stock_code: str, days: int = 5) -> str:
    """获取指定股票的历史K线数据，默认5天"""
    try:
        if stock_code.startswith('sh'):
            code = stock_code[2:]
            data = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                      start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'), adjust="qfq")
            return data.to_json(orient='records', force_ascii=False)
        elif stock_code.startswith('sz'):
            code = stock_code[2:]
            data = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                      start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'), adjust="qfq")
            return data.to_json(orient='records', force_ascii=False)
        elif stock_code == 'sh000001':
            # 获取上证指数历史数据
            data = ak.stock_zh_index_daily(symbol="sh000001")
            return data.tail(days).to_json(orient='records', force_ascii=False)
        return json.dumps({"error": "不支持的股票代码"})
    except Exception as e:
        print(f"获取历史数据出错: {str(e)}")
        return json.dumps({"error": str(e)})
    
def get_week_historical_data_func(stock_code: str, weeks: int = 5) -> str:
    """获取指定股票的历史周K线数据，默认5周"""
    try:
        # 处理股票代码如果是sh或者sz大小写开头，，则去掉前缀
        if stock_code.startswith(('sh', 'sz', 'SH', 'SZ')):
            stock_code = stock_code[2:]
        
        # 获取周K线数据
        data = ak.stock_zh_a_hist(symbol=stock_code, period="weekly", 
                                  start_date=(datetime.now() - timedelta(weeks=weeks)).strftime('%Y%m%d'),
                                  end_date=datetime.now().strftime('%Y%m%d'), adjust="qfq")
        
        if data is None or data.empty:
            return json.dumps({"error": "获取周K线数据失败：数据为空"}, ensure_ascii=False)
            
        # 确保所有必需的列都存在
        required_columns = ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', 
                          '振幅', '涨跌幅', '涨跌额', '换手率']
        
        # 创建新的DataFrame，只包含需要的列
        result_data = pd.DataFrame()
        
        result_data['日期'] = data['日期']

        # 添加日期列并格式化为yyyyMMdd
        result_data['日期格式化'] = pd.to_datetime(data['日期']).dt.strftime('%Y%m%d')
        
        # 添加股票代码列
        result_data['股票代码'] = stock_code
        
        # 添加其他列
        for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']:
            if col in data.columns:
                result_data[col] = data[col]
            else:
                result_data[col] = 0
        
        return result_data.to_json(orient='records', force_ascii=False)
            
    except Exception as e:
        print(f"获取周K线数据出错: {str(e)}")
        return json.dumps({"error": f"获取周K线数据失败: {str(e)}"}, ensure_ascii=False)

def calculate_technical_indicators_func(historical_data_json: str) -> str:
    """计算常用技术指标"""
    try:
        # 将JSON字符串转换为DataFrame，避免FutureWarning
        import io
        historical_data = pd.read_json(io.StringIO(historical_data_json), orient='records')
        
        if historical_data.empty:
            return json.dumps({"error": "没有历史数据可用于计算指标"}, ensure_ascii=False)
            
        # 按日期排序，确保顺序正确
        if '日期' in historical_data.columns:
            historical_data = historical_data.sort_values(by='日期')
        
        # 确保数据足够计算指标
        if len(historical_data) < 30:
            return json.dumps({"error": "历史数据不足，无法计算准确的技术指标，需要至少30天数据"})
        
        # 确保列名一致
        price_col = '收盘' if '收盘' in historical_data.columns else 'close'
        high_col = '最高' if '最高' in historical_data.columns else 'high'
        low_col = '最低' if '最低' in historical_data.columns else 'low'
        volume_col = '成交量' if '成交量' in historical_data.columns else 'volume'
        
        # 验证关键列存在
        for col in [price_col, high_col, low_col]:
            if col not in historical_data.columns:
                return json.dumps({"error": f"数据中缺少必要的列：{col}"})
        
        # 计算RSI (相对强弱指标) - 默认使用14日周期
        delta = historical_data[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, 0.001)  # 避免除零错误
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 计算MACD - 默认参数：快线12，慢线26，信号线9
        exp1 = historical_data[price_col].ewm(span=12, adjust=False).mean()
        exp2 = historical_data[price_col].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        # 计算移动平均线
        ma5 = historical_data[price_col].rolling(window=5, min_periods=1).mean()
        ma10 = historical_data[price_col].rolling(window=10, min_periods=1).mean()
        ma20 = historical_data[price_col].rolling(window=20, min_periods=1).mean()
        
        # 计算布林带 - 默认参数：20日移动平均线，2倍标准差
        ma20_full = historical_data[price_col].rolling(window=20, min_periods=1).mean()
        std20 = historical_data[price_col].rolling(window=20, min_periods=1).std()
        upper_band = ma20_full + (std20 * 2)
        lower_band = ma20_full - (std20 * 2)
        
        # 计算KDJ - 默认参数：9日RSV，3日平滑
        low_min = historical_data[low_col].rolling(window=9, min_periods=1).min()
        high_max = historical_data[high_col].rolling(window=9, min_periods=1).max()
        price_diff = high_max - low_min
        price_diff = price_diff.replace(0, 0.001)  # 避免除零错误
        rsv = ((historical_data[price_col] - low_min) / price_diff * 100)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = (3 * k) - (2 * d)
        
        current_price = historical_data[price_col].iloc[-1]
        
        # 获取最近的有效值（避免NaN）
        def get_last_valid(series):
            return float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else float(series.dropna().iloc[-1]) if not series.dropna().empty else 0.0
        
        result = {
            "当前价格": float(current_price),
            "MA5": float(ma5.iloc[-1]),
            "MA10": float(ma10.iloc[-1]),
            "MA20": float(ma20.iloc[-1]),
            "RSI": float(rsi.iloc[-1]),
            "MACD": {
                "MACD线": float(macd.iloc[-1]),
                "信号线": float(signal.iloc[-1]),
                "柱状图": float(hist.iloc[-1])
            },
            "布林带": {
                "上轨": get_last_valid(upper_band),
                "中轨": get_last_valid(ma20_full),
                "下轨": get_last_valid(lower_band)
            },
            "KDJ": {
                "K值": float(k.iloc[-1]),
                "D值": float(d.iloc[-1]),
                "J值": float(j.iloc[-1])
            }
        }
        
        # 添加均线趋势信息
        result["均线趋势"] = {
            "5日均线斜率": float(ma5.iloc[-1] - ma5.iloc[-5]) if len(ma5) >= 5 else 0,
            "10日均线斜率": float(ma10.iloc[-1] - ma10.iloc[-5]) if len(ma10) >= 5 else 0,
            "5日穿10日": "上穿" if ma5.iloc[-2] < ma10.iloc[-2] and ma5.iloc[-1] > ma10.iloc[-1] else
                       "下穿" if ma5.iloc[-2] > ma10.iloc[-2] and ma5.iloc[-1] < ma10.iloc[-1] else "无变化"
        }
        
        # 添加技术指标趋势
        result["指标趋势"] = {
            "RSI变化": float(rsi.iloc[-1] - rsi.iloc[-2]),
            "MACD柱状图变化": float(hist.iloc[-1] - hist.iloc[-2]),
            "KDJ金叉死叉": "金叉" if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] else
                         "死叉" if k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] else "无变化"
        }
        
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        import traceback
        print(f"计算技术指标时出错: {str(e)}")
        print(traceback.format_exc())
        return json.dumps({"error": str(e)})

def get_prompt(stock_code: str) -> str:
    """获取股票的prompt"""
    stock_data = get_stock_data_func(stock_code)
    historical_data = get_week_historical_data_func(stock_code, 8)
    technical_indicators = calculate_technical_indicators_func( get_week_historical_data_func(stock_code, 30))
    # 构建提示词
    prompt = f"""
    你是一位专业的量化交易分析师，负责分析股票技术指标并提供买入、卖出或持有的建议。
    请基于提供的股票数据，分析股票当前形式，并给出明确的交易信号和理由。

    股票数据：
    1. 实时数据：{stock_data}
    2. 技术指标：{technical_indicators}
    3. 历史数据：{historical_data}

    分析过程：
    1. 首先分析技术指标的含义和重要性
    2. 然后评估当前市场环境和趋势
    3. 接着列出影响决策的关键因素
    4. 最后给出交易建议和价格目标

    你必须以JSON格式输出最终分析结果，包含以下字段：
    - signal: 必须是"买入"、"卖出"或"持有"之一
    - strength: 信号强度，必须是1-5之间的整数，5为最强
    - reasons: 数组，列出基于技术指标给出此建议的具体原因，至少3条
    - risk_factors: 数组，列出当前市场状况下的风险因素，至少2条
    - technical_analysis: 各个技术指标的详细分析，说明为什么这些指标支持你的交易信号
    - price_targets: 对象，包含以下内容：
    * next_buy: 对象，当signal为"卖出"或"持有"时提供
        - price: 建议的下一个买入价格点
        - reason: 买入理由
    * next_sell: 对象，当signal为"买入"或"持有"时提供
        - price: 建议的下一个卖出价格点
        - reason: 卖出理由
    - reasoning_process: 对象，包含以下内容：
    * technical_indicators_analysis: 详细分析每个技术指标的含义和重要性
    * market_context: 分析当前市场环境和趋势
    * decision_factors: 列出影响最终决策的关键因素
    * alternative_signals: 分析其他可能的交易信号及其理由
    * confidence_level: 对当前分析结果的信心程度说明

    注意：
    1. 价格目标必须包含具体的数值，不能为空
    2. 买入和卖出理由必须详细说明触发条件和逻辑
    3. 所有数值必须保留2位小数
    4. 如果当前不适合买入或卖出，请提供下一个合适的价位和原因
    5. 推理过程必须详细且逻辑清晰，展示完整的分析思路
    """
    return prompt

if __name__ == "__main__":
    print(get_prompt("sh159740"))