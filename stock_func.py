
from ast import main
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import json


# 定义工具函数
def get_stock_data_func(stock_code: str) -> str:
    """获取指定股票的最新市场数据"""
    try:
        if stock_code.startswith('sh') or stock_code.startswith('sz'):
            # 获取A股实时数据
            stock_data = ak.stock_zh_a_spot_em()
            # 找到对应的股票
            code = stock_code[2:] if stock_code.startswith('sh') or stock_code.startswith('sz') else stock_code
            filtered_data = stock_data[stock_data['代码'] == code]
            if not filtered_data.empty:
                return filtered_data.to_json(orient='records', force_ascii=False)
        
        # 如果是指数
        if stock_code == 'sh000001':
            index_data = ak.stock_zh_index_spot()
            filtered_data = index_data[index_data['指数代码'] == '000001']
            return filtered_data.to_json(orient='records', force_ascii=False)
        
        # 默认返回上证指数
        filtered_data = ak.stock_zh_index_spot()[ak.stock_zh_index_spot()['指数代码'] == '000001']
        return filtered_data.to_json(orient='records', force_ascii=False)
    except Exception as e:
        print(f"获取股票数据出错: {str(e)}")
        return json.dumps({"error": str(e)})

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

if __name__ == "__main__":
    print(get_stock_data_func("sh600036"))
    print("--------------------------------")
    data = get_historical_data_func("sh600036", 60)
    print(data)
    print("--------------------------------")
    print(calculate_technical_indicators_func(data))