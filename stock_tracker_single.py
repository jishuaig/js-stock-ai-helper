import time
import akshare as ak
import pandas as pd
from datetime import datetime
import json
import argparse
from tabulate import tabulate
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
import io
from stock_func import get_stock_data_func, get_historical_data_func, calculate_technical_indicators_func

# 获取命令行参数
parser = argparse.ArgumentParser(description='股票实时追踪与交易信号生成')
parser.add_argument('--stock_code', type=str, default='sh600036', help='股票代码，例如sh000001(上证指数)')
parser.add_argument('--interval', type=int, default=60, help='追踪周期(秒)，默认60秒')
parser.add_argument('--api_key', type=str, default='sk-aacb49dcd0654de78c2b0d694296d5d1', help='DeepSeek API密钥')
parser.add_argument('--verbose', action='store_true', default=True, help='是否打印详细交互信息')
parser.add_argument('--history_days', type=int, default=5, help='获取历史数据的天数，默认5天')
args = parser.parse_args()

# 创建DeepSeek实例
model = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0.5,
    max_tokens=None,
    timeout=120,
    max_retries=2,
    api_key=args.api_key
)

# 系统提示词
system_message = SystemMessage(content="""
你是一位专业的量化交易分析师，负责分析股票技术指标并提供买入、卖出或持有的建议。
请基于提供的技术指标（如MA、RSI、MACD、KDJ等）分析股票当前状态，并给出明确的交易信号和理由。

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
""")

def format_analysis_for_display(analysis_json):
    """将JSON分析结果转换为人类可读格式"""
    try:
        data = json.loads(analysis_json)
        
        output = []
        output.append(f"交易信号: {data['signal']}")
        output.append(f"信号强度: {data['strength']}/5")
        
        output.append("\n理由:")
        for i, reason in enumerate(data['reasons'], 1):
            output.append(f"  {i}. {reason}")
        
        output.append("\n风险因素:")
        for i, risk in enumerate(data['risk_factors'], 1):
            output.append(f"  {i}. {risk}")
        
        output.append("\n技术指标分析:")
        for indicator, analysis in data['technical_analysis'].items():
            output.append(f"  - {indicator}: {analysis}")
        
        if 'price_targets' in data:
            price_targets = data['price_targets']
            output.append("\n价格目标:")
            
            if 'next_buy' in price_targets:
                buy_target = price_targets['next_buy']
                output.append(f"  - 建议买入价位: {buy_target['price']}")
                output.append(f"    原因: {buy_target['reason']}")
                
            if 'next_sell' in price_targets:
                sell_target = price_targets['next_sell']
                output.append(f"  - 建议卖出价位: {sell_target['price']}")
                output.append(f"    原因: {sell_target['reason']}")
        
        if 'reasoning_process' in data:
            reasoning = data['reasoning_process']
            output.append("\n【推理过程】")
            output.append("\n技术指标分析:")
            for indicator, analysis in reasoning.get('technical_indicators_analysis', {}).items():
                output.append(f"  - {indicator}: {analysis}")
            
            output.append("\n市场环境分析:")
            output.append(f"  {reasoning.get('market_context', 'N/A')}")
            
            output.append("\n决策关键因素:")
            for i, factor in enumerate(reasoning.get('decision_factors', []), 1):
                output.append(f"  {i}. {factor}")
            
            output.append("\n其他可能的交易信号:")
            for signal, reason in reasoning.get('alternative_signals', {}).items():
                output.append(f"  - {signal}: {reason}")
            
            output.append("\n信心程度说明:")
            output.append(f"  {reasoning.get('confidence_level', 'N/A')}")
        
        return "\n".join(output)
    except json.JSONDecodeError:
        return analysis_json
    except KeyError as e:
        return f"JSON格式错误，缺少字段: {e}\n原始响应: {analysis_json}"

def clean_json_response(content: str) -> str:
    """清理模型返回的JSON内容，移除markdown标记"""
    content = content.replace("```json", "").replace("```", "").strip()
    return content

def main():
    print(f"开始追踪股票: {args.stock_code}，刷新周期: {args.interval}秒，历史数据天数: {args.history_days}天")
    
    # 为控制台表格输出准备的历史记录
    console_history = []
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*50}")
            print(f"==== {current_time} ====")
            print(f"{'='*50}\n")
            
            # 获取实时数据
            stock_data_json = get_stock_data_func(args.stock_code)
            stock_data = pd.read_json(io.StringIO(stock_data_json), orient='records')
            
            # 获取历史数据
            hist_data_json = get_historical_data_func(args.stock_code, args.history_days)

            hist_data_json_60 = get_historical_data_func(args.stock_code, 60)
            
            # 计算技术指标
            technical_indicators = calculate_technical_indicators_func(hist_data_json_60)
            
            # 构建分析请求
            analysis_request = {
                "current_data": json.loads(stock_data_json),
                "historical_data": json.loads(hist_data_json),
                "technical_indicators": json.loads(technical_indicators)
            }
            
            # 调用大模型进行分析
            messages = [
                system_message,
                HumanMessage(content=f"请分析以下股票数据并给出交易建议：\n{json.dumps(analysis_request, ensure_ascii=False, indent=2)}")
            ]
            
            response = model.invoke(messages)
            
            # 获取推理过程
            reasoning_content = response.content
            if hasattr(response, 'reasoning_content'):
                reasoning_content = response.reasoning_content
            
            # 清理JSON响应
            analysis_json = clean_json_response(response.content)
            
            # 解析分析结果
            analysis_data = {}
            signal_text = "未知"
            signal_strength = "N/A"
            
            try:
                analysis_data = json.loads(analysis_json)
                signal_text = analysis_data.get("signal", "未知")
                signal_strength = analysis_data.get("strength", "N/A")
                
                # 显示分析结果
                print("\n【分析结果】")
                print("-" * 40)
                print(json.dumps(analysis_data, ensure_ascii=False, indent=2))
                print("-" * 40)
                
                # 显示推理过程
                print("\n【推理过程】")
                print("-" * 40)
                print(reasoning_content)
                print("-" * 40)
                
                # 显示人类可读的格式化分析
                print("\n【详细解读】")
                print("-" * 40)
                print(format_analysis_for_display(analysis_json))
                print("-" * 40)
            except json.JSONDecodeError:
                print("\n【分析结果】(非JSON格式)")
                print("-" * 40)
                print(analysis_json)
                print("-" * 40)
            
            # 获取当前价格并更新历史记录
            price_col = '最新价' if '最新价' in stock_data.columns else '收盘' if '收盘' in stock_data.columns else None
            
            if price_col and not stock_data.empty:
                current_price = stock_data.iloc[0][price_col]
                
                # 获取价格目标信息
                buy_price = "N/A"
                sell_price = "N/A"
                buy_reason = ""
                sell_reason = ""
                
                if 'price_targets' in analysis_data:
                    price_targets = analysis_data['price_targets']
                    if 'next_buy' in price_targets:
                        buy_target = price_targets['next_buy']
                        buy_price = buy_target['price']
                        buy_reason = buy_target['reason']
                    if 'next_sell' in price_targets:
                        sell_target = price_targets['next_sell']
                        sell_price = sell_target['price']
                        sell_reason = sell_target['reason']
                
                # 更新历史记录
                console_history.append([
                    current_time, 
                    f"{current_price:.2f}", 
                    signal_text, 
                    signal_strength,
                    str(buy_price),
                    str(sell_price)
                ])
                
                # 保留最近10个数据点
                if len(console_history) > 10:
                    console_history.pop(0)
                
                # 打印历史记录表格
                print("\n【历史交易信号】")
                headers = ["时间", "价格", "交易信号", "信号强度", "买入目标", "卖出目标"]
                print(tabulate(console_history, headers=headers, tablefmt="grid"))
                
                # 打印最新的买入/卖出价格理由
                if buy_price != "N/A" or sell_price != "N/A":
                    print("\n【最新价格目标详情】")
                    if buy_price != "N/A":
                        print(f"买入目标价: {buy_price}")
                        print(f"买入理由: {buy_reason}")
                    if sell_price != "N/A":
                        print(f"卖出目标价: {sell_price}")
                        print(f"卖出理由: {sell_reason}")
                    print()
            
            # 等待下一个周期
            print(f"\n等待 {args.interval} 秒进行下一次分析...")
            print()
            time.sleep(args.interval)
            
        except KeyboardInterrupt:
            print("\n程序已停止")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            break

if __name__ == "__main__":
    main() 