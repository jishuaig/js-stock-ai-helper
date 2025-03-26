import time
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import json
import argparse
import os
from tabulate import tabulate
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
import io
from stock_func import get_stock_data_func, get_historical_data_func, calculate_technical_indicators_func

# 获取命令行参数
parser = argparse.ArgumentParser(description='股票实时追踪与交易信号生成')
parser.add_argument('--stock_code', type=str, default='sh600036', help='股票代码，例如sh000001(上证指数)')
parser.add_argument('--interval', type=int, default=60, help='追踪周期(秒)，默认60秒')
parser.add_argument('--verbose', action='store_true', default=True, help='是否打印详细交互信息')
parser.add_argument('--history_days', type=int, default=5, help='获取历史数据的天数，默认5天')
args = parser.parse_args()

# 从环境变量获取 API 密钥
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

# 创建DeepSeek实例
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=120,
    max_retries=2,
    api_key=api_key
)
# 打印消息的辅助函数
def print_message(message, prefix=""):
    """打印消息内容，格式化工具调用等"""
    if isinstance(message, HumanMessage):
        print(f"{prefix}用户: {message.content}")
    elif isinstance(message, SystemMessage):
        print(f"{prefix}系统: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"{prefix}AI: {message.content}")
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"{prefix}工具调用: {json.dumps(message.tool_calls, ensure_ascii=False, indent=2)}")
    elif isinstance(message, ToolMessage):
        print(f"{prefix}工具结果 (ID: {message.tool_call_id}): ")
        # 尝试格式化JSON响应
        try:
            content = json.loads(message.content)
            print(f"{prefix}{json.dumps(content, ensure_ascii=False, indent=2)}")
        except:
            # 如果不是JSON，直接打印
            print(f"{prefix}{message.content}")
    else:
        print(f"{prefix}其他消息: {message}")

# 使用Tool类创建工具
get_stock_data = Tool(
    name="get_stock_data",
    func=get_stock_data_func,
    description="获取指定股票的最新市场数据，参数为股票代码（例如：sh000001）"
)

get_historical_data = Tool(
    name="get_historical_data",
    func=get_historical_data_func,
    description="获取指定股票的历史K线数据，参数为股票代码（例如：sh000001）和天数（可选，默认5天）"
)

calculate_technical_indicators = Tool(
    name="calculate_technical_indicators",
    func=calculate_technical_indicators_func,
    description="计算股票技术指标，参数为JSON格式的历史数据"
)

# 绑定工具到模型
tools = [get_stock_data, get_historical_data, calculate_technical_indicators]
model_with_tools = model.bind_tools(tools)

# 系统提示词
system_message = SystemMessage(content="""
你是一位专业的量化交易分析师，负责分析股票技术指标并提供买入、卖出或持有的建议。
请基于提供的技术指标（如MA、RSI、MACD、KDJ等）分析股票当前状态，并给出明确的交易信号和理由。

你必须以JSON格式输出最终分析结果，包含以下字段：
- signal: 必须是"买入"、"卖出"或"持有"之一
- strength: 信号强度，必须是1-5之间的整数，5为最强
- reasons: 数组，列出基于技术指标给出此建议的具体原因，至少3条
- risk_factors: 数组，列出当前市场状况下的风险因素，至少2条
- technical_analysis: 各个技术指标的详细分析，说明为什么这些指标支持你的交易信号
- price_targets: 对象，包含以下内容：
  * 当signal为"卖出"时：提供建议的下一个买入价格点和原因
  * 当signal为"买入"时：提供建议的下一个卖出价格点和原因
  * 当signal为"持有"时：同时提供潜在的买入和卖出价格点

示例输出格式：
{
  "signal": "买入",
  "strength": 4,
  "reasons": [
    "RSI指标显示市场处于超卖状态(29.5)",
    "MACD指标出现金叉信号",
    "股价突破20日均线形成支撑"
  ],
  "risk_factors": [
    "大盘整体处于调整阶段",
    "成交量萎缩，上涨动力不足"
  ],
  "technical_analysis": {
    "移动均线": "5日均线上穿10日均线，形成金叉",
    "RSI": "RSI值从超卖区反弹，显示上升动能",
    "MACD": "MACD柱状图由负转正，买入信号增强",
    "KDJ": "KDJ指标三线向上，K线与D线形成金叉"
  },
  "price_targets": {
    "next_sell": {
      "price": 25.80,
      "reason": "接近布林带上轨，技术性阻力位明显，且RSI接近超买区间"
    }
  }
}

或者卖出信号的示例：
{
  "signal": "卖出",
  "strength": 4,
  "reasons": [
    "股价跌破20日均线支撑位",
    "MACD指标出现死叉信号",
    "KDJ指标三线向下，显示下跌动能"
  ],
  "risk_factors": [
    "大盘可能开始见底反弹",
    "政策面利好可能出现"
  ],
  "technical_analysis": {
    "移动均线": "5日均线下穿10日均线，形成死叉",
    "RSI": "RSI值从高位回落，显示下跌动能",
    "MACD": "MACD柱状图由正转负，卖出信号增强",
    "KDJ": "KDJ指标三线向下，K线与D线形成死叉"
  },
  "price_targets": {
    "next_buy": {
      "price": 18.50,
      "reason": "接近布林带下轨，为技术性支撑位，且RSI接近超卖区间"
    }
  }
}
""")

# 为控制台表格输出准备的历史记录
console_history = []

# 执行工具调用的函数
def execute_tool_call(tool_call, stock_code, verbose=False):
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_call_id = tool_call["id"]
    
    if verbose:
        print(f"\n  执行工具: {tool_name}")
        print(f"  参数: {json.dumps(tool_args, ensure_ascii=False)}")
    
    if tool_name == "get_stock_data":
        result = get_stock_data_func(stock_code)
    elif tool_name == "get_historical_data":
        days = tool_args.get("days", args.history_days)
        if isinstance(days, dict):  # 处理嵌套字典的情况
            days = args.history_days
        result = get_historical_data_func(stock_code, days)
    elif tool_name == "calculate_technical_indicators":
        # 首先需要获取历史数据，确保获取足够多的天数用于计算指标
        hist_data_json = get_historical_data_func(stock_code, 60)  # 至少30天历史数据
        result = calculate_technical_indicators_func(hist_data_json)
    else:
        result = json.dumps({"error": f"未知工具: {tool_name}"})
    
    if verbose:
        # 尝试美化打印工具结果
        try:
            result_obj = json.loads(result)
            pretty_result = json.dumps(result_obj, ensure_ascii=False, indent=2)
            print(f"  结果: {pretty_result[:500]}{'...' if len(pretty_result) > 500 else ''}")
        except:
            print(f"  结果: {result[:500]}{'...' if len(result) > 500 else ''}")
    
    return result, tool_call_id

# 解析分析结果，将JSON转换为人类可读格式
def format_analysis_for_display(analysis_json):
    try:
        data = json.loads(analysis_json)
        
        # 构建人类可读的输出文本
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
        
        # 添加价格目标信息
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
        
        return "\n".join(output)
    except json.JSONDecodeError:
        # 如果不是有效的JSON，直接返回原文本
        return analysis_json
    except KeyError as e:
        # 如果缺少关键字段
        return f"JSON格式错误，缺少字段: {e}\n原始响应: {analysis_json}"

def clean_json_response(content: str) -> str:
    """清理模型返回的JSON内容，移除markdown标记"""
    # 移除可能的markdown代码块标记
    content = content.replace("```json", "").replace("```", "").strip()
    return content

# 主循环
def main():
    print(f"开始追踪股票: {args.stock_code}，刷新周期: {args.interval}秒，历史数据天数: {args.history_days}天")
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*50}")
            print(f"==== {current_time} ====")
            print(f"{'='*50}\n")
            
            # 构建查询
            query = f"分析股票 {args.stock_code} 的当前技术指标，并给出交易信号。请以JSON格式输出分析结果。"
            
            # 初始消息
            messages = [
                system_message,
                HumanMessage(content=query)
            ]
            
            # 打印初始消息
            if args.verbose:
                print("\n----- 初始消息 -----")
                for msg in messages:
                    print_message(msg, "  ")
            
            # 工具调用循环 - 处理连续多轮工具调用
            tool_call_round = 0
            max_tool_call_rounds = 5  # 防止无限循环
            is_final_response = False
            
            while not is_final_response and tool_call_round < max_tool_call_rounds:
                tool_call_round += 1
                
                if args.verbose:
                    print(f"\n----- 工具调用回合 {tool_call_round} -----")
                
                # 调用模型
                ai_message = model_with_tools.invoke(messages)
                
                # 打印AI响应
                if args.verbose:
                    print(f"\n----- AI响应 (回合 {tool_call_round}) -----")
                    print_message(ai_message, "  ")
                
                # 添加AI消息到对话历史
                messages.append(ai_message)
                
                # 检查是否有工具调用
                has_tool_calls = hasattr(ai_message, 'tool_calls') and ai_message.tool_calls
                
                if not has_tool_calls:
                    # 没有工具调用，说明是最终回答
                    is_final_response = True
                    final_response = ai_message
                    break
                
                # 执行工具调用
                if args.verbose:
                    print(f"\n----- 执行工具调用 (回合 {tool_call_round}) -----")
                
                for tool_call in ai_message.tool_calls:
                    result, tool_call_id = execute_tool_call(tool_call, args.stock_code, args.verbose)
                    
                    # 添加工具结果到消息队列
                    tool_message = ToolMessage(content=result, tool_call_id=tool_call_id)
                    messages.append(tool_message)
            
            # 如果没有得到最终响应，强制请求一个总结
            if not is_final_response:
                final_query = f"""
                基于上述所有工具调用结果，分析 {args.stock_code} 的当前状态，并给出明确的交易信号。
                记住，你必须以JSON格式输出分析结果，包含signal、strength、reasons、risk_factors和technical_analysis字段。
                """
                final_message = HumanMessage(content=final_query)
                messages.append(final_message)
                
                if args.verbose:
                    print("\n----- 最终分析请求 -----")
                    print_message(final_message, "  ")
                
                final_response = model_with_tools.invoke(messages)
                
                if args.verbose:
                    print("\n----- 最终分析结果 -----")
                    print_message(final_response, "  ")
            
            # 原始JSON分析结果
            analysis_json = clean_json_response(final_response.content)
            
            # 初始化变量
            analysis_data = {}
            signal_text = "未知"
            signal_strength = "N/A"
            
            # 尝试解析JSON
            try:
                analysis_data = json.loads(analysis_json)
                signal_text = analysis_data.get("signal", "未知")
                signal_strength = analysis_data.get("strength", "N/A")
                
                # 显示分析结果
                print("\n【分析结果】")
                print("-" * 40)
                # 美化输出的JSON
                print(json.dumps(analysis_data, ensure_ascii=False, indent=2))
                print("-" * 40)
                
                # 显示人类可读的格式化分析
                print("\n【详细解读】")
                print("-" * 40)
                print(format_analysis_for_display(analysis_json))
                print("-" * 40)
            except json.JSONDecodeError:
                # 如果无法解析为JSON，直接显示原始文本
                print("\n【分析结果】(非JSON格式)")
                print("-" * 40)
                print(analysis_json)
                print("-" * 40)
                
                # 尝试从文本中提取信号和强度
                for line in analysis_json.split('\n'):
                    if "交易信号：" in line or "交易信号:" in line:
                        signal_text = line.split("：")[-1].strip() if "：" in line else line.split(":")[-1].strip()
                    elif "信号强度：" in line or "信号强度:" in line:
                        signal_strength = line.split("：")[-1].strip() if "：" in line else line.split(":")[-1].strip()
            
            # 获取当前价格
            stock_data_json = get_stock_data_func(args.stock_code)
            stock_data = pd.read_json(io.StringIO(stock_data_json), orient='records')
            
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
                
                # 为控制台表格更新历史记录（添加买入卖出价格）
                console_history.append([
                    current_time, 
                    f"{current_price:.2f}", 
                    signal_text, 
                    signal_strength,
                    str(buy_price),
                    str(sell_price)
                ])
                
                # 保留最近10个数据点用于显示
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