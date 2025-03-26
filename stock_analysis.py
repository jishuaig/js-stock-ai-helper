import json
from stock_func import get_stock_data_func, get_historical_data_func, calculate_technical_indicators_func
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from tabulate import tabulate
from datetime import datetime

def print_model_io(messages, response):
    """打印模型的输入输出"""
    print("\n" + "="*50)
    print("模型输入:")
    for msg in messages:
        print(f"\n角色: {msg.type}")
        print(f"内容: {msg.content}")
        print("-"*50)
    
    print("\n模型输出:")
    print(f"内容: {response.content}")
    print("="*50 + "\n")

def clean_json_response(content: str) -> str:
    """清理模型返回的JSON内容，移除markdown标记"""
    # 移除可能的markdown代码块标记
    content = content.replace("```json", "").replace("```", "").strip()
    return content

def analyze_stock(stock_code: str):
    """
    分析指定股票并生成分析报告
    """
    print(f"\n开始分析股票: {stock_code}")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取实时数据
    realtime_data = json.loads(get_stock_data_func(stock_code))
    print("已获取实时数据")
    
    # 获取历史数据（获取60天数据用于技术分析）
    historical_data_json = get_historical_data_func(stock_code, days=5)
    print("已获取历史数据")
    
    # 计算技术指标
    technical_indicators = json.loads(calculate_technical_indicators_func(get_historical_data_func(stock_code, days=60)))
    print("已计算技术指标")
    
    # 构建提示词
    prompt = f"""
    请基于以下股票数据进行分析，并给出详细的分析报告：
    
    1. 实时数据：{json.dumps(realtime_data, ensure_ascii=False)}
    2. 技术指标：{json.dumps(technical_indicators, ensure_ascii=False)}
    3. 历史数据：{json.dumps(json.loads(historical_data_json), ensure_ascii=False)}
    
    请提供以下分析内容：
    1. 当前市场状况分析
    2. 技术面分析
    3. 买入/卖出建议（包括具体价格区间）
    4. 风险提示
    
    请以JSON格式返回，包含以下字段：
    {{
        "market_analysis": "市场状况分析",
        "technical_analysis": "技术面分析",
        "trading_suggestion": {{
            "action": "买入/卖出/持有",
            "price_range": "建议价格区间",
            "reason": "建议理由"
        }},
        "risk_tips": "风险提示"
    }}
    """
    
    # 创建DeepSeek实例
    model = ChatDeepSeek(
        model="deepseek-reasoner",  # 使用reasoner模型进行复杂推理
        temperature=0,              # 使用确定性输出
        max_tokens=None,            # 不限制生成长度
        timeout=120,                # API超时时间（秒）
        max_retries=1,              # API调用失败重试次数
        api_key="sk-aacb49dcd0654de78c2b0d694296d5d1"  # 使用您的API密钥
    )
    
    # 构建消息列表
    messages = [
        SystemMessage(content="你是一个专业的股票分析师，请基于提供的数据进行分析。"),
        HumanMessage(content=prompt)
    ]
    
    # 调用模型进行分析
    print("\n开始调用模型进行分析...")
    response = model.invoke(messages)
    print("模型分析完成")
    
    # 打印模型的输入输出
    print_model_io(messages, response)
    
    # 清理并解析分析结果
    cleaned_content = clean_json_response(response.content)
    analysis_result = json.loads(cleaned_content)
    
    # 将结果转换为表格形式
    table_data = [
        ["分析项目", "内容"],
        ["市场状况分析", analysis_result["market_analysis"]],
        ["技术面分析", analysis_result["technical_analysis"]],
        ["交易建议", f"{analysis_result['trading_suggestion']['action']} - {analysis_result['trading_suggestion']['price_range']}"],
        ["建议理由", analysis_result["trading_suggestion"]["reason"]],
        ["风险提示", analysis_result["risk_tips"]]
    ]
    
    # 打印表格
    print(f"\n{stock_code} 股票分析报告")
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    print("\n分析报告生成完成")
    return analysis_result

if __name__ == "__main__":
    # 示例：分析招商银行
    analyze_stock("sh600036") 