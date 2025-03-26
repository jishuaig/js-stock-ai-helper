import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
import json
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
import os

# 从环境变量获取 API 密钥
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

def get_sector_data():
    """获取行业板块数据"""
    try:
        # 获取行业板块行情数据
        sector_data = ak.stock_board_industry_name_em()
        
        # 打印原始数据信息
        print("\n原始数据信息:")
        print("数据形状:", sector_data.shape)
        print("列名:", sector_data.columns.tolist())
        print("\n数据示例:")
        print(sector_data.head())
        
        # 确保数据不为空
        if sector_data.empty:
            print("获取的数据为空")
            return None
            
        return sector_data
    except Exception as e:
        print(f"获取行业板块数据失败: {e}")
        return None

def get_sector_historical_data(sector_name, days=30):
    """获取行业板块历史数据"""
    try:
        # 获取行业板块历史数据
        historical_data = ak.stock_board_industry_hist_em(symbol=sector_name, period="daily", 
                                                        start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                                        end_date=datetime.now().strftime('%Y%m%d'))
        return historical_data
    except Exception as e:
        print(f"获取行业板块历史数据失败: {e}")
        return None

def calculate_sector_indicators(sector_data):
    """计算行业板块技术指标"""
    try:
        # 打印列名，用于调试
        print("数据列名:", sector_data.columns.tolist())
        
        # 确保必要的列存在
        required_columns = ['涨跌幅', '换手率', '领涨股票']
        for col in required_columns:
            if col not in sector_data.columns:
                print(f"警告: 缺少必要的列 {col}")
                return None
        
        # 计算涨跌幅
        sector_data['涨跌幅'] = sector_data['涨跌幅'].astype(float)
        
        # 计算换手率
        sector_data['换手率'] = sector_data['换手率'].astype(float)
        
        # 计算领涨股数量
        sector_data['领涨股数量'] = sector_data['领涨股票'].str.count(',') + 1
        
        return sector_data
    except Exception as e:
        print(f"计算行业板块指标失败: {e}")
        print("数据示例:")
        print(sector_data.head())
        return None

def analyze_sector_potential(sector_data):
    """分析行业板块潜力"""
    try:
        # 创建DeepSeek实例
        model = ChatDeepSeek(
            model="deepseek-reasoner",
            temperature=0,
            max_tokens=None,
            timeout=120,
            max_retries=1,
            api_key=api_key
        )
        
        # 构建分析提示词
        prompt = f"""
        请基于以下行业板块数据进行分析，找出最具潜力的板块：
        
        数据：{json.dumps(sector_data.to_dict('records'), ensure_ascii=False)}
        
        请从以下几个方面进行分析：
        1. 板块整体表现（涨跌幅、换手率）
        2. 板块内个股表现（领涨股数量、领涨股涨幅）
        3. 板块资金流向
        4. 板块估值水平
        5. 行业政策面分析
        
        请严格按照以下JSON格式返回分析结果，不要添加任何其他内容：
        {{
            "top_sectors": [
                {{
                    "name": "板块名称",
                    "score": "潜力评分(1-10)",
                    "reasons": ["潜力原因1", "潜力原因2", ...],
                    "risks": ["风险因素1", "风险因素2", ...]
                }}
            ],
            "summary": "整体市场分析总结"
        }}
        """
        
        # 构建消息列表
        messages = [
            SystemMessage(content="你是一个专业的行业分析师，请基于提供的数据进行分析。请严格按照指定的JSON格式返回结果，不要添加任何其他内容。"),
            HumanMessage(content=prompt)
        ]
        
        # 调用模型进行分析
        response = model.invoke(messages)
        
        # 打印原始响应，用于调试
        print("\nAI模型原始响应:")
        print(response.content)
        
        # 尝试清理响应内容
        content = response.content.strip()
        # 移除可能的markdown代码块标记
        content = content.replace("```json", "").replace("```", "").strip()
        
        # 尝试解析JSON
        try:
            analysis_result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print("清理后的响应内容:")
            print(content)
            return None
        
        # 验证返回的数据结构
        if not isinstance(analysis_result, dict) or 'top_sectors' not in analysis_result or 'summary' not in analysis_result:
            print("返回的数据结构不正确")
            return None
            
        return analysis_result
    except Exception as e:
        print(f"分析行业板块潜力失败: {e}")
        return None

def print_analysis_result(analysis_result):
    """打印分析结果"""
    if not analysis_result:
        print("分析结果为空")
        return
    
    print("\n=== A股市场潜力板块分析报告 ===")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n最具潜力板块：")
    for sector in analysis_result['top_sectors']:
        print(f"\n板块名称: {sector['name']}")
        print(f"潜力评分: {sector['score']}/10")
        print("潜力原因:")
        for reason in sector['reasons']:
            print(f"- {reason}")
        print("风险因素:")
        for risk in sector['risks']:
            print(f"- {risk}")
    
    print("\n市场总结:")
    print(analysis_result['summary'])

def main():
    """主函数"""
    print("开始获取行业板块数据...")
    sector_data = get_sector_data()
    if sector_data is None:
        return
    
    print("计算行业板块指标...")
    sector_data = calculate_sector_indicators(sector_data)
    if sector_data is None:
        return
    
    print("分析行业板块潜力...")
    analysis_result = analyze_sector_potential(sector_data)
    
    print_analysis_result(analysis_result)

if __name__ == "__main__":
    main() 