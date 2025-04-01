#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析不同时间段的策略表现
"""

from strategy import (
    LeftSideStrategy, 
    RightSideStrategy, 
    GridStrategy, 
    BreakoutStrategy, 
    DualMAStrategy
)
from strategy.backtest.backtest import BacktestEngine
from typing import Dict, List

def analyze_period(
    strategy_classes: List,
    stock_code: str,
    start_date: str,
    end_date: str,
    period_name: str,
    initial_capital: float = 100000.0,
    stop_loss: float = 0.1,
    take_profit: float = 0.2,
    transaction_fee: float = 0.0003
) -> Dict[str, Dict]:
    """
    分析特定时间段内各策略的表现
    :param strategy_classes: 策略类列表
    :param stock_code: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param period_name: 时间段名称
    :param initial_capital: 初始资金
    :param stop_loss: 止损比例
    :param take_profit: 止盈比例
    :param transaction_fee: 交易费用
    :return: 各策略的回测结果
    """
    results = {}
    
    print(f"\n=== {period_name} ({start_date} - {end_date}) ===")
    
    # 运行每个策略的回测
    for strategy_class in strategy_classes:
        strategy_name = strategy_class.__name__
        
        print(f"运行 {strategy_name} 回测...")
        engine = BacktestEngine(strategy_class, stock_code, initial_capital)
        report = engine.run_backtest(
            start_date, 
            end_date, 
            stop_loss=stop_loss, 
            take_profit=take_profit,
            transaction_fee=transaction_fee
        )
        
        results[strategy_name] = report
    
    # 创建格式化的表格
    header = "| 策略名称 | 总收益率 | 年化收益率 | 最大回撤 | 夏普比率 | 交易次数 |"
    separator = "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|"
    
    print(header)
    print(separator)
    
    # 打印每个策略的结果
    for strategy, report in results.items():
        if "error" in report:
            row = f"| {strategy:<8} | {'N/A':<8} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} |"
        else:
            row = f"| {strategy:<8} | {report['total_return']*100:8.2f}% | {report['annual_return']*100:10.2f}% | {report['max_drawdown']*100:8.2f}% | {report['sharpe_ratio']:8.2f} | {report['total_trades']:8} |"
        print(row)
    
    # 找出该时间段最优策略
    best_strategy = None
    best_value = -float('inf')
    
    for strategy, report in results.items():
        if "error" in report:
            continue
            
        value = report['total_return']
        if value > best_value:
            best_value = value
            best_strategy = strategy
    
    if best_strategy:
        print(f"\n该时间段最优策略: {best_strategy} (收益率: {best_value*100:.2f}%)")
        
    return results

def main():
    # 设置回测参数
    stock_code = "510300"  # 沪深300ETF
    initial_capital = 100000
    stop_loss = 0.1  # 止损比例
    take_profit = 0.2  # 止盈比例
    
    # 所有策略类
    strategies = [
        LeftSideStrategy,
        RightSideStrategy,
        GridStrategy,
        BreakoutStrategy,
        DualMAStrategy
    ]
    
    # 定义不同的时间段
    periods = [
        ("2020-2024全周期", "20200101", "20240328"),
        ("2020年牛市", "20200101", "20201231"),
        ("2021年震荡市", "20210101", "20211231"),
        ("2022年熊市", "20220101", "20221231"),
        ("2023年复苏市", "20230101", "20231231"),
        ("2024年至今", "20240101", "20240328")
    ]
    
    print("分析不同时间段的策略表现...")
    
    # 记录每个时间段的最优策略
    best_strategies = {}
    
    # 分析每个时间段
    for period_name, start_date, end_date in periods:
        results = analyze_period(
            strategies,
            stock_code,
            start_date,
            end_date,
            period_name,
            initial_capital,
            stop_loss,
            take_profit,
            transaction_fee=0.0003
        )
        
        # 找出最优策略
        best_strategy = None
        best_value = -float('inf')
        
        for strategy, report in results.items():
            if "error" in report:
                continue
                
            value = report['total_return']
            if value > best_value:
                best_value = value
                best_strategy = strategy
                
        if best_strategy:
            best_strategies[period_name] = (best_strategy, best_value)
    
    # 打印总结
    print("\n=== 不同市场环境下的最优策略 ===")
    for period, (strategy, value) in best_strategies.items():
        print(f"{period}: {strategy} (收益率: {value*100:.2f}%)")

if __name__ == "__main__":
    main() 