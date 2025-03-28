#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较所有交易策略的表现（仅文本输出）
"""

from strategy import (
    LeftSideStrategy, 
    RightSideStrategy, 
    GridStrategy, 
    BreakoutStrategy, 
    DualMAStrategy
)
from backtest import BacktestEngine
from typing import Dict, List

def compare_strategies_text(
    strategy_classes: List,
    stock_code: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    stop_loss: float = 0.1,
    take_profit: float = 0.2,
    transaction_fee: float = 0.0003
) -> Dict[str, Dict]:
    """
    对比多个策略的表现，仅输出文本结果
    :param strategy_classes: 策略类列表
    :param stock_code: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param initial_capital: 初始资金
    :param stop_loss: 止损比例
    :param take_profit: 止盈比例
    :param transaction_fee: 交易费用
    :return: 各策略的回测结果
    """
    results = {}
    
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
    
    return results

def main():
    # 设置回测参数
    stock_code = "510300"  # 沪深300ETF
    start_date = "20250101"  # 从2020年开始
    end_date = "20250328"
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
    
    print("正在对比所有交易策略...")
    
    # 对比所有策略
    results = compare_strategies_text(
        strategies,
        stock_code,
        start_date,
        end_date,
        initial_capital,
        stop_loss,
        take_profit,
        transaction_fee=0.0003
    )
    
    # 打印回测结果
    print("\n=== 回测结果对比 ===")
    
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
    
    # 找出每个指标的最优策略
    metrics = [
        ("总收益率", "total_return", "%"),
        ("年化收益率", "annual_return", "%"),
        ("最大回撤", "max_drawdown", "%"),
        ("夏普比率", "sharpe_ratio", ""),
        ("交易次数", "total_trades", "")
    ]
    
    print("\n=== 最优策略 ===")
    
    for metric_name, metric_key, unit in metrics:
        best_strategy = None
        best_value = -float('inf')
        
        for strategy, report in results.items():
            if "error" in report:
                continue
                
            if metric_key in ['total_return', 'annual_return', 'sharpe_ratio']:
                # 这些指标越大越好
                value = report[metric_key]
                if value > best_value:
                    best_value = value
                    best_strategy = strategy
            elif metric_key == 'max_drawdown':
                # 回撤绝对值越小越好
                value = report[metric_key]
                if abs(value) < abs(best_value) or best_value == -float('inf'):
                    best_value = value
                    best_strategy = strategy
            else:
                # 其他指标正常比较
                value = report[metric_key]
                if value > best_value:
                    best_value = value
                    best_strategy = strategy
        
        if best_strategy:
            if metric_key in ['total_return', 'annual_return', 'max_drawdown']:
                print(f"- {metric_name}: {best_strategy} ({best_value*100:.2f}{unit})")
            else:
                print(f"- {metric_name}: {best_strategy} ({best_value}{unit})")

if __name__ == "__main__":
    main() 