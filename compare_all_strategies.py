#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较所有交易策略的表现
"""

from strategy import (
    LeftSideStrategy, 
    RightSideStrategy, 
    GridStrategy, 
    BreakoutStrategy, 
    DualMAStrategy
)
from backtest import BacktestEngine
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from typing import Dict, List

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def compare_multiple_strategies(
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
    对比多个策略的表现
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
    strategy_names = []
    all_daily_capital = []
    
    # 运行每个策略的回测
    for strategy_class in strategy_classes:
        strategy_name = strategy_class.__name__
        strategy_names.append(strategy_name)
        
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
        
        if engine.strategy.daily_capital:
            all_daily_capital.append({
                'name': strategy_name,
                'dates': [d['date'] for d in engine.strategy.daily_capital],
                'capital': [d['capital'] for d in engine.strategy.daily_capital]
            })
    
    # 绘制对比图
    plt.figure(figsize=(14, 8))
    
    for data in all_daily_capital:
        plt.plot(data['dates'], data['capital'], label=data['name'])
    
    plt.title('各策略资金曲线对比')
    plt.xlabel('日期')
    plt.ylabel('资金')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 绘制策略对比表格
    if results:
        plt.figure(figsize=(12, len(results) * 0.5 + 2))
        
        strategies = list(results.keys())
        metrics = ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'total_trades']
        metric_names = ['总收益率', '年化收益率', '最大回撤', '夏普比率', '交易次数']
        
        data = []
        for strategy in strategies:
            if "error" not in results[strategy]:
                row = [
                    f"{results[strategy]['total_return']*100:.2f}%",
                    f"{results[strategy]['annual_return']*100:.2f}%",
                    f"{results[strategy]['max_drawdown']*100:.2f}%",
                    f"{results[strategy]['sharpe_ratio']:.2f}",
                    f"{results[strategy]['total_trades']}"
                ]
                data.append(row)
            else:
                data.append(["N/A", "N/A", "N/A", "N/A", "N/A"])
        
        plt.table(
            cellText=data,
            rowLabels=strategies,
            colLabels=metric_names,
            loc='center',
            cellLoc='center',
            colWidths=[0.15] * len(metrics)
        )
        
        plt.axis('off')
        plt.title('策略表现对比')
        plt.tight_layout()
        plt.show()
    
    return results

def main():
    # 设置回测参数
    stock_code = "510300"  # 沪深300ETF
    start_date = "20230101"
    end_date = "20240328"
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
    results = compare_multiple_strategies(
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
    metrics = [
        ("总收益率", "total_return", "%"),
        ("年化收益率", "annual_return", "%"),
        ("最大回撤", "max_drawdown", "%"),
        ("夏普比率", "sharpe_ratio", ""),
        ("交易次数", "total_trades", "")
    ]
    
    # 找出每个指标的最优策略
    for metric_name, metric_key, unit in metrics:
        print(f"\n{metric_name}对比:")
        
        best_strategy = None
        best_value = -float('inf')
        
        for strategy, report in results.items():
            if "error" in report:
                value = "N/A"
                print(f"  {strategy}: {value}")
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
                    
            if metric_key in ['total_return', 'annual_return', 'max_drawdown']:
                print(f"  {strategy}: {value*100:.2f}{unit}")
            else:
                print(f"  {strategy}: {value}{unit}")
        
        if best_strategy:
            print(f"  最优策略: {best_strategy}")

if __name__ == "__main__":
    main() 