#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较左侧和右侧交易策略的表现
"""

from strategy import LeftSideStrategy, RightSideStrategy
from backtest import compare_strategies
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    # 设置回测参数
    stock_code = "510300"  # 沪深300ETF
    start_date = "20230101"
    end_date = "20240328"
    initial_capital = 100000
    stop_loss = 0.1  # 止损比例
    take_profit = 0.2  # 止盈比例
    
    print("正在对比左侧交易策略和右侧交易策略...")
    
    # 对比两个策略
    report1, report2 = compare_strategies(
        LeftSideStrategy,
        RightSideStrategy,
        stock_code,
        start_date,
        end_date,
        initial_capital,
        stop_loss,
        take_profit,
        transaction_fee=0.0003
    )
    
    # 打印左侧策略的回测结果
    print(f"\n=== {LeftSideStrategy.__name__}回测结果 ===")
    if "error" in report1:
        print(f"回测失败: {report1['error']}")
    else:
        print(f"初始资金: {initial_capital:.2f}")
        print(f"最终资金: {report1['final_capital']:.2f}")
        print(f"总收益率: {report1['total_return']*100:.2f}%")
        print(f"年化收益率: {report1['annual_return']*100:.2f}%")
        print(f"最大回撤: {report1['max_drawdown']*100:.2f}%")
        print(f"夏普比率: {report1['sharpe_ratio']:.2f}")
        print(f"总交易次数: {report1['total_trades']}")
    
    # 打印右侧策略的回测结果
    print(f"\n=== {RightSideStrategy.__name__}回测结果 ===")
    if "error" in report2:
        print(f"回测失败: {report2['error']}")
    else:
        print(f"初始资金: {initial_capital:.2f}")
        print(f"最终资金: {report2['final_capital']:.2f}")
        print(f"总收益率: {report2['total_return']*100:.2f}%")
        print(f"年化收益率: {report2['annual_return']*100:.2f}%")
        print(f"最大回撤: {report2['max_drawdown']*100:.2f}%")
        print(f"夏普比率: {report2['sharpe_ratio']:.2f}")
        print(f"总交易次数: {report2['total_trades']}")
    
    # 比较表现指标
    print("\n=== 策略比较 ===")
    if "error" not in report1 and "error" not in report2:
        metrics = [
            ("总收益率", report1["total_return"]*100, report2["total_return"]*100, "%"),
            ("年化收益率", report1["annual_return"]*100, report2["annual_return"]*100, "%"),
            ("最大回撤", report1["max_drawdown"]*100, report2["max_drawdown"]*100, "%"),
            ("夏普比率", report1["sharpe_ratio"], report2["sharpe_ratio"], ""),
            ("总交易次数", report1["total_trades"], report2["total_trades"], "")
        ]
        
        for name, val1, val2, unit in metrics:
            better = "左侧策略" if val1 > val2 else "右侧策略"
            if name == "最大回撤":  # 回撤越小（绝对值越小）越好
                better = "左侧策略" if abs(val1) < abs(val2) else "右侧策略"
            diff = abs(val1 - val2)
            print(f"{name}: 左侧策略={val1:.2f}{unit} vs 右侧策略={val2:.2f}{unit}, 差异={diff:.2f}{unit}, 更优: {better}")

if __name__ == "__main__":
    main() 