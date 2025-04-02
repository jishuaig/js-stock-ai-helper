import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strategy.open_price_strategy import OpenPriceStrategy
from datetime import datetime
import matplotlib as mpl

def test_open_price_strategy_with_real_data():
    # 设置绘图风格
    sns.set_style('whitegrid')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统可用的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 初始化策略，使用优化后的参数和新的仓位管理参数
    strategy = OpenPriceStrategy(
        stock_code="563300",
        initial_capital=50000.0,
        high_threshold=0.0005,  # 使用之前优化的值
        low_threshold=0.00,   # 使用之前优化的值
        # 新增仓位管理参数 (使用默认值)
        position_sizes=[0.5, 0.5],
        max_total_position_ratio=0.9,
        atr_periods=14,
        risk_per_trade_ratio=0.02,
        atr_risk_multiplier=2.0
    )
    
    # 获取历史数据
    df = strategy.fetch_data('20250101', '20250330') 
    if df is None:
        pytest.fail("无法获取历史数据")
    
    # 运行回测
    final_portfolio = strategy.backtest(df)
    
    # 打印回测结果
    print(f"\nBacktest Results:")
    print(f"Initial Capital: {strategy.initial_capital:,.2f}")
    print(f"Final Capital: {final_portfolio['final_capital']:,.2f}")
    print(f"Total Return: {final_portfolio['returns']:.2%}")
    print(f"\nStrategy Metrics:")
    print(f"Sharpe Ratio: {final_portfolio['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {final_portfolio['metrics']['max_drawdown']:.2%}")
    print(f"Annual Return: {final_portfolio['metrics']['annual_return']:.2%}")
    print(f"Total Trades: {final_portfolio['metrics']['total_trades']}")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    fig.suptitle('ZZ2000ETF Opening Price Strategy Backtest', fontsize=14, y=0.95)
    
    # 绘制收益曲线（上图）
    daily_capital = pd.DataFrame(final_portfolio['daily_capital'])
    daily_capital.set_index('date', inplace=True)
    
    # 计算策略收益率
    strategy_returns = (daily_capital['capital'] - strategy.initial_capital) / strategy.initial_capital * 100
    
    # 计算基准收益率
    benchmark_returns = (df['收盘'] / df['收盘'].iloc[0] - 1) * 100
    
    # 绘制策略收益率
    ax1.plot(strategy_returns.index, strategy_returns, 
            label='Strategy Return', color='#1f77b4', linewidth=2)
    
    # 绘制基准收益率
    ax1.plot(benchmark_returns.index, benchmark_returns, 
            label='Benchmark Return', color='#ff7f0e', linestyle='--', linewidth=2)
    
    # 设置上图属性
    ax1.set_title('Return Comparison', pad=15)
    ax1.set_xlabel('')
    ax1.set_ylabel('Return (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # 添加0线
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 格式化x轴日期
    ax1.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
    
    # 绘制交易点位（下图）
    ax2.plot(df.index, df['收盘'], color='gray', alpha=0.5, label='Close Price')
    
    # 标记买入点和卖出点
    buy_points = [(t['date'], t['price']) for t in final_portfolio['transactions'] if t['type'] == 'buy']
    sell_points = [(t['date'], t['price']) for t in final_portfolio['transactions'] if t['type'] == 'sell']
    
    if buy_points:
        buy_dates, buy_prices = zip(*buy_points)
        ax2.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, label='Buy Points')
    
    if sell_points:
        sell_dates, sell_prices = zip(*sell_points)
        ax2.scatter(sell_dates, sell_prices, color='green', marker='v', s=100, label='Sell Points')
    
    # 设置下图属性
    ax2.set_title('Trading Points', pad=15)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')
    
    # 格式化x轴日期
    ax2.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    # 打印交易记录
    print("\nTransaction History:")
    if final_portfolio['transactions']:
        # 定义列宽
        col_width_date = 12
        col_width_type = 8
        col_width_price = 10
        col_width_shares = 10
        col_width_amount = 15
        col_width_balance = 15
        col_width_total = 15
        
        # 创建表格样式的交易记录
        print("┌" + "─" * col_width_date + "┬" + "─" * col_width_type + "┬" + 
              "─" * col_width_price + "┬" + "─" * col_width_shares + "┬" + 
              "─" * col_width_amount + "┬" + "─" * col_width_balance + "┬" + 
              "─" * col_width_total + "┐")
        
        print("│{:^{}}│{:^{}}│{:^{}}│{:^{}}│{:^{}}│{:^{}}│{:^{}}│".format(
            "Date", col_width_date,
            "Type", col_width_type,
            "Price", col_width_price,
            "Shares", col_width_shares,
            "Amount", col_width_amount,
            "Balance", col_width_balance,
            "Total", col_width_total
        ))
        
        print("├" + "─" * col_width_date + "┼" + "─" * col_width_type + "┼" + 
              "─" * col_width_price + "┼" + "─" * col_width_shares + "┼" + 
              "─" * col_width_amount + "┼" + "─" * col_width_balance + "┼" + 
              "─" * col_width_total + "┤")
        
        # 跟踪当前持仓
        current_position = 0
        
        for trade in final_portfolio['transactions']:
            date_str = trade['date'].strftime('%Y-%m-%d')
            trade_type = trade['type']
            price = trade['price']
            shares = trade['shares']
            
            # 获取交易金额和账户余额
            if trade_type == 'buy':
                amount = trade.get('cost', price * shares)
                amount_str = f"-{amount:.2f}"
                current_position += shares
            else:  # sell
                amount = trade.get('revenue', price * shares)
                amount_str = f"+{amount:.2f}"
                current_position = 0  # 在策略中，sell是清仓操作
            
            balance = trade.get('capital_after', 0)
            
            # 计算总资产价值 (现金 + 持仓市值)
            total_value = balance + (current_position * price)
            
            print("│{:^{}}│{:^{}}│{:^{}.3f}│{:^{}}│{:^{}}│{:^{}.2f}│{:^{}.2f}│".format(
                date_str, col_width_date,
                trade_type, col_width_type,
                price, col_width_price,
                shares, col_width_shares,
                amount_str, col_width_amount,
                balance, col_width_balance,
                total_value, col_width_total
            ))
        
        print("└" + "─" * col_width_date + "┴" + "─" * col_width_type + "┴" + 
              "─" * col_width_price + "┴" + "─" * col_width_shares + "┴" + 
              "─" * col_width_amount + "┴" + "─" * col_width_balance + "┴" + 
              "─" * col_width_total + "┘")
    else:
        print("No transactions found")

if __name__ == '__main__':
    test_open_price_strategy_with_real_data()