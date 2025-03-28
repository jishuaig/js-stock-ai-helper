import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Type
from strategy.base_strategy import BaseStrategy

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class BacktestEngine:
    def __init__(self, strategy_class: Type[BaseStrategy], 
                 stock_code: str, 
                 initial_capital: float = 100000.0):
        """
        初始化回测引擎
        :param strategy_class: 策略类（必须是BaseStrategy的子类）
        :param stock_code: 股票代码
        :param initial_capital: 初始资金
        """
        self.strategy = strategy_class(stock_code, initial_capital)
        
    def run_backtest(self, start_date: str, end_date: str, 
                    stop_loss: float = 0.1, 
                    take_profit: float = 0.2,
                    transaction_fee: float = 0.0003) -> Dict:
        """
        运行回测
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param stop_loss: 止损比例
        :param take_profit: 止盈比例
        :param transaction_fee: 交易费用
        :return: 回测结果
        """
        # 获取历史数据
        df = self.strategy.fetch_data(start_date, end_date)
        if df is None:
            return {"error": "获取数据失败"}
            
        # 初始化回测结果
        self.strategy.trades = []
        self.strategy.daily_capital = []
        self.strategy.current_capital = self.strategy.initial_capital
        self.strategy.position = 0
        self.strategy.data = df
        
        # 计算技术指标
        df = self.strategy.calculate_indicators(df)
        
        # 记录每日资金
        for date, row in df.iterrows():
            # 计算当日资金
            daily_value = self.strategy.current_capital + self.strategy.position * row['收盘']
            self.strategy.daily_capital.append({
                'date': date,
                'capital': daily_value
            })
            
            # 获取交易信号
            signal = self.strategy.get_trading_signal(row)
            
            # 执行交易
            if signal == 'buy' and self.strategy.position == 0:
                # 计算可买入数量
                # 检查是否有position_levels属性（适应不同策略）
                if hasattr(self.strategy, 'position_levels') and hasattr(self.strategy, 'current_position_level'):
                    available_capital = self.strategy.current_capital / (self.strategy.position_levels - self.strategy.current_position_level + 1)
                else:
                    available_capital = self.strategy.current_capital
                    
                shares = int(available_capital / (row['收盘'] * (1 + transaction_fee)))
                if shares > 0:
                    cost = shares * row['收盘'] * (1 + transaction_fee)
                    self.strategy.current_capital -= cost
                    self.strategy.position += shares
                    
                    # 创建交易记录
                    trade_record = {
                        'date': date,
                        'type': 'buy',
                        'price': row['收盘'],
                        'shares': shares,
                        'cost': cost
                    }
                    
                    # 如果策略有position_level属性，添加到交易记录
                    if hasattr(self.strategy, 'current_position_level'):
                        trade_record['position_level'] = self.strategy.current_position_level
                        
                    self.strategy.trades.append(trade_record)
                    
            elif signal == 'sell' and self.strategy.position > 0:
                # 计算卖出收益
                revenue = self.strategy.position * row['收盘'] * (1 - transaction_fee)
                self.strategy.current_capital += revenue
                self.strategy.trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': row['收盘'],
                    'shares': self.strategy.position,
                    'revenue': revenue
                })
                self.strategy.position = 0
                
                # 如果策略有current_position_level属性，重置
                if hasattr(self.strategy, 'current_position_level'):
                    self.strategy.current_position_level = 0
                
            # 检查止损止盈
            if self.strategy.position > 0:
                entry_price = self.strategy.trades[-1]['price'] if self.strategy.trades else 0
                if entry_price > 0:
                    current_return = (row['收盘'] - entry_price) / entry_price
                    if current_return <= -stop_loss or current_return >= take_profit:
                        # 触发止损止盈
                        revenue = self.strategy.position * row['收盘'] * (1 - transaction_fee)
                        self.strategy.current_capital += revenue
                        self.strategy.trades.append({
                            'date': date,
                            'type': 'stop_loss_take_profit',
                            'price': row['收盘'],
                            'shares': self.strategy.position,
                            'revenue': revenue,
                            'return': current_return
                        })
                        self.strategy.position = 0
                        
                        # 如果策略有current_position_level属性，重置
                        if hasattr(self.strategy, 'current_position_level'):
                            self.strategy.current_position_level = 0
                        
        # 计算回测结果
        return self.strategy.calculate_performance_metrics()
        
    def plot_results(self):
        """绘制回测结果"""
        if not self.strategy.daily_capital:
            return
            
        dates = [d['date'] for d in self.strategy.daily_capital]
        capital = [d['capital'] for d in self.strategy.daily_capital]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, capital)
        plt.title('资金曲线')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def compare_strategies(strategy1_class: Type[BaseStrategy],
                      strategy2_class: Type[BaseStrategy],
                      stock_code: str,
                      start_date: str,
                      end_date: str,
                      initial_capital: float = 100000.0,
                      stop_loss: float = 0.1,
                      take_profit: float = 0.2,
                      transaction_fee: float = 0.0003) -> Tuple[Dict, Dict]:
    """
    对比两个策略的表现
    :param strategy1_class: 第一个策略类
    :param strategy2_class: 第二个策略类
    :param stock_code: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param initial_capital: 初始资金
    :param stop_loss: 止损比例
    :param take_profit: 止盈比例
    :param transaction_fee: 交易费用
    :return: 两个策略的回测结果
    """
    # 创建两个回测引擎
    engine1 = BacktestEngine(strategy1_class, stock_code, initial_capital)
    engine2 = BacktestEngine(strategy2_class, stock_code, initial_capital)
    
    # 运行回测
    report1 = engine1.run_backtest(start_date, end_date, stop_loss, take_profit, transaction_fee)
    report2 = engine2.run_backtest(start_date, end_date, stop_loss, take_profit, transaction_fee)
    
    # 绘制对比图
    if not engine1.strategy.daily_capital or not engine2.strategy.daily_capital:
        return report1, report2
        
    dates1 = [d['date'] for d in engine1.strategy.daily_capital]
    capital1 = [d['capital'] for d in engine1.strategy.daily_capital]
    dates2 = [d['date'] for d in engine2.strategy.daily_capital]
    capital2 = [d['capital'] for d in engine2.strategy.daily_capital]
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates1, capital1, label=strategy1_class.__name__)
    plt.plot(dates2, capital2, label=strategy2_class.__name__)
    plt.title('策略对比')
    plt.xlabel('日期')
    plt.ylabel('资金')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return report1, report2

def main():
    from strategy import LeftSideStrategy, ComprehensiveStrategy
    
    # 设置回测参数
    stock_code = "510300"  # 沪深300ETF
    start_date = "20230101"
    end_date = "20240328"
    initial_capital = 100000
    stop_loss = 0.1  # 止损比例
    take_profit = 0.2  # 止盈比例
    
    # 对比两个策略
    report1, report2 = compare_strategies(
        LeftSideStrategy,
        ComprehensiveStrategy,
        stock_code,
        start_date,
        end_date,
        initial_capital,
        stop_loss,
        take_profit,
        transaction_fee=0.0003
    )
    
    # 打印策略1的回测结果
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
    
    # 打印策略2的回测结果
    print(f"\n=== {ComprehensiveStrategy.__name__}回测结果 ===")
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

if __name__ == "__main__":
    main() 