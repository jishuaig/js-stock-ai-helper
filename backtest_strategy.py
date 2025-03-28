import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from stock_analysis_comprehensive import ComprehensiveStockAnalyzer
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class BacktestSystem:
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        self.stock_code = stock_code
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0  # 持仓数量
        self.trades = []  # 交易记录
        self.daily_capital = []  # 每日资金记录
        self.analyzer = ComprehensiveStockAnalyzer(stock_code)
        self.data = None
        
    def run_backtest(self, start_date: str, end_date: str, 
                    stop_loss: float = 0.1, take_profit: float = 0.2,
                    transaction_fee: float = 0.0003) -> Dict:
        """运行回测"""
        # 获取历史数据
        df = self.analyzer.fetch_data(start_date, end_date)
        if df is None:
            return {"error": "获取数据失败"}
            
        # 初始化回测结果
        self.trades = []
        self.daily_capital = []
        self.current_capital = self.initial_capital
        self.position = 0
        self.data = df
        
        # 记录每日资金
        for date, row in df.iterrows():
            # 计算当日资金
            daily_value = self.current_capital + self.position * row['收盘']
            self.daily_capital.append({
                'date': date,
                'capital': daily_value
            })
            
            # 获取交易信号
            signal = self._get_trading_signal(row)
            
            # 执行交易
            if signal == 'buy' and self.position == 0:
                # 计算可买入数量
                shares = int(self.current_capital / (row['收盘'] * (1 + transaction_fee)))
                if shares > 0:
                    cost = shares * row['收盘'] * (1 + transaction_fee)
                    self.current_capital -= cost
                    self.position = shares
                    self.trades.append({
                        'date': date,
                        'type': 'buy',
                        'price': row['收盘'],
                        'shares': shares,
                        'cost': cost
                    })
                    
            elif signal == 'sell' and self.position > 0:
                # 计算卖出收益
                revenue = self.position * row['收盘'] * (1 - transaction_fee)
                self.current_capital += revenue
                self.trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': row['收盘'],
                    'shares': self.position,
                    'revenue': revenue
                })
                self.position = 0
                
            # 检查止损止盈
            if self.position > 0:
                entry_price = self.trades[-1]['price'] if self.trades else 0
                if entry_price > 0:
                    current_return = (row['收盘'] - entry_price) / entry_price
                    if current_return <= -stop_loss or current_return >= take_profit:
                        # 触发止损止盈
                        revenue = self.position * row['收盘'] * (1 - transaction_fee)
                        self.current_capital += revenue
                        self.trades.append({
                            'date': date,
                            'type': 'stop_loss_take_profit',
                            'price': row['收盘'],
                            'shares': self.position,
                            'revenue': revenue,
                            'return': current_return
                        })
                        self.position = 0
                        
        # 计算回测结果
        final_capital = self.current_capital + self.position * df['收盘'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        days = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 计算最大回撤
        daily_returns = pd.Series([d['capital'] for d in self.daily_capital])
        rolling_max = daily_returns.expanding().max()
        drawdowns = (daily_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 生成回测报告
        report = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'trades': self.trades,
            'daily_capital': self.daily_capital
        }
        
        return report
        
    def _get_trading_signal(self, row: pd.Series) -> str:
        """获取交易信号"""
        # 使用分析器的指标生成交易信号
        if self.data is None:
            return 'hold'
        
        df = self.data
        current_date = row.name
        
        # 更新分析器的数据
        self.analyzer.data = df[:current_date]
        indicators = self.analyzer.calculate_indicators()
        analysis = self.analyzer.analyze()
        
        # 根据分析结果生成信号
        if analysis['summary']['overall_status'] == '强势' and self.position == 0:
            return 'buy'
        elif analysis['summary']['overall_status'] == '弱势' and self.position > 0:
            return 'sell'
        return 'hold'
        
    def plot_capital_curve(self):
        """绘制资金曲线"""
        if not self.daily_capital:
            return
            
        dates = [d['date'] for d in self.daily_capital]
        capital = [d['capital'] for d in self.daily_capital]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, capital)
        plt.title('Capital Curve')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # 测试回测系统
    stock_code = "sh002456"  # 欧菲光
    initial_capital = 100000.0
    start_date = "20230101"
    end_date = "20240321"
    
    backtest = BacktestSystem(stock_code, initial_capital)
    # 调整止损止盈参数和交易费率
    report = backtest.run_backtest(start_date, end_date, 
                                 stop_loss=0.2,  # 进一步放宽止损
                                 take_profit=0.4,  # 提高止盈
                                 transaction_fee=0.0005)  # 考虑更合理的交易费率
    
    print("\n=== 回测结果 ===")
    print(f"初始资金: {report['initial_capital']:.2f}")
    print(f"最终资金: {report['final_capital']:.2f}")
    print(f"总收益率: {report['total_return']*100:.2f}%")
    print(f"年化收益率: {report['annual_return']*100:.2f}%")
    print(f"最大回撤: {report['max_drawdown']*100:.2f}%")
    print(f"总交易次数: {report['total_trades']}")
    
    # 打印交易记录
    print("\n=== 交易记录 ===")
    for trade in report['trades']:
        trade_info = {
            'date': trade['date'].strftime('%Y-%m-%d'),
            'type': trade['type'],
            'price': trade['price'],
            'shares': trade['shares']
        }
        if 'cost' in trade:
            trade_info['cost'] = trade['cost']
        if 'revenue' in trade:
            trade_info['revenue'] = trade['revenue']
        if 'return' in trade:
            trade_info['return'] = trade['return']
        print(json.dumps(trade_info, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 