from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import akshare as ak
import matplotlib.pyplot as plt

class BaseStrategy(ABC):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        """
        初始化策略基类
        :param stock_code: 股票代码
        :param initial_capital: 初始资金
        """
        self.stock_code = stock_code
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0  # 持仓数量
        self.trades = []  # 交易记录
        self.daily_capital = []  # 每日资金记录
        self.data = None  # 历史数据
        
    def fetch_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: DataFrame或None
        """
        try:
            # 验证日期
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            today = pd.to_datetime('today')
            
            if start > end:
                print("开始日期不能晚于结束日期")
                return None
                
            if end > today:
                print("结束日期不能晚于今天")
                return None
            
            # 判断是否为ETF
            is_etf = (self.stock_code.startswith('51') or 
                     self.stock_code.startswith('15') or 
                     self.stock_code.startswith('56') or 
                     self.stock_code.startswith('159'))
            
            if is_etf:
                df = ak.fund_etf_hist_em(symbol=self.stock_code, 
                                        start_date=start_date, 
                                        end_date=end_date, 
                                        adjust="qfq")
            else:
                df = ak.stock_zh_a_hist(symbol=self.stock_code, 
                                      start_date=start_date, 
                                      end_date=end_date, 
                                      adjust="qfq")
            
            # 统一列名
            column_mapping = {
                '日期': '日期',
                'date': '日期',
                '开盘': '开盘',
                'open': '开盘',
                '收盘': '收盘',
                'close': '收盘',
                '最高': '最高',
                'high': '最高',
                '最低': '最低',
                'low': '最低',
                '成交量': '成交量',
                'volume': '成交量'
            }
            
            df = df.rename(columns=column_mapping)
            
            if '日期' not in df.columns:
                print("数据格式不正确，请检查数据列名")
                return None
            
            # 转换日期格式
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
                
            # 确保必要的列存在
            required_columns = ['开盘', '收盘', '最高', '最低', '成交量']
            if not all(col in df.columns for col in required_columns):
                print("数据缺少必要的列，请检查数据格式")
                return None
                
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None
            
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        :param df: 输入数据
        :return: 添加指标后的DataFrame
        """
        pass
        
    @abstractmethod
    def get_trading_signal(self, row: pd.Series) -> str:
        """
        获取交易信号
        :param row: 当前数据行
        :return: 交易信号
        """
        pass
        
    def calculate_performance_metrics(self) -> Dict:
        """
        计算策略表现指标
        :return: 表现指标字典
        """
        if not self.daily_capital:
            return {}
            
        # 计算总收益率
        final_capital = self.daily_capital[-1]['capital']
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        days = (self.daily_capital[-1]['date'] - self.daily_capital[0]['date']).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        daily_returns = pd.Series([d['capital'] for d in self.daily_capital])
        rolling_max = daily_returns.expanding().max()
        drawdowns = (daily_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 计算夏普比率
        daily_returns = pd.Series([d['capital'] for d in self.daily_capital]).pct_change()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades)
        }
        
    def plot_capital_curve(self):
        """
        绘制资金曲线
        """
        if not self.daily_capital:
            return
            
        dates = [d['date'] for d in self.daily_capital]
        capital = [d['capital'] for d in self.daily_capital]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, capital)
        plt.title(f'{self.__class__.__name__}资金曲线')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        执行回测
        :param df: 股票数据，包含必要的价格和成交量数据
        :return: 回测结果字典
        """
        # 重置状态
        self.current_capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.daily_capital = []
        
        # 保存数据并计算指标
        self.data = df.copy()
        if self.data.empty:
            print("输入数据为空，无法执行回测")
            return self.get_results()
            
        self.data = self.calculate_indicators(self.data)
        
        # 遍历每个交易日
        for index, row in self.data.iterrows():
            # 获取当前价格
            current_price = row['收盘']
            
            # 获取交易信号
            signal = self.get_trading_signal(row)
            
            # 执行交易
            if signal == 'buy' and self.position == 0:
                # 计算可买入数量（假设使用90%的资金，考虑手续费）
                max_shares = int((self.current_capital * 0.9) / current_price / 100) * 100
                if max_shares > 0:
                    self.position = max_shares
                    cost = current_price * self.position * (1 + 0.0003)  # 考虑手续费0.03%
                    self.current_capital -= cost
                    self.trades.append({
                        'date': index,
                        'type': 'buy',
                        'price': current_price,
                        'shares': self.position,
                        'cost': cost,
                        'capital': self.current_capital
                    })
            
            elif signal == 'sell' and self.position > 0:
                # 卖出所有持仓
                revenue = current_price * self.position * (1 - 0.0003)  # 考虑手续费0.03%
                self.current_capital += revenue
                self.trades.append({
                    'date': index,
                    'type': 'sell',
                    'price': current_price,
                    'shares': self.position,
                    'revenue': revenue,
                    'capital': self.current_capital
                })
                self.position = 0
            
            # 记录每日资金情况
            daily_capital = self.current_capital
            if self.position > 0:
                daily_capital += self.position * current_price
            self.daily_capital.append({
                'date': index,
                'capital': daily_capital
            })
        
        return self.get_results()
        
    def get_results(self):
        """获取回测结果"""
        if not self.data.empty and self.daily_capital:
            # 计算基本指标
            initial_capital = self.initial_capital
            final_capital = self.daily_capital[-1]['capital']
            total_return = (final_capital - initial_capital) / initial_capital
            
            # 计算年化收益率
            first_date = self.daily_capital[0]['date']
            last_date = self.daily_capital[-1]['date']
            if isinstance(first_date, str):
                first_date = pd.to_datetime(first_date)
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            # 计算交易天数
            try:
                days = (last_date - first_date).days
                if days <= 0:
                    days = 1  # 防止除以零
                annual_return = total_return * (365 / days)
            except:
                annual_return = total_return  # 如果日期计算失败，直接使用总收益
            
            # 计算胜率
            if len(self.trades) > 0:
                wins = 0
                for i in range(0, len(self.trades), 2):
                    if i + 1 < len(self.trades):
                        buy_trade = self.trades[i]
                        sell_trade = self.trades[i + 1]
                        if sell_trade['type'] == 'sell' and buy_trade['type'] == 'buy':
                            if sell_trade.get('revenue', 0) > buy_trade.get('cost', 0):
                                wins += 1
                win_rate = wins / (len(self.trades) // 2) if len(self.trades) >= 2 else 0
            else:
                win_rate = 0
            
            # 计算最大回撤
            max_drawdown = 0
            peak = initial_capital
            for daily in self.daily_capital:
                capital = daily['capital']
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # 计算夏普比率（假设无风险收益率为3%）
            if len(self.daily_capital) > 1:
                daily_returns = []
                for i in range(1, len(self.daily_capital)):
                    prev_capital = self.daily_capital[i-1]['capital']
                    curr_capital = self.daily_capital[i]['capital']
                    daily_return = (curr_capital - prev_capital) / prev_capital
                    daily_returns.append(daily_return)
                
                # 计算夏普比率
                risk_free_rate = 0.03  # 假设无风险收益率为3%
                if len(daily_returns) > 0:
                    avg_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns)
                    if std_return > 0:
                        sharpe_ratio = (avg_return - risk_free_rate / 365) / std_return * np.sqrt(252)
                    else:
                        sharpe_ratio = 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'trade_count': len(self.trades),
                'daily_capital': self.daily_capital,
                'trades': self.trades
            }
        else:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'trade_count': 0,
                'daily_capital': [],
                'trades': []
            }
        
    def plot_results(self):
        """绘制回测结果"""
        if not self.daily_capital:
            print("没有回测数据可绘制")
            return
            
        # 创建一个新的图表
        plt.figure(figsize=(12, 8))
        
        # 提取日期和资金数据
        dates = [item['date'] for item in self.daily_capital]
        capital = [item['capital'] for item in self.daily_capital]
        
        # 绘制权益曲线
        plt.subplot(2, 1, 1)
        plt.plot(dates, capital, label='资金曲线', color='blue')
        plt.title(f'{self.stock_code} 策略回测结果')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.grid(True)
        plt.legend()
        
        # 标记买入点和卖出点
        buys = [trade for trade in self.trades if trade['type'] == 'buy']
        sells = [trade for trade in self.trades if trade['type'] == 'sell']
        
        if buys:
            buy_dates = [trade['date'] for trade in buys]
            buy_prices = [trade.get('cost', 0) / trade.get('shares', 1) for trade in buys]
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='买入')
            
        if sells:
            sell_dates = [trade['date'] for trade in sells]
            sell_prices = [trade.get('revenue', 0) / trade.get('shares', 1) for trade in sells]
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='卖出')
            
        # 计算每日收益率
        daily_returns = []
        for i in range(1, len(self.daily_capital)):
            prev_capital = self.daily_capital[i-1]['capital']
            curr_capital = self.daily_capital[i]['capital']
            daily_return = (curr_capital - prev_capital) / prev_capital
            daily_returns.append(daily_return)
            
        # 绘制每日收益率
        plt.subplot(2, 1, 2)
        plt.plot(dates[1:], daily_returns, label='每日收益率', color='purple')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('日期')
        plt.ylabel('每日收益率')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout() 

if __name__ == "__main__":
    base_strategy = BaseStrategy("563300", 100000.0)
    df = base_strategy.fetch_data("20240101", "20240830")
    print
