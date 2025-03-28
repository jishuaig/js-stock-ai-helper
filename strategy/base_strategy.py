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
            is_etf = self.stock_code.startswith('51') or self.stock_code.startswith('15')
            
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
                '收盘': '收盘',
                'close': '收盘',
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
            required_columns = ['收盘', '成交量']
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
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 计算最大回撤
        daily_returns = pd.Series([d['capital'] for d in self.daily_capital])
        rolling_max = daily_returns.expanding().max()
        drawdowns = (daily_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 计算夏普比率
        daily_returns = pd.Series([d['capital'] for d in self.daily_capital]).pct_change()
        std_dev = daily_returns.std()
        if std_dev == 0 or np.isnan(std_dev):
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / std_dev
        
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