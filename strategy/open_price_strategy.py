import pandas as pd
from typing import Dict
from strategy.base_strategy import BaseStrategy

class OpenPriceStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0, 
                 high_threshold: float = 0.005, low_threshold: float = 0.005):
        """
        初始化开盘价策略
        :param stock_code: 股票代码
        :param initial_capital: 初始资金
        :param high_threshold: 高开阈值，默认2%
        :param low_threshold: 低开阈值，默认2%
        """
        super().__init__(stock_code, initial_capital)
        self.high_threshold = high_threshold  # 高开阈值
        self.low_threshold = low_threshold    # 低开阈值
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 确保数据包含'开盘'列
        if '开盘' not in df.columns:
            print("数据缺少'开盘'列，请检查数据格式")
            return df
        
        # 计算当天开盘价与前一天收盘价的差距百分比
        df['前日收盘'] = df['收盘'].shift(1)
        df['开盘差幅'] = (df['开盘'] - df['前日收盘']) / df['前日收盘']
        
        # 标记高开和低开
        df['高开'] = df['开盘差幅'] > self.high_threshold
        df['低开'] = df['开盘差幅'] < -self.low_threshold
        
        return df
        
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取交易信号"""
        if pd.isna(row.get('前日收盘')):
            return 'hold'  # 第一个交易日没有前日收盘价，不交易
        
        # 高开买入信号
        if row['高开'] and self.position == 0:
            return 'buy'
            
        # 低开卖出信号
        if row['低开'] and self.position > 0:
            return 'sell'
            
        return 'hold'