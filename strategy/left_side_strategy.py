import pandas as pd
import numpy as np
from typing import Dict
from strategy.base_strategy import BaseStrategy

class LeftSideStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        super().__init__(stock_code, initial_capital)
        self.position_levels = 3  # 分批建仓的层数
        self.current_position_level = 0  # 当前建仓层级
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # RSI
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['STD20'] = df['收盘'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + 2 * df['STD20']
        df['Lower'] = df['MA20'] - 2 * df['STD20']
        
        # 计算价格动量
        df['Price_Momentum'] = df['收盘'].pct_change(periods=5)
        
        # 计算成交量变化
        df['Volume_Change'] = df['成交量'].pct_change(periods=5)
        
        return df
        
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取左侧交易信号"""
        if self.data is None:
            return 'hold'
            
        # 获取当前指标值
        current_rsi = row['RSI']
        current_price = row['收盘']
        current_lower = row['Lower']
        current_ma20 = row['MA20']
        current_momentum = row['Price_Momentum']
        current_volume = row['Volume_Change']
        
        # 左侧买入条件
        if self.position == 0:  # 没有持仓
            # 超跌条件
            oversold_condition = (
                current_rsi < 30 and  # RSI超跌
                current_price < current_lower and  # 价格跌破布林带下轨
                current_price < current_ma20 and  # 价格在20日均线下方
                current_momentum < -0.02 and  # 价格动量向下
                current_volume > 0  # 成交量增加
            )
            
            if oversold_condition:
                # 根据建仓层级决定买入数量
                if self.current_position_level < self.position_levels:
                    self.current_position_level += 1
                    return 'buy'
                    
        # 左侧卖出条件
        elif self.position > 0:  # 有持仓
            # 超买条件
            overbought_condition = (
                current_rsi > 70 or  # RSI超买
                current_price > current_ma20 * 1.05 or  # 价格突破20日均线5%以上
                current_momentum > 0.02  # 价格动量向上
            )
            
            if overbought_condition:
                self.current_position_level = 0  # 重置建仓层级
                return 'sell'
                
        return 'hold' 