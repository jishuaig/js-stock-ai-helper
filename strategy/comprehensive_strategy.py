import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy

class ComprehensiveStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        super().__init__(stock_code, initial_capital)
        self.position_levels = 3  # 分批建仓的层数
        self.current_position_level = 0  # 当前建仓层级
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算移动平均线
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['MA60'] = df['收盘'].rolling(window=60).mean()
        
        # 计算MACD
        exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # 计算RSI
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['BB_Middle'] = df['收盘'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['收盘'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['收盘'].rolling(window=20).std()
        
        # 计算价格动量
        df['Price_Momentum'] = df['收盘'].pct_change(periods=5)
        
        # 计算成交量变化
        df['Volume_Change'] = df['成交量'].pct_change(periods=5)
        
        return df
        
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取交易信号"""
        if self.data is None:
            return 'hold'
            
        # 获取当前指标值
        current_price = row['收盘']
        current_ma5 = row['MA5']
        current_ma20 = row['MA20']
        current_ma60 = row['MA60']
        current_rsi = row['RSI']
        current_macd = row['MACD']
        current_signal = row['Signal']
        current_momentum = row['Price_Momentum']
        current_volume = row['Volume_Change']
        
        # 计算趋势得分
        trend_score = 0
        if current_price > current_ma5 > current_ma20:
            trend_score += 2
        elif current_price > current_ma5:
            trend_score += 1
            
        if current_ma5 > current_ma20 > current_ma60:
            trend_score += 2
        elif current_ma5 > current_ma20:
            trend_score += 1
            
        # 计算动量得分
        momentum_score = 0
        if current_macd > current_signal:
            momentum_score += 1
        if current_rsi > 50:
            momentum_score += 1
        if current_momentum > 0:
            momentum_score += 1
            
        # 计算成交量得分
        volume_score = 0
        if current_volume > 0:
            volume_score += 1
            
        # 买入条件
        if self.position == 0:  # 没有持仓
            if (trend_score >= 3 and  # 趋势强势
                momentum_score >= 2 and  # 动量强势
                volume_score >= 1 and  # 成交量配合
                current_rsi < 70):  # 非超买
                
                if self.current_position_level < self.position_levels:
                    self.current_position_level += 1
                    return 'buy'
                    
        # 卖出条件
        elif self.position > 0:  # 有持仓
            if (trend_score <= 1 or  # 趋势转弱
                momentum_score <= 1 or  # 动量转弱
                current_rsi > 70 or  # 超买
                current_price < current_ma5):  # 跌破5日均线
                
                self.current_position_level = 0
                return 'sell'
                
        return 'hold' 