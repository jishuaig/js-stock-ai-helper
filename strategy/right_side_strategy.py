import pandas as pd
import numpy as np
from typing import Dict
from strategy.base_strategy import BaseStrategy

class RightSideStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        super().__init__(stock_code, initial_capital)
        self.position_levels = 3  # 分批建仓的层数
        self.current_position_level = 0  # 当前建仓层级
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 趋势判断指标
        # 1. 移动平均线
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['MA50'] = df['收盘'].rolling(window=50).mean()
        df['MA120'] = df['收盘'].rolling(window=120).mean()
        
        # 2. MACD
        df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']
        
        # 3. 相对强弱指数RSI
        delta = df['收盘'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. 波动率指标ATR (Average True Range)
        high_low = df['最高'] - df['最低']
        high_close = (df['最高'] - df['收盘'].shift()).abs()
        low_close = (df['最低'] - df['收盘'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # 5. 价格动量
        df['Price_Momentum'] = df['收盘'].pct_change(periods=5)
        
        # 6. 成交量变化
        df['Volume_Change'] = df['成交量'].pct_change(periods=5)
        
        # 7. 均线多头排列
        df['Bullish_Alignment'] = ((df['MA20'] > df['MA50']) & (df['MA50'] > df['MA120'])).astype(int)
        
        return df
        
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取右侧交易信号"""
        if self.data is None:
            return 'hold'
            
        # 获取当前指标值
        current_price = row['收盘']
        current_ma20 = row['MA20']
        current_ma50 = row['MA50']
        current_ma120 = row['MA120']
        current_macd = row['MACD_Line']
        current_signal = row['Signal_Line']
        current_histogram = row['MACD_Histogram']
        current_rsi = row['RSI']
        current_bullish = row['Bullish_Alignment']
        current_momentum = row['Price_Momentum']
        current_volume = row['Volume_Change']
        
        # 右侧买入条件：趋势确认后入场 (放宽条件)
        if self.position == 0:  # 没有持仓
            # 强势上涨条件 (减少条件)
            uptrend_condition = (
                current_price > current_ma20 and       # 价格在短期均线之上
                current_macd > current_signal and      # MACD金叉或保持多头
                current_rsi > 50 and                   # RSI处于上升通道
                (current_ma20 > current_ma50 or current_momentum > 0)  # 短期均线在中期均线之上 或 价格动量为正
            )
            
            if uptrend_condition:
                # 根据建仓层级决定买入数量
                if self.current_position_level < self.position_levels:
                    self.current_position_level += 1
                    return 'buy'
                    
        # 右侧卖出条件：趋势转向后出场
        elif self.position > 0:  # 有持仓
            # 趋势转向条件
            downtrend_condition = (
                (current_price < current_ma20 and current_macd < current_signal) or  # 价格跌破短期均线且MACD死叉
                current_rsi > 75 or                       # RSI超买
                (current_rsi < 40 and current_macd < 0)   # RSI转弱且MACD在零线下方
            )
            
            if downtrend_condition:
                self.current_position_level = 0  # 重置建仓层级
                return 'sell'
                
        return 'hold' 