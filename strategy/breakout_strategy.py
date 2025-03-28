import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        super().__init__(stock_code, initial_capital)
        self.breakout_period = 20  # 突破周期
        self.volume_filter = True  # 成交量过滤
        self.confirmation_days = 2  # 突破确认天数
        self.current_confirmation = 0  # 当前确认天数
        self.breakout_direction = None  # 当前突破方向
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算突破相关指标
        df['High_20'] = df['最高'].rolling(window=self.breakout_period).max()
        df['Low_20'] = df['最低'].rolling(window=self.breakout_period).min()
        
        # 识别高点和低点突破
        df['High_Breakout'] = (df['收盘'] > df['High_20'].shift(1)).astype(int)
        df['Low_Breakout'] = (df['收盘'] < df['Low_20'].shift(1)).astype(int)
        
        # 计算均线
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['MA50'] = df['收盘'].rolling(window=50).mean()
        
        # 计算成交量指标
        df['Volume_MA20'] = df['成交量'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['成交量'] / df['Volume_MA20']
        
        # 计算ATR（Average True Range）波动率指标
        high_low = df['最高'] - df['最低']
        high_close = (df['最高'] - df['收盘'].shift()).abs()
        low_close = (df['最低'] - df['收盘'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
        
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取突破交易信号"""
        if self.data is None:
            return 'hold'
            
        current_price = row['收盘']
        current_high_breakout = row['High_Breakout']
        current_low_breakout = row['Low_Breakout']
        current_volume_ratio = row['Volume_Ratio']
        current_atr = row['ATR']
        current_ma20 = row['MA20']
        current_ma50 = row['MA50']
        
        # 向上突破条件
        if self.position == 0:  # 没有持仓
            # 上升突破
            if current_high_breakout == 1:
                # 成交量确认（成交量大于20日均量的1.5倍）
                if not self.volume_filter or current_volume_ratio > 1.5:
                    if self.breakout_direction != 'up':
                        self.breakout_direction = 'up'
                        self.current_confirmation = 1
                    else:
                        self.current_confirmation += 1
                    
                    # 突破确认
                    if self.current_confirmation >= self.confirmation_days:
                        return 'buy'
        
        # 向下突破条件
        elif self.position > 0:  # 有持仓
            # 下降突破
            if current_low_breakout == 1:
                # 成交量确认
                if not self.volume_filter or current_volume_ratio > 1.5:
                    if self.breakout_direction != 'down':
                        self.breakout_direction = 'down'
                        self.current_confirmation = 1
                    else:
                        self.current_confirmation += 1
                    
                    # 突破确认
                    if self.current_confirmation >= self.confirmation_days:
                        return 'sell'
            
            # 或者价格跌破20日均线且均线向下
            if current_price < current_ma20 and current_ma20 < current_ma20:
                return 'sell'
                
        # 重置确认状态
        if (self.breakout_direction == 'up' and current_price < row['High_20'] - current_atr) or \
           (self.breakout_direction == 'down' and current_price > row['Low_20'] + current_atr):
            self.current_confirmation = 0
            self.breakout_direction = None
                
        return 'hold' 