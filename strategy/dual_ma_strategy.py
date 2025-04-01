import pandas as pd
import numpy as np
from typing import Dict
from strategy.base_strategy import BaseStrategy

class DualMAStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        super().__init__(stock_code, initial_capital)
        self.short_period = 10  # 短期均线周期
        self.long_period = 30   # 长期均线周期
        self.signal_filter = True  # 是否使用过滤器
        self.last_cross_type = None  # 上次交叉类型（金叉或死叉）
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算双均线
        df[f'MA{self.short_period}'] = df['收盘'].rolling(window=self.short_period).mean()
        df[f'MA{self.long_period}'] = df['收盘'].rolling(window=self.long_period).mean()
        
        # 计算均线交叉信号
        df['Golden_Cross'] = ((df[f'MA{self.short_period}'] > df[f'MA{self.long_period}']) & 
                             (df[f'MA{self.short_period}'].shift(1) <= df[f'MA{self.long_period}'].shift(1))).astype(int)
        df['Death_Cross'] = ((df[f'MA{self.short_period}'] < df[f'MA{self.long_period}']) & 
                            (df[f'MA{self.short_period}'].shift(1) >= df[f'MA{self.long_period}'].shift(1))).astype(int)
        
        # 计算MACD用于过滤信号
        df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']
        
        # 计算相对强弱指数RSI
        delta = df['收盘'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算ADX趋势强度指标
        tr1 = abs(df['最高'] - df['最低'])
        tr2 = abs(df['最高'] - df['收盘'].shift(1))
        tr3 = abs(df['最低'] - df['收盘'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # +DI和-DI
        df['Plus_DM'] = ((df['最高'] - df['最高'].shift(1)) > (df['最低'].shift(1) - df['最低'])) & ((df['最高'] - df['最高'].shift(1)) > 0)
        df['Plus_DM'] = df['Plus_DM'] * (df['最高'] - df['最高'].shift(1))
        df['Minus_DM'] = ((df['最低'].shift(1) - df['最低']) > (df['最高'] - df['最高'].shift(1))) & ((df['最低'].shift(1) - df['最低']) > 0)
        df['Minus_DM'] = df['Minus_DM'] * (df['最低'].shift(1) - df['最低'])
        
        df['Plus_DI'] = 100 * (df['Plus_DM'].rolling(window=14).mean() / df['ATR'])
        df['Minus_DI'] = 100 * (df['Minus_DM'].rolling(window=14).mean() / df['ATR'])
        
        # ADX
        df['DX'] = 100 * (abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI']).replace(0, 0.0001))
        df['ADX'] = df['DX'].rolling(window=14).mean()
        
        return df
        
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取双均线交易信号"""
        if self.data is None:
            return 'hold'
            
        current_golden_cross = row['Golden_Cross']
        current_death_cross = row['Death_Cross']
        current_macd = row['MACD_Line']
        current_signal = row['Signal_Line']
        current_rsi = row['RSI']
        current_adx = row.get('ADX', 0)  # 安全获取ADX，如果不存在返回0
        
        # 买入信号：金叉且其他指标确认
        if self.position == 0:  # 没有持仓
            if current_golden_cross == 1:
                # 根据需要增加过滤条件
                buy_signal = True
                
                if self.signal_filter:
                    # 确保MACD同向
                    if current_macd <= current_signal:
                        buy_signal = False
                    # 确保RSI不是超买
                    if current_rsi > 70:
                        buy_signal = False
                    # 确保有足够的趋势强度
                    if current_adx < 20:
                        buy_signal = False
                
                if buy_signal:
                    self.last_cross_type = 'golden'
                    return 'buy'
                    
        # 卖出信号：死叉且其他指标确认
        elif self.position > 0:  # 有持仓
            if current_death_cross == 1:
                # 根据需要增加过滤条件
                sell_signal = True
                
                if self.signal_filter:
                    # 确保MACD同向
                    if current_macd >= current_signal:
                        sell_signal = False
                    # 确保RSI不是超卖
                    if current_rsi < 30:
                        sell_signal = False
                
                if sell_signal:
                    self.last_cross_type = 'death'
                    return 'sell'
                    
        return 'hold' 