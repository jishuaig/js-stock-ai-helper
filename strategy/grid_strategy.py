import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy

class GridStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        super().__init__(stock_code, initial_capital)
        self.grid_levels = 5  # 网格数量
        self.grid_range = 0.15  # 网格价格范围（上下15%）
        self.grids = []  # 存储网格价格
        self.reference_price = None  # 参考价格
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算波动率
        df['Volatility'] = df['收盘'].rolling(window=20).std() / df['收盘'].rolling(window=20).mean()
        
        # 计算相对强弱指数RSI
        delta = df['收盘'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['STD20'] = df['收盘'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + 2 * df['STD20']
        df['Lower'] = df['MA20'] - 2 * df['STD20']
        
        # 添加振荡指标
        df['BIAS'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
        
        return df
        
    def setup_grids(self, current_price):
        """设置网格价格"""
        self.reference_price = current_price
        grid_step = (self.grid_range * 2) / (self.grid_levels - 1)
        
        self.grids = []
        for i in range(self.grid_levels):
            grid_price = self.reference_price * (1 - self.grid_range + i * grid_step)
            self.grids.append(grid_price)
            
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取网格交易信号"""
        if self.data is None:
            return 'hold'
            
        current_price = row['收盘']
        current_rsi = row['RSI']
        current_ma20 = row['MA20']
        current_upper = row['Upper']
        current_lower = row['Lower']
        current_bias = row['BIAS']
        
        # 初始化网格
        if not self.grids:
            self.setup_grids(current_price)
            
        # 重新设置网格（如果价格超出了网格范围的一定比例）
        if (current_price > self.grids[-1] * 1.1) or (current_price < self.grids[0] * 0.9):
            self.setup_grids(current_price)
            
        # 网格交易逻辑
        if self.position == 0:  # 没有持仓
            # 买入条件：价格触及网格下沿或下方网格，且RSI低于40
            buy_condition = (
                current_price <= self.grids[0] or
                (current_price <= self.grids[1] and current_rsi < 40) or
                current_price < current_lower  # 价格跌破布林带下轨
            )
            
            if buy_condition:
                return 'buy'
                
        else:  # 有持仓
            # 卖出条件：价格触及网格上沿或上方网格，且RSI高于60
            sell_condition = (
                current_price >= self.grids[-1] or
                (current_price >= self.grids[-2] and current_rsi > 60) or
                current_price > current_upper or  # 价格突破布林带上轨
                current_bias > 10  # 乖离率过高
            )
            
            if sell_condition:
                return 'sell'
                
        return 'hold' 