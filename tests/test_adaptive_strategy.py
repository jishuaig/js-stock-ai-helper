import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy import (
    LeftSideStrategy, 
    RightSideStrategy, 
    GridStrategy, 
    BreakoutStrategy, 
    DualMAStrategy
)
from backtest import BacktestEngine

class MarketStateClassifier:
    """市场状态分类器，用于自适应市场策略的测试"""
    
    def __init__(self):
        self.lookback_period = 60  # 回看60个交易日判断市场状态
        
    def classify_market(self, df: pd.DataFrame) -> str:
        """
        根据历史数据判断市场状态
        :param df: 包含价格数据的DataFrame
        :return: 市场状态('bull', 'bear', 或 'sideways')
        """
        if len(df) < self.lookback_period:
            return 'unknown'
            
        # 计算最近60个交易日的涨跌幅
        recent_df = df.iloc[-self.lookback_period:]
        start_price = recent_df['收盘'].iloc[0]
        end_price = recent_df['收盘'].iloc[-1]
        change_pct = (end_price - start_price) / start_price
        
        # 计算波动率
        volatility = recent_df['收盘'].pct_change().std() * np.sqrt(252)
        
        # 计算趋势强度 (用收盘价对时间的线性回归斜率)
        x = np.arange(len(recent_df))
        y = recent_df['收盘'].values
        slope, _ = np.polyfit(x, y, 1)
        trend_strength = slope / start_price * 100  # 归一化为百分比
        
        # 根据涨跌幅和趋势强度判断市场状态
        if change_pct > 0.05 and trend_strength > 0.03:
            return 'bull'  # 牛市
        elif change_pct < -0.05 and trend_strength < -0.03:
            return 'bear'  # 熊市
        else:
            return 'sideways'  # 震荡市

class AdaptiveStrategy:
    """
    自适应市场策略，根据市场状态切换不同的交易策略
    这个类仅用于测试，不是实际的交易策略实现
    """
    
    def __init__(self, stock_code: str, initial_capital: float = 100000.0):
        self.stock_code = stock_code
        self.initial_capital = initial_capital
        self.market_classifier = MarketStateClassifier()
        
        # 创建各种策略实例
        self.bull_strategy = DualMAStrategy(stock_code, initial_capital)  # 牛市策略
        self.bear_strategy = LeftSideStrategy(stock_code, initial_capital)  # 熊市策略
        self.sideways_strategy = GridStrategy(stock_code, initial_capital)  # 震荡市策略
        
        # 当前使用的策略
        self.current_strategy = None
        self.market_state = 'unknown'
        
    def backtest(self, start_date: str, end_date: str, 
                stop_loss: float = 0.1, 
                take_profit: float = 0.2,
                transaction_fee: float = 0.0003):
        """
        运行自适应策略回测
        """
        # 获取数据
        df = self.bull_strategy.fetch_data(start_date, end_date)
        if df is None:
            return {"error": "获取数据失败"}
            
        # 每个月重新评估市场状态并切换策略
        results = {'market_states': []}
        current_date = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        monthly_periods = []
        while current_date < end:
            next_date = current_date + pd.DateOffset(months=1)
            if next_date > end:
                next_date = end
                
            period_start = current_date.strftime('%Y%m%d')
            period_end = next_date.strftime('%Y%m%d')
            monthly_periods.append((period_start, period_end))
            
            current_date = next_date
            
        # 回测各个时间段
        for i, (period_start, period_end) in enumerate(monthly_periods):
            # 获取当前时间段的数据
            period_df = df[(df.index >= period_start) & (df.index <= period_end)]
            
            # 判断市场状态
            market_state = self.market_classifier.classify_market(df[df.index <= period_start])
            results['market_states'].append({
                'period': f"{period_start}-{period_end}",
                'state': market_state
            })
            
            # 根据市场状态选择策略
            if market_state == 'bull':
                engine = BacktestEngine(DualMAStrategy, self.stock_code, self.initial_capital)
            elif market_state == 'bear':
                engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
            elif market_state == 'sideways':
                engine = BacktestEngine(GridStrategy, self.stock_code, self.initial_capital)
            else:
                # 默认使用左侧策略
                engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
                
            # 运行该时间段的回测
            report = engine.run_backtest(period_start, period_end, stop_loss, take_profit, transaction_fee)
            
            results[f"period_{i}"] = {
                'period': f"{period_start}-{period_end}",
                'market_state': market_state,
                'strategy': engine.strategy.__class__.__name__,
                'report': report
            }
            
        return results

class TestAdaptiveStrategy(unittest.TestCase):
    """测试自适应市场策略"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.stock_code = "510300"
        self.initial_capital = 100000.0
        
    def test_market_classifier(self):
        """测试市场状态分类器"""
        # 创建分类器
        classifier = MarketStateClassifier()
        
        # 创建LeftSideStrategy实例来获取数据
        strategy = LeftSideStrategy(self.stock_code, self.initial_capital)
        
        # 测试牛市时期 (2020年下半年)
        bull_df = strategy.fetch_data("20200601", "20201231")
        if bull_df is not None:
            market_state = classifier.classify_market(bull_df)
            self.assertEqual(market_state, 'bull')
        
        # 测试熊市时期 (2022年上半年)
        bear_df = strategy.fetch_data("20220101", "20220630")
        if bear_df is not None:
            market_state = classifier.classify_market(bear_df)
            self.assertEqual(market_state, 'bull')
        
        # 测试震荡市时期 (2021年)
        sideways_df = strategy.fetch_data("20210101", "20211231")
        if sideways_df is not None:
            market_state = classifier.classify_market(sideways_df)
            self.assertEqual(market_state, 'sideways')
    
    def test_adaptive_strategy_switching(self):
        """测试自适应策略的切换"""
        # 创建自适应策略实例
        adaptive_strategy = AdaptiveStrategy(self.stock_code, self.initial_capital)
        
        # 运行一段较长时间的回测，确保会有策略切换
        results = adaptive_strategy.backtest("20200101", "20220630")
        
        # 验证结果
        self.assertIn('market_states', results)
        self.assertGreater(len(results['market_states']), 0)
        
        # 验证是否有不同的市场状态和策略
        market_states = set()
        strategies = set()
        
        for i in range(len(results['market_states'])):
            key = f"period_{i}"
            if key in results:
                market_states.add(results[key]['market_state'])
                strategies.add(results[key]['strategy'])
        
        # 验证是否存在多种市场状态和策略
        self.assertGreater(len(market_states), 1)
        self.assertGreater(len(strategies), 1)
    
    def test_adaptive_vs_single_strategies(self):
        """测试自适应策略与单一策略的对比"""
        # 测试时间段
        start_date = "20200101"
        end_date = "20230630"  # 包含牛市、熊市和震荡市
        
        # 创建自适应策略
        adaptive_strategy = AdaptiveStrategy(self.stock_code, self.initial_capital)
        adaptive_results = adaptive_strategy.backtest(start_date, end_date)
        
        # 创建单一策略的回测引擎
        engines = {
            'LeftSideStrategy': BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital),
            'RightSideStrategy': BacktestEngine(RightSideStrategy, self.stock_code, self.initial_capital),
            'GridStrategy': BacktestEngine(GridStrategy, self.stock_code, self.initial_capital)
        }
        
        # 运行单一策略回测
        single_results = {}
        for name, engine in engines.items():
            report = engine.run_backtest(start_date, end_date)
            single_results[name] = report
        
        # 验证结果
        # 注意：这个测试主要是验证功能，不是验证性能
        for name, report in single_results.items():
            self.assertIsInstance(report, dict)
            if "error" not in report:
                self.assertIn('total_return', report)
                self.assertIn('max_drawdown', report)

if __name__ == '__main__':
    unittest.main() 