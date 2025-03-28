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

class TestStrategies(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.stock_code = "510300"
        self.initial_capital = 100000.0
        
    def test_all_strategies_basic(self):
        """测试所有策略的基本功能"""
        start_date = "20230101"
        end_date = "20231231"
        
        strategies = [
            LeftSideStrategy,
            RightSideStrategy,
            GridStrategy,
            BreakoutStrategy,
            DualMAStrategy
        ]
        
        for strategy_class in strategies:
            with self.subTest(strategy=strategy_class.__name__):
                # 创建回测引擎
                engine = BacktestEngine(strategy_class, self.stock_code, self.initial_capital)
                
                # 运行回测
                report = engine.run_backtest(start_date, end_date)
                
                # 验证回测结果
                self.assertIsInstance(report, dict)
                if "error" not in report:
                    self.assertIn('initial_capital', report)
                    self.assertIn('final_capital', report)
                    self.assertIn('total_return', report)
                    self.assertIn('annual_return', report)
                    self.assertIn('max_drawdown', report)
                    self.assertIn('sharpe_ratio', report)
                    self.assertIn('total_trades', report)
                    
                    # 验证数值合理性
                    self.assertGreater(report['final_capital'], 0)
                    self.assertGreaterEqual(report['max_drawdown'], -1)
                    self.assertGreaterEqual(report['total_trades'], 0)
    
    def test_different_time_periods(self):
        """测试不同时间段的策略表现"""
        periods = [
            ("牛市", "20200101", "20201231"),
            ("震荡市", "20210101", "20211231"),
            ("熊市", "20220101", "20221231")
        ]
        
        strategy_class = LeftSideStrategy  # 使用左侧交易策略作为测试
        
        for period_name, start_date, end_date in periods:
            with self.subTest(period=period_name):
                engine = BacktestEngine(strategy_class, self.stock_code, self.initial_capital)
                report = engine.run_backtest(start_date, end_date)
                
                # 验证回测结果
                self.assertIsInstance(report, dict)
                if "error" not in report:
                    self.assertIn('total_return', report)
                    self.assertIn('max_drawdown', report)
    
    def test_different_parameters(self):
        """测试不同参数对策略的影响"""
        start_date = "20230101"
        end_date = "20231231"
        
        stop_loss_values = [0.05, 0.1, 0.15]
        take_profit_values = [0.1, 0.2, 0.3]
        
        strategy_class = GridStrategy  # 使用网格交易策略作为测试
        
        for stop_loss in stop_loss_values:
            for take_profit in take_profit_values:
                with self.subTest(stop_loss=stop_loss, take_profit=take_profit):
                    engine = BacktestEngine(strategy_class, self.stock_code, self.initial_capital)
                    report = engine.run_backtest(
                        start_date, 
                        end_date, 
                        stop_loss=stop_loss, 
                        take_profit=take_profit
                    )
                    
                    # 验证回测结果
                    self.assertIsInstance(report, dict)
                    if "error" not in report:
                        self.assertIn('total_return', report)
    
    def test_transaction_fees(self):
        """测试不同交易费用的影响"""
        start_date = "20230101"
        end_date = "20231231"
        
        fees = [0.0001, 0.0003, 0.0005, 0.001]
        
        strategy_class = DualMAStrategy  # 使用双均线策略作为测试
        
        for fee in fees:
            with self.subTest(transaction_fee=fee):
                engine = BacktestEngine(strategy_class, self.stock_code, self.initial_capital)
                report = engine.run_backtest(
                    start_date, 
                    end_date, 
                    transaction_fee=fee
                )
                
                # 验证回测结果
                self.assertIsInstance(report, dict)
                if "error" not in report:
                    self.assertIn('total_return', report)
    
    def test_initial_capital(self):
        """测试不同初始资金的影响"""
        start_date = "20230101"
        end_date = "20231231"
        
        capitals = [10000, 50000, 100000, 500000]
        
        strategy_class = BreakoutStrategy  # 使用突破策略作为测试
        
        for capital in capitals:
            with self.subTest(initial_capital=capital):
                engine = BacktestEngine(strategy_class, self.stock_code, capital)
                report = engine.run_backtest(start_date, end_date)
                
                # 验证回测结果
                self.assertIsInstance(report, dict)
                if "error" not in report:
                    self.assertEqual(report['initial_capital'], capital)
    
    def test_invalid_inputs(self):
        """测试无效输入的处理"""
        # 测试无效日期
        engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
        report = engine.run_backtest("20250101", "20250331")  # 未来日期
        self.assertIn('error', report)
        
        # 测试结束日期早于开始日期
        report = engine.run_backtest("20230630", "20230101")
        self.assertIn('error', report)
        
        # 测试无效股票代码
        invalid_engine = BacktestEngine(LeftSideStrategy, "999999", self.initial_capital)
        report = invalid_engine.run_backtest("20230101", "20231231")
        self.assertIn('error', report)
    
    def test_strategy_comparison(self):
        """测试策略比较"""
        start_date = "20230101"
        end_date = "20231231"
        
        strategies = [
            LeftSideStrategy,
            RightSideStrategy,
            GridStrategy
        ]
        
        results = {}
        
        for strategy_class in strategies:
            engine = BacktestEngine(strategy_class, self.stock_code, self.initial_capital)
            report = engine.run_backtest(start_date, end_date)
            results[strategy_class.__name__] = report
            
        # 检查结果
        for strategy_name, report in results.items():
            self.assertIsInstance(report, dict)
            if "error" not in report:
                self.assertIn('total_return', report)
                self.assertIn('sharpe_ratio', report)
    
    def test_extreme_market_conditions(self):
        """测试极端市场条件下的策略表现"""
        # 大幅下跌市场 (2022年最大跌幅时期)
        start_date = "20220401"
        end_date = "20220430"
        
        strategies = [
            LeftSideStrategy,
            RightSideStrategy,
            GridStrategy
        ]
        
        for strategy_class in strategies:
            with self.subTest(f"{strategy_class.__name__}_in_crash"):
                engine = BacktestEngine(strategy_class, self.stock_code, self.initial_capital)
                report = engine.run_backtest(start_date, end_date)
                
                # 验证回测结果
                self.assertIsInstance(report, dict)
                if "error" not in report:
                    self.assertIn('max_drawdown', report)
        
        # 快速上涨市场 (2020年7月)
        start_date = "20200701"
        end_date = "20200731"
        
        for strategy_class in strategies:
            with self.subTest(f"{strategy_class.__name__}_in_rally"):
                engine = BacktestEngine(strategy_class, self.stock_code, self.initial_capital)
                report = engine.run_backtest(start_date, end_date)
                
                # 验证回测结果
                self.assertIsInstance(report, dict)
                if "error" not in report:
                    self.assertIn('total_return', report)

if __name__ == '__main__':
    unittest.main() 