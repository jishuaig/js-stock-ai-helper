import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy import LeftSideStrategy
from backtest import BacktestEngine

class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.stock_code = "510300"
        self.initial_capital = 100000.0
        self.start_date = "20230101"
        self.end_date = "20240328"
        
    def test_left_side_strategy_backtest(self):
        """测试左侧交易策略回测"""
        # 创建回测引擎
        engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
        
        # 运行回测
        report = engine.run_backtest(self.start_date, self.end_date)
        
        # 验证回测结果
        self.assertIsInstance(report, dict)
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
        
    def test_invalid_dates(self):
        """测试无效日期"""
        engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
        
        # 测试结束日期早于开始日期
        report = engine.run_backtest(self.end_date, self.start_date)
        self.assertIn('error', report)
        
        # 测试未来日期
        future_date = (datetime.now() + timedelta(days=365)).strftime('%Y%m%d')
        report = engine.run_backtest(self.start_date, future_date)
        self.assertIn('error', report)
        
    def test_different_capital(self):
        """测试不同初始资金"""
        test_capitals = [50000, 100000, 200000]
        
        for capital in test_capitals:
            engine = BacktestEngine(LeftSideStrategy, self.stock_code, capital)
            report = engine.run_backtest(self.start_date, self.end_date)
            
            # 验证初始资金正确设置
            self.assertEqual(report['initial_capital'], capital)
            
            # 验证最终资金大于初始资金
            self.assertGreater(report['final_capital'], capital)
            
    def test_different_stop_loss_take_profit(self):
        """测试不同的止损止盈设置"""
        test_params = [
            (0.05, 0.15),  # 保守设置
            (0.1, 0.2),    # 默认设置
            (0.15, 0.3)    # 激进设置
        ]
        
        for stop_loss, take_profit in test_params:
            engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
            report = engine.run_backtest(
                self.start_date, 
                self.end_date,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # 验证回测结果合理
            self.assertGreater(report['final_capital'], 0)
            self.assertGreaterEqual(report['max_drawdown'], -1)
            
    def test_transaction_fee(self):
        """测试交易费用影响"""
        test_fees = [0.0001, 0.0003, 0.0005]
        
        for fee in test_fees:
            engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
            report = engine.run_backtest(
                self.start_date,
                self.end_date,
                transaction_fee=fee
            )
            
            # 验证交易费用影响
            self.assertGreater(report['final_capital'], 0)
            
    def test_plot_results(self):
        """测试结果可视化"""
        engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
        report = engine.run_backtest(self.start_date, self.end_date)
        
        # 验证绘图功能不抛出异常
        try:
            engine.plot_results()
        except Exception as e:
            self.fail(f"plot_results raised {type(e)} unexpectedly!")

if __name__ == '__main__':
    unittest.main() 