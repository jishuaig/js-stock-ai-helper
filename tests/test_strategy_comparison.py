import unittest
from datetime import datetime, timedelta
from strategy import LeftSideStrategy, ComprehensiveStrategy
from backtest import compare_strategies

class TestStrategyComparison(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.stock_code = "510300"
        self.initial_capital = 100000.0
        self.start_date = "20230101"
        self.end_date = "20240328"
        
    def test_strategy_comparison(self):
        """测试策略对比功能"""
        # 运行策略对比
        report1, report2 = compare_strategies(
            LeftSideStrategy,
            ComprehensiveStrategy,
            self.stock_code,
            self.start_date,
            self.end_date,
            self.initial_capital
        )
        
        # 验证两个策略的回测结果
        self.assertIsInstance(report1, dict)
        self.assertIsInstance(report2, dict)
        
        # 验证回测结果包含所有必要的指标
        required_metrics = [
            'initial_capital',
            'final_capital',
            'total_return',
            'annual_return',
            'max_drawdown',
            'sharpe_ratio',
            'total_trades'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, report1)
            self.assertIn(metric, report2)
            
        # 验证数值合理性
        self.assertGreater(report1['final_capital'], 0)
        self.assertGreater(report2['final_capital'], 0)
        self.assertGreaterEqual(report1['max_drawdown'], -1)
        self.assertGreaterEqual(report2['max_drawdown'], -1)
        self.assertGreaterEqual(report1['total_trades'], 0)
        self.assertGreaterEqual(report2['total_trades'], 0)
        
    def test_strategy_comparison_with_different_params(self):
        """测试不同参数下的策略对比"""
        test_params = [
            (0.05, 0.15),  # 保守设置
            (0.1, 0.2),    # 默认设置
            (0.15, 0.3)    # 激进设置
        ]
        
        for stop_loss, take_profit in test_params:
            report1, report2 = compare_strategies(
                LeftSideStrategy,
                ComprehensiveStrategy,
                self.stock_code,
                self.start_date,
                self.end_date,
                self.initial_capital,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # 验证回测结果合理
            self.assertGreater(report1['final_capital'], 0)
            self.assertGreater(report2['final_capital'], 0)
            self.assertGreaterEqual(report1['max_drawdown'], -1)
            self.assertGreaterEqual(report2['max_drawdown'], -1)
            
    def test_strategy_comparison_with_different_capital(self):
        """测试不同初始资金下的策略对比"""
        test_capitals = [50000, 100000, 200000]
        
        for capital in test_capitals:
            report1, report2 = compare_strategies(
                LeftSideStrategy,
                ComprehensiveStrategy,
                self.stock_code,
                self.start_date,
                self.end_date,
                initial_capital=capital
            )
            
            # 验证初始资金正确设置
            self.assertEqual(report1['initial_capital'], capital)
            self.assertEqual(report2['initial_capital'], capital)
            
            # 验证最终资金大于初始资金
            self.assertGreater(report1['final_capital'], capital)
            self.assertGreater(report2['final_capital'], capital)
            
    def test_strategy_comparison_with_different_periods(self):
        """测试不同时间段的策略对比"""
        test_periods = [
            ("20230101", "20230630"),  # 上半年
            ("20230701", "20231231"),  # 下半年
            ("20230101", "20231231")   # 全年
        ]
        
        for start, end in test_periods:
            report1, report2 = compare_strategies(
                LeftSideStrategy,
                ComprehensiveStrategy,
                self.stock_code,
                start,
                end,
                self.initial_capital
            )
            
            # 验证回测结果合理
            self.assertGreater(report1['final_capital'], 0)
            self.assertGreater(report2['final_capital'], 0)
            self.assertGreaterEqual(report1['max_drawdown'], -1)
            self.assertGreaterEqual(report2['max_drawdown'], -1)

if __name__ == '__main__':
    unittest.main() 