import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy import LeftSideStrategy
from backtest import BacktestEngine
from typing import Dict, List, Tuple

class ParameterOptimizer:
    """参数优化器，用于寻找策略的最优参数"""
    
    def __init__(self, strategy_class, stock_code: str, initial_capital: float = 100000.0):
        self.strategy_class = strategy_class
        self.stock_code = stock_code
        self.initial_capital = initial_capital
        
    def grid_search(self, 
                   parameter_grid: Dict[str, List], 
                   start_date: str, 
                   end_date: str,
                   transaction_fee: float = 0.0003,
                   metric: str = 'total_return') -> Tuple[Dict, List[Dict]]:
        """
        网格搜索找到最优参数
        :param parameter_grid: 参数网格，如{'stop_loss': [0.05, 0.1], 'take_profit': [0.1, 0.2]}
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param transaction_fee: 交易费用
        :param metric: 优化指标，如'total_return', 'sharpe_ratio'等
        :return: (最优参数, 所有参数组合的结果)
        """
        # 生成所有参数组合
        param_combinations = self._generate_combinations(parameter_grid)
        
        # 存储所有结果
        all_results = []
        
        # 执行网格搜索
        for params in param_combinations:
            engine = BacktestEngine(self.strategy_class, self.stock_code, self.initial_capital)
            
            # 运行回测
            report = engine.run_backtest(
                start_date=start_date,
                end_date=end_date,
                transaction_fee=transaction_fee,
                **params  # 传入参数
            )
            
            # 存储结果
            result = {
                'params': params,
                'report': report
            }
            all_results.append(result)
        
        # 找到最优参数
        best_params = None
        best_value = -float('inf')
        
        for result in all_results:
            report = result['report']
            if "error" not in report and metric in report:
                value = report[metric]
                
                # 如果是最大回撤，我们希望它的绝对值越小越好
                if metric == 'max_drawdown':
                    value = -abs(value)
                    
                if value > best_value:
                    best_value = value
                    best_params = result['params']
        
        return best_params, all_results
        
    def walk_forward_analysis(self, 
                             parameter_grid: Dict[str, List], 
                             start_date: str, 
                             end_date: str,
                             train_window: int = 6,  # 6个月
                             test_window: int = 1,   # 1个月
                             metric: str = 'total_return') -> List[Dict]:
        """
        执行Walk-forward分析，使用滚动窗口进行参数优化
        :param parameter_grid: 参数网格
        :param start_date: 整体回测开始日期
        :param end_date: 整体回测结束日期
        :param train_window: 训练窗口长度（月）
        :param test_window: 测试窗口长度（月）
        :param metric: 优化指标
        :return: 每个测试窗口的结果列表
        """
        # 生成训练和测试窗口
        windows = self._generate_train_test_windows(
            start_date, end_date, train_window, test_window
        )
        
        # 存储每个窗口的结果
        window_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"Window {i+1}: Training {train_start}-{train_end}, Testing {test_start}-{test_end}")
            
            # 在训练集上找到最优参数
            best_params, _ = self.grid_search(
                parameter_grid, train_start, train_end, metric=metric
            )
            
            if best_params is None:
                continue
                
            # 在测试集上评估最优参数
            engine = BacktestEngine(self.strategy_class, self.stock_code, self.initial_capital)
            test_report = engine.run_backtest(
                start_date=test_start,
                end_date=test_end,
                **best_params
            )
            
            # 存储结果
            window_result = {
                'window': i + 1,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'best_params': best_params,
                'train_metric': None,  # 可以再次评估训练集结果
                'test_result': test_report
            }
            window_results.append(window_result)
        
        return window_results
        
    def _generate_combinations(self, parameter_grid: Dict[str, List]) -> List[Dict]:
        """生成所有参数组合"""
        from itertools import product
        
        # 获取所有参数名和参数值列表
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        # 生成所有组合
        combinations = []
        for values in product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
            
        return combinations
        
    def _generate_train_test_windows(self, 
                                    start_date: str, 
                                    end_date: str, 
                                    train_window: int, 
                                    test_window: int) -> List[Tuple[str, str, str, str]]:
        """生成训练和测试窗口"""
        # 转换日期格式
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        windows = []
        current = start
        
        while current + pd.DateOffset(months=train_window + test_window) <= end:
            # 计算训练窗口
            train_start = current
            train_end = train_start + pd.DateOffset(months=train_window) - pd.DateOffset(days=1)
            
            # 计算测试窗口
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=test_window) - pd.DateOffset(days=1)
            
            # 添加窗口
            windows.append((
                train_start.strftime('%Y%m%d'),
                train_end.strftime('%Y%m%d'),
                test_start.strftime('%Y%m%d'),
                test_end.strftime('%Y%m%d')
            ))
            
            # 向前移动
            current = test_start + pd.DateOffset(days=1)
            
        return windows

class TestParameterOptimization(unittest.TestCase):
    """测试参数优化"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.stock_code = "510300"
        self.initial_capital = 100000.0
        self.optimizer = ParameterOptimizer(LeftSideStrategy, self.stock_code, self.initial_capital)
        
    def test_grid_search(self):
        """测试网格搜索参数优化"""
        # 定义参数网格
        parameter_grid = {
            'stop_loss': [0.05, 0.1, 0.15],
            'take_profit': [0.1, 0.15, 0.2, 0.25]
        }
        
        # 执行网格搜索
        best_params, all_results = self.optimizer.grid_search(
            parameter_grid,
            start_date="20230101",
            end_date="20230630",
            metric='total_return'
        )
        
        # 验证结果
        self.assertIsNotNone(best_params)
        self.assertIn('stop_loss', best_params)
        self.assertIn('take_profit', best_params)
        self.assertEqual(len(all_results), 12)  # 3个stop_loss * 4个take_profit = 12种组合
        
    def test_walk_forward_analysis(self):
        """测试Walk-forward分析"""
        # 定义参数网格
        parameter_grid = {
            'stop_loss': [0.05, 0.1],
            'take_profit': [0.15, 0.2]
        }
        
        # 执行Walk-forward分析
        window_results = self.optimizer.walk_forward_analysis(
            parameter_grid,
            start_date="20200101",
            end_date="20210630",
            train_window=3,  # 3个月
            test_window=1,   # 1个月
            metric='sharpe_ratio'
        )
        
        # 验证结果
        self.assertGreater(len(window_results), 0)
        for result in window_results:
            self.assertIn('window', result)
            self.assertIn('train_period', result)
            self.assertIn('test_period', result)
            self.assertIn('best_params', result)
            self.assertIn('test_result', result)
            
    def test_parameter_sensitivity(self):
        """测试参数敏感性"""
        # 测试stop_loss参数的敏感性
        stop_loss_values = [0.05, 0.08, 0.1, 0.15, 0.2]
        stop_loss_results = []
        
        for stop_loss in stop_loss_values:
            engine = BacktestEngine(LeftSideStrategy, self.stock_code, self.initial_capital)
            report = engine.run_backtest(
                start_date="20230101",
                end_date="20230630",
                stop_loss=stop_loss,
                take_profit=0.2  # 固定参数
            )
            
            if "error" not in report:
                stop_loss_results.append({
                    'stop_loss': stop_loss,
                    'total_return': report['total_return'],
                    'max_drawdown': report['max_drawdown']
                })
        
        # 验证结果
        self.assertGreater(len(stop_loss_results), 0)
        
        # 检查参数随止损变化的单调性
        # 这只是一个功能性测试，实际结果可能不满足单调性
        if len(stop_loss_results) >= 2:
            # 验证我们至少得到了两个不同的结果
            total_returns = [result['total_return'] for result in stop_loss_results]
            self.assertGreater(len(set(total_returns)), 1)

if __name__ == '__main__':
    unittest.main() 