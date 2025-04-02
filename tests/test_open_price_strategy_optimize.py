import pytest
import pandas as pd
import numpy as np
from strategy.open_price_strategy import OpenPriceStrategy
import matplotlib.pyplot as plt
import seaborn as sns

def test_optimize_parameters():
    # 设置参数网格（聚焦在最优区域）
    high_thresholds = np.unique(np.concatenate([
        np.arange(0.01, 0.02, 0.001),    # 1.0% 到 2.0% 步长0.1%
        np.arange(0.02, 0.026, 0.002)    # 2.0% 到 2.5% 步长0.2%
    ]))
    
    low_thresholds = np.unique(np.concatenate([
        np.arange(0.02, 0.04, 0.002),    # 2.0% 到 4.0% 步长0.2%
        np.arange(0.04, 0.051, 0.005)    # 4.0% 到 5.0% 步长0.5%
    ]))
    
    print(f"参数组合总数: {len(high_thresholds) * len(low_thresholds)}")
    print(f"高开阈值列表: {[f'{x*100:.1f}%' for x in high_thresholds]}")
    print(f"低开阈值列表: {[f'{x*100:.1f}%' for x in low_thresholds]}")
    
    # 存储结果
    results = []
    
    # 获取一次数据，避免重复获取
    base_strategy = OpenPriceStrategy(stock_code="159985", initial_capital=100000.0)
    df = base_strategy.fetch_data("20230101", "20231230")
    if df is None:
        return
    
    total_combinations = len(high_thresholds) * len(low_thresholds)
    current_combination = 0
    
    # 遍历参数组合
    for high in high_thresholds:
        for low in low_thresholds:
            current_combination += 1
            print(f"\r进度: {current_combination}/{total_combinations} "
                  f"({current_combination/total_combinations*100:.1f}%)", end="")
            
            # 初始化策略
            strategy = OpenPriceStrategy(
                stock_code="159985",
                initial_capital=100000.0,
                high_threshold=high,
                low_threshold=low
            )
            
            # 执行回测
            result = strategy.backtest(df.copy())
            
            # 计算额外的指标
            daily_returns = pd.Series([d['capital'] for d in result['daily_capital']]).pct_change()
            win_rate = sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
            
            # 计算年化收益率
            days = (df.index[-1] - df.index[0]).days
            annual_return = (1 + result['returns']) ** (365/days) - 1 if days > 0 else 0
            
            # 计算收益风险比
            risk_return_ratio = abs(result['returns'] / result['metrics']['max_drawdown']) if result['metrics']['max_drawdown'] != 0 else 0
            
            # 记录结果
            results.append({
                'high_threshold': high,
                'low_threshold': low,
                'return': result['returns'],
                'annual_return': annual_return,
                'sharpe': result['metrics']['sharpe_ratio'],
                'max_drawdown': result['metrics']['max_drawdown'],
                'total_trades': result['metrics']['total_trades'],
                'win_rate': win_rate,
                'risk_return_ratio': risk_return_ratio
            })
    
    print("\n")  # 换行
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 创建子图
    plt.figure(figsize=(20, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统可用的中文字体
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置基本样式
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 创建两个子图
    ax1 = plt.subplot(121)  # 左边的图：固定高开阈值，展示不同低开阈值的收益率变化
    ax2 = plt.subplot(122)  # 右边的图：固定低开阈值，展示不同高开阈值的收益率变化
    
    # 找出收益率最高的5个参数组合
    top5_combinations = results_df.nlargest(5, 'return')
    
    # 颜色列表（使用醒目的颜色）
    colors = ['#FF4B4B', '#4CAF50', '#2196F3', '#9C27B0', '#FF9800']
    
    # 绘制左图：固定高开阈值
    for high_threshold in high_thresholds:
        data = results_df[results_df['high_threshold'] == high_threshold]
        line = ax1.plot(data['low_threshold'] * 100, data['return'] * 100, 
                       alpha=0.2, color='gray', linestyle='--', linewidth=1)
        
    # 突出显示最佳组合
    for idx, (_, row) in enumerate(top5_combinations.iterrows()):
        high = row['high_threshold']
        data = results_df[results_df['high_threshold'] == high]
        line = ax1.plot(data['low_threshold'] * 100, data['return'] * 100,
                       label=f'高开{high*100:.1f}% (收益率{row["return"]*100:.1f}%)', 
                       color=colors[idx], 
                       linewidth=2.5)
        
    ax1.set_xlabel('低开阈值 (%)', fontsize=12)
    ax1.set_ylabel('收益率 (%)', fontsize=12)
    ax1.set_title('固定高开阈值下的收益率变化', fontsize=14, pad=20)
    ax1.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=10)
    
    # 绘制右图：固定低开阈值
    for low_threshold in low_thresholds:
        data = results_df[results_df['low_threshold'] == low_threshold]
        line = ax2.plot(data['high_threshold'] * 100, data['return'] * 100,
                       alpha=0.2, color='gray', linestyle='--', linewidth=1)
        
    # 突出显示最佳组合
    for idx, (_, row) in enumerate(top5_combinations.iterrows()):
        low = row['low_threshold']
        data = results_df[results_df['low_threshold'] == low]
        line = ax2.plot(data['high_threshold'] * 100, data['return'] * 100,
                       label=f'低开{low*100:.1f}% (收益率{row["return"]*100:.1f}%)',
                       color=colors[idx],
                       linewidth=2.5)
        
    ax2.set_xlabel('高开阈值 (%)', fontsize=12)
    ax2.set_ylabel('收益率 (%)', fontsize=12)
    ax2.set_title('固定低开阈值下的收益率变化', fontsize=14, pad=20)
    ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=10)
    
    plt.suptitle('参数优化结果 - 收益率变化', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('parameter_optimization_lines.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印最优参数组合
    print("\n收益率最高的前5个参数组合：")
    for idx, (_, params) in enumerate(top5_combinations.iterrows(), 1):
        print(f"\n第{idx}优参数组合：")
        print(f"高开阈值: {params['high_threshold']:.3f} ({params['high_threshold']*100:.1f}%)")
        print(f"低开阈值: {params['low_threshold']:.3f} ({params['low_threshold']*100:.1f}%)")
        print(f"总收益率: {params['return']:.2%}")
        print(f"年化收益率: {params['annual_return']:.2%}")
        print(f"夏普比率: {params['sharpe']:.2f}")
        print(f"最大回撤: {params['max_drawdown']:.2%}")
        print(f"胜率: {params['win_rate']:.2%}")
        print(f"收益风险比: {params['risk_return_ratio']:.2f}")
        print(f"总交易次数: {params['total_trades']}")
    
    # 使用收益率最优的参数运行一次回测，展示详细结果
    best_params = top5_combinations.iloc[0]
    best_strategy = OpenPriceStrategy(
        stock_code="159985",
        initial_capital=100000.0,
        high_threshold=best_params['high_threshold'],
        low_threshold=best_params['low_threshold']
    )
    
    result = best_strategy.backtest(df.copy())
    
    print("\n收益率最优参数的交易记录：")
    for trade in result['transactions']:
        print(f"日期: {trade['date'].strftime('%Y-%m-%d')}, "
              f"类型: {trade['type']}, "
              f"价格: {trade['price']:.2f}, "
              f"数量: {trade['shares']}") 