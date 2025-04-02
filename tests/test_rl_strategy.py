import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
from datetime import datetime
import time
import tensorflow as tf
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在导入策略模块
from strategy.rl_strategy import DeepRLStrategy

# GPU诊断信息
print("TensorFlow版本:", tf.__version__)
print("检测到的GPU设备:")
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("  -", gpu.name, ":", gpu.device_type)

# 如果没有检测到GPU，打印更多信息
if not gpus:
    print("警告: 未检测到GPU！")
    print("检查环境变量:")
    print("  CUDA_VISIBLE_DEVICES =", os.environ.get('CUDA_VISIBLE_DEVICES', '未设置'))
    print("\n请确保:")
    print("1. 已安装GPU驱动")
    print("2. 已安装CUDA和cuDNN")
    print("3. 安装了GPU版本的TensorFlow (tensorflow-gpu)")
    print("4. 运行 'nvidia-smi' 确认GPU可用")

# 强制使用GPU配置
try:
    # 清除之前的会话和设备设置
    tf.keras.backend.clear_session()
    
    # 显式设置GPU可见性
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一块GPU
    
    # 允许GPU内存动态增长
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # 优化GPU性能设置
    tf.config.optimizer.set_jit(True)  # 启用XLA JIT编译
    
    # 使用混合精度
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"设置计算精度为: {policy.name}")
    
    # 验证GPU是否可用
    with tf.device('/GPU:0'):
        random_tensor = tf.random.normal([1000, 1000])
        result = tf.matmul(random_tensor, random_tensor)
        # 尝试获取结果，若GPU不可用会抛出异常
        result.numpy()
        print("GPU测试成功 - GPU已正确配置并可使用")
        
    # 设置线程
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    
except Exception as e:
    print(f"GPU配置错误: {str(e)}")
    print("将回退到CPU")

def test_rl_strategy():
    """测试深度强化学习交易策略"""
    # 使用akshare获取股票数据
    stock_code = "510300"  # 沪深300ETF
    
    # 获取较短时间段的训练数据
    train_start_date = "20230701"  # 只使用2023下半年数据
    train_end_date = "20231231"
    train_data = ak.fund_etf_hist_em(symbol=stock_code, period="daily", 
                                    start_date=train_start_date, 
                                    end_date=train_end_date, 
                                    adjust="qfq")
    
    # 获取更短的测试数据集
    test_start_date = "20240101"
    test_end_date = "20240331"  # 只使用2024年Q1数据
    test_data = ak.fund_etf_hist_em(symbol=stock_code, period="daily", 
                                  start_date=test_start_date, 
                                  end_date=test_end_date, 
                                  adjust="qfq")
    
    # 打印列名以便调试
    print("数据列名:", train_data.columns.tolist())
    
    # 重命名列以匹配我们的策略需要的列名 
    column_mapping = {
        "日期": "日期",
        "开盘": "开盘", 
        "收盘": "收盘",
        "最高": "最高",
        "最低": "最低",
        "成交量": "成交量"
    }
    
    # 检查实际的列名，并进行适当的映射
    if "日期" not in train_data.columns and "date" in train_data.columns:
        column_mapping = {
            "date": "日期",
            "open": "开盘", 
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量"
        }
        
    train_data = train_data.rename(columns=column_mapping)
    test_data = test_data.rename(columns=column_mapping)
    
    # 设置日期索引
    train_data.set_index("日期", inplace=True)
    test_data.set_index("日期", inplace=True)
    
    # 确保日期是datetime类型
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)
    
    # 数据降采样（可选）- 使用每周数据而非每日数据
    # train_data = train_data.resample('W').agg({
    #     '开盘': 'first', '最高': 'max', '最低': 'min', '收盘': 'last', '成交量': 'sum'
    # })
    # test_data = test_data.resample('W').agg({
    #     '开盘': 'first', '最高': 'max', '最低': 'min', '收盘': 'last', '成交量': 'sum'
    # })
    
    print(f"训练数据: {len(train_data)} 条记录，从 {train_data.index.min()} 到 {train_data.index.max()}")
    print(f"测试数据: {len(test_data)} 条记录，从 {test_data.index.min()} 到 {test_data.index.max()}")
    
    # 创建RL策略（使用优化后的参数）
    rl_strategy = DeepRLStrategy(
        stock_code=stock_code, 
        initial_capital=100000.0,
        lookback_window_size=10,  # 减小回溯窗口大小
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9,        # 加快探索率的衰减
        gamma=0.95,
        learning_rate=0.001,
        batch_size=64,            # 增大批量大小提高训练效率
        memory_size=2000          # 减小记忆容量
    )
    
    # 训练并回测
    print("开始训练和回测...")
    start_time = time.time()
    
    results = rl_strategy.train_and_backtest(
        train_df=train_data,
        test_df=test_data,
        episodes=10,             # 减少训练回合数
        batch_size=64            # 增大批量大小
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 打印回测结果
    print("\n=== 回测结果 ===")
    print(f"总回报: {results['total_return']*100:.2f}%")
    print(f"年化回报: {results['annual_return']*100:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
    print(f"胜率: {results['win_rate']*100:.2f}%")
    print(f"交易次数: {results['trade_count']}")
    
    # 绘制权益曲线
    rl_strategy.plot_results()
    plt.show()

if __name__ == "__main__":
    test_rl_strategy() 