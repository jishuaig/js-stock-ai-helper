import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from strategy.base_strategy import BaseStrategy

class DeepRLStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 100000.0, 
                 lookback_window_size: int = 60,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 memory_size: int = 10000):
        """
        初始化深度强化学习策略
        :param stock_code: 股票代码
        :param initial_capital: 初始资金
        :param lookback_window_size: 回溯窗口大小，用于观察历史数据
        :param gamma: 折扣因子
        :param epsilon: 探索率
        :param epsilon_min: 最小探索率
        :param epsilon_decay: 探索率衰减
        :param learning_rate: 学习率
        :param batch_size: 批量大小
        :param memory_size: 记忆容量
        """
        super().__init__(stock_code, initial_capital)
        self.lookback_window_size = lookback_window_size
        
        # RL参数
        self.state_size = lookback_window_size * 5  # 5个特征：开盘、收盘、最高、最低、成交量
        self.action_size = 3  # 买入、卖出、持有
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma    # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # 创建模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # 交易状态
        self.inventory = 0  # 持仓数量
        self.current_step = 0  # 当前步骤
        
    def _build_model(self):
        """构建深度神经网络模型"""
        model = Sequential()
        # 使用更简化的网络结构
        model.add(LSTM(32, input_shape=(self.lookback_window_size, 5), return_sequences=False))
        model.add(Dropout(0.1))  # 减少dropout比例
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        # 使用更高效的优化器
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """更新目标模型"""
        self.target_model.set_weights(self.model.get_weights())
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 标准化数据
        df["norm_open"] = (df["开盘"] - df["开盘"].mean()) / df["开盘"].std()
        df["norm_close"] = (df["收盘"] - df["收盘"].mean()) / df["收盘"].std()
        df["norm_high"] = (df["最高"] - df["最高"].mean()) / df["最高"].std()
        df["norm_low"] = (df["最低"] - df["最低"].mean()) / df["最低"].std()
        df["norm_volume"] = (df["成交量"] - df["成交量"].mean()) / df["成交量"].std()
        
        # 计算额外技术指标（可以根据需要添加更多）
        df["MA5"] = df["收盘"].rolling(window=5).mean()
        df["MA10"] = df["收盘"].rolling(window=10).mean()
        df["MA20"] = df["收盘"].rolling(window=20).mean()
        
        # 计算MACD
        exp1 = df["收盘"].ewm(span=12, adjust=False).mean()
        exp2 = df["收盘"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # 计算RSI
        delta = df["收盘"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        return df
    
    def get_state(self, data, t):
        """
        获取当前状态
        :param data: 数据
        :param t: 当前时间步
        :return: 状态向量
        """
        if t - self.lookback_window_size + 1 < 0:
            # 如果没有足够的历史数据，填充前面的时间步
            padding = -1 * (t - self.lookback_window_size + 1)
            state = np.concatenate((
                np.zeros((padding, 5)),
                data[0:t+1]
            ), axis=0)
        else:
            state = data[t-self.lookback_window_size+1:t+1]
            
        return state
    
    def memorize(self, state, action, reward, next_state, done):
        """将经验存储到记忆中"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """根据当前状态选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 重塑状态为模型输入形状
        state = np.reshape(state, [1, self.lookback_window_size, 5])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """从记忆中随机抽样进行经验回放"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.lookback_window_size, 5])
            next_state = np.reshape(next_state, [1, self.lookback_window_size, 5])
            
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)
                
            self.model.fit(state, target, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_trading_signal(self, row: pd.Series) -> str:
        """获取交易信号"""
        # 在回测阶段不使用RL模型，回测前需要先训练模型
        return "hold"
    
    def train(self, train_data, episodes=100, batch_size=32):
        """
        训练RL模型
        :param train_data: 训练数据，包含规范化后的特征
        :param episodes: 训练回合数
        :param batch_size: 批量大小
        """
        data = np.array([
            train_data["norm_open"].values,
            train_data["norm_close"].values,
            train_data["norm_high"].values,
            train_data["norm_low"].values,
            train_data["norm_volume"].values
        ]).T
        
        # 计算价格趋势，用于改进奖励函数
        price_trend = np.zeros(len(train_data))
        for i in range(5, len(train_data)):
            price_trend[i] = (train_data["收盘"].iloc[i] - train_data["收盘"].iloc[i-5]) / train_data["收盘"].iloc[i-5]
        
        best_profit = -float('inf')
        target_update_freq = 5  # 每5个回合更新一次目标网络
        
        for e in range(episodes):
            print(f"Episode {e+1}/{episodes}")
            state = self.get_state(data, 0)
            
            done = False
            total_profit = 0
            self.inventory = 0
            self.current_step = 0
            last_buy_price = 0
            
            # 使用更高效的训练循环
            for t in range(len(data) - 1):
                # 选择动作
                action = self.act(state)
                
                # 执行动作
                next_state = self.get_state(data, t + 1)
                reward = 0
                
                # 简化奖励计算
                if action == 0:  # 买入
                    if self.inventory == 0:
                        self.inventory += 1
                        last_buy_price = train_data["收盘"].iloc[t]
                        future_idx = min(t + 5, len(data) - 1)
                        future_return = (train_data["收盘"].iloc[future_idx] - train_data["收盘"].iloc[t]) / train_data["收盘"].iloc[t]
                        reward = future_return * 0.2
                elif action == 1:  # 卖出
                    if self.inventory > 0:
                        self.inventory -= 1
                        profit = (train_data["收盘"].iloc[t] - last_buy_price) / last_buy_price
                        reward = profit
                        total_profit += profit
                else:  # 持有
                    reward = -0.001
                    if self.inventory > 0:
                        hold_return = (train_data["收盘"].iloc[t] - last_buy_price) / last_buy_price
                        reward += hold_return * 0.01
                
                done = t == len(data) - 2
                
                self.memorize(state, action, reward, next_state, done)
                state = next_state
                self.current_step += 1
                
                # 减少经验回放频率，每10步回放一次
                if len(self.memory) > batch_size and t % 10 == 0:
                    self.replay(batch_size)
                    
                if done:
                    # 每几个回合才更新一次目标网络，而不是每回合更新
                    if e % target_update_freq == 0:
                        self.update_target_model()
                    print(f"Episode: {e+1}/{episodes}, Total Profit: {total_profit:.4f}")
                    
                    # 保存表现更好的模型
                    if total_profit > best_profit:
                        best_profit = total_profit
                        print(f"保存最佳模型，利润: {best_profit:.4f}")
                    
            # 逐渐减小探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def predict_action(self, state):
        """使用训练好的模型预测动作"""
        state = np.reshape(state, [1, self.lookback_window_size, 5])
        return np.argmax(self.model.predict(state, verbose=0)[0])
    
    def backtest_with_model(self, df: pd.DataFrame):
        """
        使用训练好的模型执行回测
        :param df: 股票数据
        :return: 回测结果
        """
        # 重置状态
        self.current_capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.daily_capital = []
        
        # 保存数据并计算指标
        self.data = df.copy()
        if self.data.empty:
             print("输入数据为空，无法执行回测")
             return self.get_results()
        
        self.data = self.calculate_indicators(self.data)
        
        # 提取特征数据
        data = np.array([
            self.data["norm_open"].values,
            self.data["norm_close"].values,
            self.data["norm_high"].values,
            self.data["norm_low"].values,
            self.data["norm_volume"].values
        ]).T
        
        # 遍历每个交易日
        for t in range(self.lookback_window_size, len(self.data)):
            index = self.data.index[t]
            current_price = self.data["收盘"].iloc[t]
            
            # 获取当前状态
            state = self.get_state(data, t-1)
            
            # 预测动作
            action = self.predict_action(state)
            
            # 0: 买入, 1: 卖出, 2: 持有
            if action == 0 and self.position == 0:  # 买入
                # 计算可买入数量
                max_shares = int((self.current_capital * 0.9) / current_price / 100) * 100
                if max_shares > 0:
                    self.position = max_shares
                    cost = current_price * self.position * (1 + 0.0003)  # 考虑手续费0.03%
                    self.current_capital -= cost
                    self.trades.append({
                        "date": index,
                        "type": "buy",
                        "price": current_price,
                        "shares": self.position,
                        "cost": cost,
                        "capital": self.current_capital
                    })
            
            elif action == 1 and self.position > 0:  # 卖出
                # 卖出所有持仓
                revenue = current_price * self.position * (1 - 0.0003)  # 考虑手续费0.03%
                self.current_capital += revenue
                self.trades.append({
                    "date": index,
                    "type": "sell",
                    "price": current_price,
                    "shares": self.position,
                    "revenue": revenue,
                    "capital": self.current_capital
                })
                self.position = 0
            
            # 记录每日资金情况
            daily_capital = self.current_capital
            if self.position > 0:
                daily_capital += self.position * current_price
            self.daily_capital.append({
                "date": index,
                "capital": daily_capital
            })
            
        return self.get_results()

    # 训练并回测的便捷方法
    def train_and_backtest(self, train_df, test_df, episodes=100, batch_size=32):
        """
        训练模型并执行回测
        :param train_df: 训练数据
        :param test_df: 测试数据
        :param episodes: 训练回合数
        :param batch_size: 批量大小
        :return: 回测结果
        """
        # 计算训练数据的指标
        train_df = self.calculate_indicators(train_df.copy())
        
        # 训练模型
        self.train(train_df, episodes, batch_size)
        
        # 使用训练好的模型进行回测
        return self.backtest_with_model(test_df) 