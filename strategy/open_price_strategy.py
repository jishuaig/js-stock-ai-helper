import pandas as pd
import numpy as np
from typing import Dict, List
from strategy.base_strategy import BaseStrategy

class OpenPriceStrategy(BaseStrategy):
    def __init__(self, stock_code: str, initial_capital: float = 50000.0,
                 high_threshold: float = 0.005, low_threshold: float = 0.005,
                 position_sizes: List[float] = [0.5, 0.5],  # 分批建仓比例
                 max_total_position_ratio: float = 0.9,  # 最大总仓位比例
                 atr_periods: int = 14,  # ATR周期
                 risk_per_trade_ratio: float = 0.02,  # 单次交易风险比例
                 atr_risk_multiplier: float = 2.0):  # ATR风险倍数
        """
        初始化开盘价策略
        :param stock_code: 股票代码
        :param initial_capital: 初始资金
        :param high_threshold: 高开阈值
        :param low_threshold: 低开阈值
        :param position_sizes: 分批建仓的比例列表，总和应接近1
        :param max_total_position_ratio: 最大总仓位占总资金的比例
        :param atr_periods: ATR计算周期
        :param risk_per_trade_ratio: 单次交易风险占总资金的比例
        :param atr_risk_multiplier: 计算风险时ATR的倍数
        """
        super().__init__(stock_code, initial_capital)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        # 仓位管理参数
        if not np.isclose(sum(position_sizes), 1.0):
            print(f"警告: position_sizes 总和 {sum(position_sizes)} 不接近 1.0，将重新归一化。")
            total = sum(position_sizes)
            self.position_sizes = [p / total for p in position_sizes]
        else:
            self.position_sizes = position_sizes
        self.max_total_position_ratio = max_total_position_ratio
        self.atr_periods = atr_periods
        self.risk_per_trade_ratio = risk_per_trade_ratio
        self.atr_risk_multiplier = atr_risk_multiplier

        # 仓位管理状态
        self.current_batch = 0  # 当前建仓批次索引
        self.batch_trades = [] # 记录每批买入的交易信息 (shares, price)

    def reset_position_state(self):
        """重置仓位状态"""
        self.current_batch = 0
        self.batch_trades = []
        self.position = 0 # 确保基类中的仓位也重置

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标 (包括ATR)"""
        if '开盘' not in df.columns or '收盘' not in df.columns or '最高' not in df.columns or '最低' not in df.columns:
            print("数据缺少必要的列 ('开盘', '收盘', '最高', '最低')，请检查数据格式")
            # 返回原始df或引发错误可能更合适
            return df

        # 计算开盘差幅
        df['前日收盘'] = df['收盘'].shift(1)
        df['开盘差幅'] = (df['开盘'] - df['前日收盘']) / df['前日收盘']

        # 标记高开和低开
        df['高开'] = df['开盘差幅'] > self.high_threshold
        df['低开'] = df['开盘差幅'] < -self.low_threshold

        # 计算ATR
        df['TR'] = pd.DataFrame({
            'HL': df['最高'] - df['最低'],
            'HC': abs(df['最高'] - df['收盘'].shift(1)),
            'LC': abs(df['最低'] - df['收盘'].shift(1))
        }).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=self.atr_periods, min_periods=self.atr_periods).mean()

        return df

    def get_trading_signal(self, row: pd.Series) -> str:
        """获取交易信号 (考虑分批建仓)"""
        # 检查数据有效性
        if pd.isna(row.get('前日收盘')) or pd.isna(row.get('ATR')):
            return 'hold'

        # 高开买入信号 (只有在未完成所有批次建仓时才触发)
        if row['高开'] and self.current_batch < len(self.position_sizes):
            # 检查是否超过最大总仓位限制(预估)
            potential_total_capital_at_risk = self.current_capital + self.position * row['收盘'] # 近似总权益
            max_allowed_position_value = potential_total_capital_at_risk * self.max_total_position_ratio
            current_position_value = self.position * row['收盘']
            if current_position_value < max_allowed_position_value:
                 return 'buy'
            else:
                return 'hold' # 已达最大仓位，不再买入

        # 低开卖出信号 (清仓)
        if row['低开'] and self.position > 0:
            return 'sell'

        return 'hold'

    def calculate_trade_size(self, price: float, atr: float, total_equity: float) -> int:
        """根据风险和资金计算当前批次的交易数量"""
        if self.current_batch >= len(self.position_sizes):
            return 0 # 已完成所有批次

        # 1. 基于风险的规模计算
        risk_amount_per_trade = total_equity * self.risk_per_trade_ratio
        atr_risk = atr * self.atr_risk_multiplier
        if atr_risk <= 0: # 避免除以零
             risk_based_shares = float('inf') # 如果ATR风险为0，理论上可以买无限多？给个极大值
        else:
             risk_based_shares = risk_amount_per_trade / atr_risk

        # 2. 基于资金比例的规模计算
        # 注意：这里的资金比例是针对 *当前批次* 占 *总计划仓位* 的比例
        # 总计划投入资金 = 总权益 * 最大总仓位比例
        # 当前批次计划投入资金 = 总计划投入资金 * 当前批次比例
        target_total_position_value = total_equity * self.max_total_position_ratio
        batch_position_ratio = self.position_sizes[self.current_batch]
        capital_for_this_batch = target_total_position_value * batch_position_ratio
        capital_based_shares = capital_for_this_batch / price

        # 取较小值，并向下取整到100股的倍数
        shares = min(risk_based_shares, capital_based_shares)
        shares = int(shares / 100) * 100

        # 确保不超过可用现金
        max_affordable_shares = int((self.current_capital / price) / 100) * 100
        shares = min(shares, max_affordable_shares)

        # 确保最终总仓位不超过限制
        current_position_value = self.position * price
        max_allowed_total_value = total_equity * self.max_total_position_ratio
        allowed_additional_value = max_allowed_total_value - current_position_value
        if allowed_additional_value <=0:
            return 0
        max_additional_shares = int((allowed_additional_value / price)/100)*100
        shares = min(shares, max_additional_shares)


        return shares if shares > 0 else 0


    # 重写 backtest 方法
    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        执行回测 (重写以支持分批建仓和仓位管理)
        :param df: 股票数据，包含必要的价格和成交量数据
        :return: 回测结果字典
        """
        # 重置状态
        self.current_capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.daily_capital = []
        self.reset_position_state() # 重置分批建仓状态

        # 保存数据并计算指标
        self.data = df.copy()
        if self.data.empty:
             print("输入数据为空，无法执行回测")
             return self.get_results() # 返回空结果

        self.data = self.calculate_indicators(self.data)
        if self.data.empty or 'ATR' not in self.data.columns: # 确保指标计算成功
             print("指标计算失败或数据不足，无法执行回测")
             return self.get_results()

        # 打印表头
        print("\n策略执行情况：")
        print(f"{'日期':<12}{'开盘价':>8}{'收盘价':>8}{'最高价':>8}{'最低价':>8}{'开盘差幅':>10}{'ATR':>8}{'信号':>8}{'仓位':>10}{'资金':>14}")
        print("-" * 100)

        # 遍历每个交易日
        for index, row in self.data.iterrows():
            current_price = row['收盘'] # 使用收盘价进行交易决策和记录 (或可改为开盘价)
            current_atr = row['ATR']

             # 检查数据有效性
            if pd.isna(current_price) or pd.isna(current_atr):
                 # 记录当天资金 (假设维持前一天状态)
                 if self.daily_capital:
                     self.daily_capital.append({
                         'date': index,
                         'capital': self.daily_capital[-1]['capital']
                     })
                 else:
                     self.daily_capital.append({
                         'date': index,
                         'capital': self.initial_capital
                     })
                 continue # 跳过这个无效数据点

            # 获取交易信号
            signal = self.get_trading_signal(row)

            # 计算当前总权益（现金+持仓市值）
            total_equity = self.current_capital + self.position * current_price
            
            # 打印当日数据和策略执行情况
            date_str = index.strftime('%Y-%m-%d')
            diff_pct = row.get('开盘差幅', 0) * 100 if not pd.isna(row.get('开盘差幅', 0)) else 0
            print(f"{date_str:<12}{row['开盘']:8.3f}{row['收盘']:8.3f}{row['最高']:8.3f}{row['最低']:8.3f}{diff_pct:10.2f}%{row['ATR']:8.3f}{signal:>8}{self.position:10}{self.current_capital:14.2f}")

            # 执行交易
            if signal == 'buy' and self.current_batch < len(self.position_sizes):
                trade_size = self.calculate_trade_size(current_price, current_atr, total_equity)

                if trade_size > 0:
                    cost = current_price * trade_size * (1 + 0.0003)  # 考虑手续费
                    if cost <= self.current_capital: # 再次确认现金足够
                        self.position += trade_size
                        self.current_capital -= cost
                        self.batch_trades.append({'shares': trade_size, 'price': current_price})
                        self.current_batch += 1
                        self.trades.append({
                            'date': index,
                            'type': 'buy',
                            'price': current_price,
                            'shares': trade_size,
                            'cost': cost,
                            'capital_after': self.current_capital,
                            'batch': self.current_batch # 记录这是第几批买入
                        })
                        # 打印买入交易详情
                        print(f"    >>> 买入: {trade_size}股, 价格: {current_price:.3f}, 成本: {cost:.2f}, 剩余资金: {self.current_capital:.2f}, 第{self.current_batch}批")

            elif signal == 'sell' and self.position > 0:
                # 卖出所有持仓
                revenue = current_price * self.position * (1 - 0.0003)  # 考虑手续费
                shares_sold = self.position
                self.current_capital += revenue
                self.position = 0 # 清空仓位
                self.trades.append({
                    'date': index,
                    'type': 'sell',
                    'price': current_price,
                    'shares': shares_sold, # 记录卖出的总量
                    'revenue': revenue,
                    'capital_after': self.current_capital
                })
                # 打印卖出交易详情
                print(f"    >>> 卖出: {shares_sold}股, 价格: {current_price:.3f}, 收入: {revenue:.2f}, 剩余资金: {self.current_capital:.2f}")
                self.reset_position_state() # 重置分批建仓状态

            # 记录每日资金情况 (需要在交易之后更新总权益)
            current_total_equity = self.current_capital + self.position * current_price
            self.daily_capital.append({
                'date': index,
                'capital': current_total_equity
            })

        # 回测结束，强制平仓最后的持仓
        if self.position > 0:
            last_price = self.data['收盘'].iloc[-1]
            if not pd.isna(last_price): # 确保最后价格有效
                revenue = last_price * self.position * (1 - 0.0003)
                shares_sold = self.position
                self.current_capital += revenue
                self.trades.append({
                    'date': self.data.index[-1],
                    'type': 'sell',
                    'price': last_price,
                    'shares': shares_sold,
                    'revenue': revenue,
                    'capital_after': self.current_capital,
                    'note': 'End of backtest force sell'
                })
                # 打印强制平仓详情
                print(f"\n    >>> 回测结束强制平仓: {shares_sold}股, 价格: {last_price:.3f}, 收入: {revenue:.2f}, 最终资金: {self.current_capital:.2f}")
                self.position = 0
                self.reset_position_state()
                # 更新最后一天的资金记录
                if self.daily_capital:
                    self.daily_capital[-1]['capital'] = self.current_capital


        # 返回结果
        return self.get_results()

    def get_results(self) -> Dict:
         """获取回测结果"""
         performance_metrics = self.calculate_performance_metrics() # 使用基类的方法计算指标
         return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital if not self.daily_capital else self.daily_capital[-1]['capital'], # 以daily_capital为准
            'returns': (self.daily_capital[-1]['capital'] - self.initial_capital) / self.initial_capital if self.daily_capital else 0,
            'transactions': self.trades,
            'daily_capital': self.daily_capital,
            'metrics': performance_metrics
        }