import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import akshare as ak
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import matplotlib as mpl
import os
from matplotlib.font_manager import FontProperties
from font_manager import init_plot_style

# 初始化中文字体
chinese_font = init_plot_style()

# 风格轮动策略 https://www.zhihu.com/question/21469608/answer/3156342205

class RSRSMomentum(bt.Indicator):
    lines = ('score',)
    params = (('window', 20),)
    
    def __init__(self):
        self.addminperiod(self.p.window)
        
    def next(self):
        if len(self.data) < self.p.window:
            return
            
        # 计算斜率和R²
        prices = np.array(self.data.get(size=self.p.window))
        X = np.arange(1, self.p.window + 1).reshape(-1, 1)
        y = prices / prices[0]  # 标准化价格
        
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        r_squared = model.score(X, y)
        
        # 计算综合得分
        self.lines.score[0] = slope * r_squared * 10000

class MomentumRotationStrategy(bt.Strategy):
    params = (
        ('momentum_window', 20),
        ('hold_period', 1),
        ('printlog', True),
    )

    def __init__(self):
        # 为每个数据创建RSRS指标
        self.scores = {}
        for i, d in enumerate(self.datas):
            self.scores[d._name] = RSRSMomentum(d, window=self.p.momentum_window)
        
        self.counter = 0
        # 追踪交易统计数据
        self.trades_info = []
        self.trade_counts = {'total': 0, 'win': 0, 'loss': 0}

    def next(self):
        self.counter += 1
        
        # 每天计算一次
        if self.counter % self.p.hold_period != 0:
            return
            
        # 获取当前所有标的的得分
        valid_scores = {}
        for d in self.datas:
            if len(d) > self.p.momentum_window:
                score = self.scores[d._name].score[0]
                if not np.isnan(score):
                    valid_scores[d._name] = score
        
        if not valid_scores:
            return
            
        # 选择得分最高的标的
        best_asset = max(valid_scores, key=valid_scores.get)
        
        # 调整持仓
        for d in self.datas:
            if d._name == best_asset:
                if self.getposition(d).size == 0:
                    self.close()  # 先平其他仓位
                    self.order_target_percent(d, target=0.99)  # 保留1%现金
            else:
                self.order_target_percent(d, target=0)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                logstr = 'BUY EXECUTED: %s, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                    order.data._name,
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm)
                
            elif order.issell():
                logstr = 'SELL EXECUTED: %s, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                    order.data._name,
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm)
                
            self.log(logstr)

    def notify_trade(self, trade):
        """跟踪每笔交易的盈亏情况"""
        if trade.isclosed:
            self.trade_counts['total'] += 1
            if trade.pnlcomm > 0:
                self.trade_counts['win'] += 1
            else:
                self.trade_counts['loss'] += 1
                
            self.trades_info.append({
                'symbol': trade.data._name,
                'size': trade.size,
                'price': trade.price,
                'value': trade.value,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'commission': trade.commission,
                'is_win': trade.pnlcomm > 0
            })
            
            logstr = f'TRADE COMPLETED: {trade.data._name}, Profit: {trade.pnlcomm:.2f}, Value: {trade.value:.2f}'
            self.log(logstr)

    def get_win_rate(self):
        """计算交易胜率"""
        if self.trade_counts['total'] == 0:
            return 0.0
        return (self.trade_counts['win'] / self.trade_counts['total']) * 100

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

# 回测设置
def run_backtest():
    cerebro = bt.Cerebro(stdstats=False)
    
    # 设置回测时间范围
    start_date = datetime(2018, 8, 1) 
    end_date = datetime(2025, 3, 31)   
    
    # 定义ETF列表
    etfs = {
        '510300': '沪深300ETF', # 大盘风格
        '512100': '中证1000ETF', # 小盘风格
        '510880': '红利ETF', # 价值风格
        '159915': '创业板ETF' # 成长风格
    }

    # 获取上证指数数据作为基准
    print("正在获取上证指数数据...")
    benchmark_data = None
    benchmark_df = None
    try:
        benchmark_df = ak.stock_zh_index_daily(symbol="sh000001")
        
        # 仅保留回测期间的数据
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        benchmark_df = benchmark_df[(benchmark_df['date'] >= start_date) & (benchmark_df['date'] <= end_date)]
        
        # 重命名列以匹配backtrader需要的格式
        benchmark_df = benchmark_df.rename(columns={
            'date': 'date',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'volume'
        })
        
        # 将日期列设置为索引
        benchmark_df.set_index('date', inplace=True)
        
        # 添加上证指数作为基准数据，但不参与交易
        benchmark_data = bt.feeds.PandasData(dataname=benchmark_df, name='上证指数')
       
        print("成功添加上证指数数据作为基准")
    except Exception as e:
        print(f"获取上证指数数据出错: {str(e)}")
    
    # 使用akshare获取数据
    etf_data_dict = {}
    for code, name in etfs.items():
        try:
            print(f"正在获取 {name}({code}) 的数据...")
            df = ak.fund_etf_hist_em(symbol=code, 
                                   period="daily",
                                   start_date=start_date.strftime('%Y%m%d'),
                                   end_date=end_date.strftime('%Y%m%d'),
                                   adjust="qfq")
            
            # 重命名列以匹配backtrader需要的格式
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            })
            
            # 将日期列设置为索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 保存数据到字典
            etf_data_dict[code] = df
            
            # 创建数据源
            data = bt.feeds.PandasData(dataname=df, name=code)
            cerebro.adddata(data)
            print(f"成功添加 {name}({code}) 的数据")
            
        except Exception as e:
            print(f"获取 {name}({code}) 数据时出错: {str(e)}")
            continue
    
    # 策略参数
    cerebro.addstrategy(MomentumRotationStrategy,
                       momentum_window=25,
                       hold_period=1,
                       printlog=True)
    
    # 回测设置
    cerebro.broker.setcash(100000)  # 设置初始资金为10万
    cerebro.broker.setcommission(commission=0.0005)  # 0.1%手续费
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    
    # 添加净值曲线观察器
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Cash)
    
    # 运行回测
    results = cerebro.run()
    
    # 输出结果
    strat = results[0]
    print('初始资金: 100,000.00')
    print('最终资产价值: %.2f' % cerebro.broker.getvalue())
    print('总收益率: %.2f%%' % ((cerebro.broker.getvalue() / 100000 - 1) * 100))
    print('夏普比率: %.2f' % strat.analyzers.sharpe.get_analysis()['sharperatio'])
    print('最大回撤: %.2f%%' % strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
    print('年化收益率: %.2f%%' % strat.analyzers.returns.get_analysis()['rnorm100'])
    print('交易胜率: %.2f%%' % strat.get_win_rate())
    print('总交易次数: %d' % strat.trade_counts['total'])
    print('盈利交易次数: %d' % strat.trade_counts['win'])
    print('亏损交易次数: %d' % strat.trade_counts['loss'])
    
    # 获取资金变化数据
    pyfolio_returns, positions, transactions, gross_lev = strat.analyzers.pyfolio.get_pf_items()
    
    # 创建自定义的可视化图表
    create_custom_charts(strat, benchmark_df, etf_data_dict, start_date, end_date, pyfolio_returns)
    
    # 绘制backtrader自带的净值曲线，包含上证指数作为基准比较
    # cerebro.plot(style='candlestick', volume=False, iplot=False, barup='red', bardown='green')

def create_custom_charts(strat, benchmark_df, etf_data_dict, start_date, end_date, pyfolio_returns):
    """创建适合小白看懂的可视化图表，将所有图表整合到一个画板中"""
    
    # 获取策略的每日收益率
    returns = pd.Series(strat.analyzers.time_return.get_analysis())
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    
    # 计算累计收益
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # 计算上证指数的累计收益率作为对比
    if benchmark_df is not None:
        bench_returns = benchmark_df['close'].pct_change()
        bench_cumulative = (1 + bench_returns.fillna(0)).cumprod() - 1
    
    # 创建子图布局 - 4行2列，以容纳新增的胜率图表
    fig = plt.figure(figsize=(20, 22))
    
    # 1. 收益对比图 - 展示策略与上证指数的累计收益对比
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(cumulative_returns.index, cumulative_returns * 100, label='风格轮动策略', linewidth=2)
    
    if benchmark_df is not None:
        ax1.plot(bench_cumulative.index, bench_cumulative * 100, label='上证指数', linewidth=2, linestyle='--')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('日期', fontproperties=chinese_font)
    ax1.set_ylabel('累计收益率 (%)', fontproperties=chinese_font)
    ax1.set_title('风格轮动策略 vs 上证指数累计收益对比', fontproperties=chinese_font, fontsize=12)
    # 创建中文图例
    if chinese_font:
        legend = ax1.legend(prop=chinese_font)
    else:
        legend = ax1.legend()
    
    # 2. 月度收益热力图
    monthly_returns = returns.groupby([lambda x: x.year, lambda x: x.month]).apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly_returns = monthly_returns.unstack(level=-1)
    
    # 准备数据
    years = monthly_returns.index.values
    months = [calendar.month_abbr[i] for i in range(1, 13)]
    
    # 绘制热力图
    ax2 = fig.add_subplot(4, 2, 2)
    sns.heatmap(monthly_returns * 100, 
                ax=ax2,
                annot=True, 
                fmt=".1f", 
                cmap='RdYlGn', 
                linewidths=0.5,
                cbar_kws={'label': '月度收益率 (%)'},
                xticklabels=months,
                yticklabels=years)
    ax2.set_title('每月收益率热力图 (%)', fontproperties=chinese_font, fontsize=12)
    
    # 3. 年度收益柱状图
    yearly_returns = returns.groupby(lambda x: x.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    if benchmark_df is not None:
        bench_yearly = bench_returns.groupby(lambda x: x.year).apply(
            lambda x: (1 + x).prod() - 1
        )
    
    ax3 = fig.add_subplot(4, 2, 3)
    
    # 设置柱状图的位置
    bar_width = 0.35
    x = np.arange(len(yearly_returns))
    
    # 绘制策略年度收益
    ax3.bar(x, yearly_returns * 100, bar_width, color='#5cb85c', label='风格轮动策略')
    
    # 绘制基准年度收益
    if benchmark_df is not None:
        ax3.bar(x + bar_width, bench_yearly * 100, bar_width, color='#d9534f', label='上证指数')
    
    # 设置图表格式
    ax3.set_xlabel('年份', fontproperties=chinese_font)
    ax3.set_ylabel('年度收益率 (%)', fontproperties=chinese_font)
    ax3.set_title('策略与基准的年度收益率对比', fontproperties=chinese_font, fontsize=12)
    ax3.set_xticks(x + bar_width / 2)
    ax3.set_xticklabels(yearly_returns.index)
    if chinese_font:
        legend = ax3.legend(prop=chinese_font)
    else:
        legend = ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注具体数值
    for i, v in enumerate(yearly_returns * 100):
        ax3.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    if benchmark_df is not None:
        for i, v in enumerate(bench_yearly * 100):
            ax3.text(i + bar_width, v + 0.5, f"{v:.1f}%", ha='center')
    
    # 4. 持仓表现分析图
    ax4 = fig.add_subplot(4, 2, 4)
    if etf_data_dict:
        # 创建一个基于初始日期的基准数据框
        start_values = {}
        for code, df in etf_data_dict.items():
            if not df.empty:
                # 使用第一个有效价格作为起始点
                first_valid = df['close'].iloc[0]
                if not pd.isna(first_valid) and first_valid > 0:
                    start_values[code] = first_valid
        
        # 计算各ETF的归一化价格曲线
        normalized_prices = pd.DataFrame()
        for code, start_value in start_values.items():
            if code in etf_data_dict and not etf_data_dict[code].empty:
                normalized_prices[code] = etf_data_dict[code]['close'] / start_value
        
        # 绘制所有ETF的表现
        for code in normalized_prices.columns:
            ax4.plot(normalized_prices.index, normalized_prices[code], label=code, linewidth=1.5)
        
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('日期', fontproperties=chinese_font)
        ax4.set_ylabel('归一化价格', fontproperties=chinese_font)
        ax4.set_title('各ETF表现对比', fontproperties=chinese_font, fontsize=12)
        if chinese_font:
            legend = ax4.legend(prop=chinese_font)
        else:
            legend = ax4.legend()
    
    # 5. 资金变化图
    ax5 = fig.add_subplot(4, 2, 5)
    
    # 计算资金曲线（初始资金 * (1+收益率)）
    equity_curve = (1 + pyfolio_returns).cumprod() * 100000
    
    # 绘制资金曲线
    ax5.plot(equity_curve.index, equity_curve, label='账户资金', color='#3366cc', linewidth=2)
    
    # 标记最大回撤区间
    # 找到资金曲线的峰值
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    
    # 找到最大回撤的开始和结束时间
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()
    # 找到这个回撤的开始点
    dd_start = equity_curve[:max_dd_end].idxmax()
    
    # 在图表上标记最大回撤区间
    ax5.fill_between(drawdown.index, 0, drawdown * 100, color='red', alpha=0.3)
    ax5.scatter(dd_start, equity_curve.loc[dd_start], color='green', s=50, marker='^', label='回撤开始')
    ax5.scatter(max_dd_end, equity_curve.loc[max_dd_end], color='red', s=50, marker='v', label='回撤结束')
    
    # 添加最大回撤标注
    dd_text = f"最大回撤: {max_dd * 100:.2f}%"
    ax5.annotate(dd_text, 
                xy=(max_dd_end, equity_curve.loc[max_dd_end]),
                xytext=(30, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontproperties=chinese_font)
    
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('日期', fontproperties=chinese_font)
    ax5.set_ylabel('资金 (元)', fontproperties=chinese_font)
    ax5.set_title('账户资金变化曲线', fontproperties=chinese_font, fontsize=12)
    if chinese_font:
        legend = ax5.legend(prop=chinese_font)
    else:
        legend = ax5.legend()
    
    # 6. 月度收益波动
    ax6 = fig.add_subplot(4, 2, 6)
    
    # 计算月度收益的均值和波动
    monthly_returns_flat = monthly_returns.T.stack()
    monthly_returns_flat = monthly_returns_flat * 100  # 转换为百分比
    
    # 绘制月度收益分布
    sns.boxplot(data=monthly_returns.T * 100, ax=ax6)
    
    ax6.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax6.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])
    ax6.set_xlabel('月份', fontproperties=chinese_font)
    ax6.set_ylabel('收益率 (%)', fontproperties=chinese_font)
    ax6.set_title('月度收益率分布', fontproperties=chinese_font, fontsize=12)

    # 7. 交易胜率饼图
    ax7 = fig.add_subplot(4, 2, 7)
    
    # 获取交易数据
    win_count = strat.trade_counts['win']
    loss_count = strat.trade_counts['loss']
    total_count = strat.trade_counts['total']
    win_rate = strat.get_win_rate()
    
    # 绘制饼图
    labels = ['盈利交易', '亏损交易']
    sizes = [win_count, loss_count]
    colors = ['#5cb85c', '#d9534f']
    explode = (0.1, 0)  # 突出显示盈利部分
    
    ax7.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontproperties': chinese_font})
    ax7.axis('equal')  # 确保饼图是圆的
    ax7.set_title(f'交易胜率分析 (总交易: {total_count}次)', fontproperties=chinese_font, fontsize=12)
    
    # 8. ETF交易分布和收益分析
    ax8 = fig.add_subplot(4, 2, 8)
    
    # 分析每个交易标的的交易次数和平均收益
    if hasattr(strat, 'trades_info') and strat.trades_info:
        # 创建DataFrame来分析交易数据
        trades_df = pd.DataFrame(strat.trades_info)
        if not trades_df.empty:
            # 按交易标的分组统计
            symbol_stats = trades_df.groupby('symbol').agg({
                'pnlcomm': ['sum', 'mean', 'count'],
                'is_win': 'sum'
            })
            
            # 计算每个标的的胜率
            symbol_stats['win_rate'] = symbol_stats[('is_win', 'sum')] / symbol_stats[('pnlcomm', 'count')] * 100
            
            # 准备绘图数据
            symbols = symbol_stats.index
            trade_counts = symbol_stats[('pnlcomm', 'count')]
            win_rates = symbol_stats['win_rate']
            
            # 创建双轴图表
            ax8_count = ax8
            ax8_rate = ax8.twinx()
            
            # 绘制交易次数柱状图
            bars = ax8_count.bar(symbols, trade_counts, color='#5bc0de', alpha=0.7, label='交易次数')
            ax8_count.set_xlabel('交易标的', fontproperties=chinese_font)
            ax8_count.set_ylabel('交易次数', fontproperties=chinese_font)
            
            # 绘制胜率折线图
            line = ax8_rate.plot(symbols, win_rates, 'ro-', linewidth=2, label='胜率')
            ax8_rate.set_ylabel('胜率 (%)', fontproperties=chinese_font)
            
            # 为每个柱添加标签
            for bar in bars:
                height = bar.get_height()
                ax8_count.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontproperties=chinese_font)
            
            # 添加图例
            lines, labels = ax8_count.get_legend_handles_labels()
            lines2, labels2 = ax8_rate.get_legend_handles_labels()
            if chinese_font:
                ax8.legend(lines + lines2, labels + labels2, loc=0, prop=chinese_font)
            else:
                ax8.legend(lines + lines2, labels + labels2, loc=0)
                
            ax8.set_title('各交易标的交易次数与胜率分析', fontproperties=chinese_font, fontsize=12)
            ax8.grid(True, alpha=0.3)
    
    # 调整整体布局
    plt.suptitle('ETF风格轮动策略回测分析', fontproperties=chinese_font, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 显示图表
    plt.show()
    
    print("\n图表展示完成")

def calculate_today_signal(window=25):
    """
    计算当天的ETF交易信号
    
    参数:
    window (int): 计算RSRS动量的窗口期，默认25天
    
    返回:
    dict: 包含交易信号的字典
    """
    # 定义ETF列表
    etfs = {
        '510300': '沪深300ETF', # 大盘风格
        '512100': '中证1000ETF', # 小盘风格
        '510880': '红利ETF', # 价值风格
        '159915': '创业板ETF' # 成长风格
    }
    
    # 设置日期范围 - 获取比窗口更长的历史数据以确保计算准确
    end_date = datetime.now()
    start_date = end_date - timedelta(days=window + 10)  # 多取几天数据，防止节假日等情况
    
    # 存储所有ETF的收盘价数据
    prices_data = {}
    
    # 获取数据
    print("正在获取ETF数据...")
    for code, name in etfs.items():
        try:
            print(f"获取 {name}({code}) 的数据...")
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"
            )
            
            if df.empty:
                print(f"警告: 未获取到 {name}({code}) 数据")
                continue
                
            # 确保日期列为datetime类型
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            
            # 保存收盘价数据
            prices_data[code] = df['收盘']
            print(f"成功获取 {name}({code}) 数据，共{len(df)}条记录")
            
        except Exception as e:
            print(f"获取 {code} 数据时出错: {str(e)}")
    
    # 确保有数据返回
    if not prices_data:
        return {"error": "未能获取任何ETF数据"}
    
    # 计算各ETF的RSRS动量得分
    scores = {}
    latest_prices = {}
    
    for code, prices in prices_data.items():
        try:
            # 确保有足够的数据
            if len(prices) < window:
                print(f"警告: {code} 数据点不足，跳过计算")
                continue
            
            # 获取最近窗口期的收盘价
            recent_prices = prices.iloc[-window:].values
            
            # 数据有效性检查
            if np.any(np.isnan(recent_prices)) or np.any(np.isinf(recent_prices)) or recent_prices[0] == 0:
                print(f"警告: {code} 数据包含无效值")
                continue
            
            # 计算RSRS动量得分
            X = np.arange(1, window + 1).reshape(-1, 1)
            y = recent_prices / recent_prices[0]  # 标准化价格
            
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            # 计算最终得分
            score = slope * r_squared * 10000
            
            # 保存得分和最新价格
            scores[code] = score
            latest_prices[code] = prices.iloc[-1]
            
        except Exception as e:
            print(f"计算 {code} 的动量得分时出错: {str(e)}")
    
    # 确保有得分数据
    if not scores:
        return {"error": "未能计算任何ETF的动量得分"}
    
    # 找出得分最高的ETF
    best_etf = max(scores, key=scores.get)
    
    # 构建结果
    result = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "best_etf": {
            "code": best_etf,
            "name": etfs[best_etf],
            "score": scores[best_etf],
            "price": latest_prices[best_etf]
        },
        "all_scores": {code: {"name": etfs[code], "score": score, "price": latest_prices[code]} 
                     for code, score in scores.items()},
        "action": f"买入 {etfs[best_etf]}({best_etf})"
    }
    
    # 绘制各ETF得分对比图
    plt.figure(figsize=(10, 6))
    codes = list(scores.keys())
    score_values = [scores[code] for code in codes]
    colors = ['#3cb371' if score > 0 else '#e74c3c' for score in score_values]
    
    plt.bar(codes, score_values, color=colors)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('ETF动量得分对比 - ' + datetime.now().strftime('%Y-%m-%d'))
    plt.ylabel('RSRS动量得分')
    plt.tight_layout()
    
    # 保存图表
    chart_filename = 'ETF动量得分_' + datetime.now().strftime('%Y%m%d') + '.png'
    plt.savefig(chart_filename)
    plt.close()
    
    result["chart"] = chart_filename
    
    return result

def print_signal_report(signal):
    """打印交易信号报告"""
    if "error" in signal:
        print(f"获取交易信号出错: {signal['error']}")
        return
        
    print("\n" + "="*50)
    print(f"ETF风格轮动交易信号 - {signal['date']}")
    print("="*50)
    
    best = signal["best_etf"]
    print(f"\n今日推荐: {best['name']}({best['code']})")
    print(f"动量得分: {best['score']:.2f}")
    print(f"最新价格: {best['price']:.4f}")
    print(f"\n交易行动: {signal['action']}")
    
    print("\n所有ETF得分排名:")
    print("-"*40)
    print(f"{'ETF代码':<10}{'ETF名称':<15}{'动量得分':<15}{'最新价格':<10}")
    print("-"*40)
    
    # 按得分从高到低排序
    sorted_etfs = sorted(signal["all_scores"].items(), key=lambda x: x[1]["score"], reverse=True)
    for code, info in sorted_etfs:
        print(f"{code:<10}{info['name']:<15}{info['score']:<15.2f}{info['price']:<10.4f}")
    
    print("\n" + "="*50)
    print(f"信号图表已保存为: {signal.get('chart', '无图表')}")
    print("="*50)

# 添加主函数调用
if __name__ == '__main__':
    # 运行回测
    # run_backtest()
    
    # 计算今日交易信号
    today_signal = calculate_today_signal()
    print_signal_report(today_signal)