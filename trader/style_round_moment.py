import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import akshare as ak
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 风格轮动策略 https://www.zhihu.com/question/21469608/answer/3156342205

class RSRSMomentum(bt.Indicator):
    lines = ('score',)
    params = (('window', 25),)
    
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
        ('momentum_window', 25),
        ('hold_period', 1),
        ('printlog', True),
    )

    def __init__(self):
        # 为每个数据创建RSRS指标
        self.scores = {}
        for i, d in enumerate(self.datas):
            self.scores[d._name] = RSRSMomentum(d, window=self.p.momentum_window)
        
        self.counter = 0

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

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

# 回测设置
def run_backtest():
    cerebro = bt.Cerebro(stdstats=False)
    
    # 设置回测时间范围
    start_date = datetime(2025, 1, 1)  # 2018年8月1日
    end_date = datetime(2025, 4, 1)   # 2022年3月31日
    
    # 定义ETF列表
    etfs = {
        '510300': '沪深300ETF',
        '510500': '中证500ETF',
        '510880': '红利ETF',
        '159915': '创业板ETF'
    }
    
    # 使用akshare获取数据
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
    cerebro.broker.setcash(1000000)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 运行回测
    results = cerebro.run()
    
    # 输出结果
    strat = results[0]
    print('最终资产价值: %.2f' % cerebro.broker.getvalue())
    print('夏普比率:', strat.analyzers.sharpe.get_analysis()['sharperatio'])
    print('最大回撤:', strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
    print('年化收益率:', strat.analyzers.returns.get_analysis()['rnorm100'])
    
    # 绘制净值曲线
    cerebro.plot(style='candlestick', volume=False)

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
        '510300': '沪深300ETF',
        '510500': '中证500ETF',
        '510880': '红利ETF',
        '159915': '创业板ETF'
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