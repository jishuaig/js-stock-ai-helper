import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import akshare as ak
from datetime import datetime, timedelta

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

if __name__ == '__main__':
    run_backtest()