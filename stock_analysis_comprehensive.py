import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

class ComprehensiveStockAnalyzer:
    def __init__(self, code: str):
        """
        初始化分析器
        :param code: 代码，例如：sh510300（ETF）或 sh600036（股票）
        """
        self.code = code
        self.data = None
        self.indicators = {}
        self.realtime_data = None
        self.historical_data = None
        self.analysis_result = {}
        # 判断是否为ETF
        self.is_etf = code.startswith('sh51') or code.startswith('sz15')

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取数据"""
        try:
            if self.is_etf:
                # 获取ETF数据
                df = ak.fund_etf_hist_em(symbol=self.code[2:], period="daily", 
                                       start_date=start_date, end_date=end_date, adjust="qfq")
            else:
                # 获取股票数据
                df = ak.stock_zh_a_hist(symbol=self.code[2:], period="daily", 
                                      start_date=start_date, end_date=end_date, adjust="qfq")
            
            # 统一列名
            column_map = {
                '日期': '日期',
                '开盘': '开盘',
                '收盘': '收盘',
                '最高': '最高',
                '最低': '最低',
                '成交量': '成交量',
                '成交额': '成交额',
                '振幅': '振幅',
                '涨跌幅': '涨跌幅',
                '涨跌额': '涨跌额',
                '换手率': '换手率'
            }
            df.columns = [column_map.get(col, col) for col in df.columns]
            
            # 将日期设为索引
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            
            # 计算基本指标
            df['MA5'] = df['收盘'].rolling(window=5).mean()
            df['MA10'] = df['收盘'].rolling(window=10).mean()
            df['MA20'] = df['收盘'].rolling(window=20).mean()
            df['MA60'] = df['收盘'].rolling(window=60).mean()
            
            # 计算MACD
            exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
            exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']
            
            # 计算RSI
            delta = df['收盘'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算布林带
            df['BB_Middle'] = df['收盘'].rolling(window=20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['收盘'].rolling(window=20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['收盘'].rolling(window=20).std()
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"获取数据失败: {str(e)}")
            return None

    def calculate_indicators(self) -> Dict:
        """计算技术指标"""
        if self.data is None:
            return {}
            
        df = self.data
        
        # 计算趋势指标
        self.indicators['trend'] = {
            'ma5': df['MA5'].iloc[-1],
            'ma10': df['MA10'].iloc[-1],
            'ma20': df['MA20'].iloc[-1],
            'ma60': df['MA60'].iloc[-1],
            'price': df['收盘'].iloc[-1],
            'ma_trend': '上升' if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] else '下降'
        }
        
        # 计算MACD指标
        self.indicators['macd'] = {
            'macd': df['MACD'].iloc[-1],
            'signal': df['Signal'].iloc[-1],
            'hist': df['MACD_Hist'].iloc[-1],
            'trend': '上升' if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else '下降'
        }
        
        # 计算RSI指标
        self.indicators['rsi'] = {
            'value': df['RSI'].iloc[-1],
            'status': '超买' if df['RSI'].iloc[-1] > 70 else '超卖' if df['RSI'].iloc[-1] < 30 else '正常'
        }
        
        # 计算布林带指标
        self.indicators['bollinger'] = {
            'upper': df['BB_Upper'].iloc[-1],
            'middle': df['BB_Middle'].iloc[-1],
            'lower': df['BB_Lower'].iloc[-1],
            'position': '上轨' if df['收盘'].iloc[-1] > df['BB_Upper'].iloc[-1] else 
                       '下轨' if df['收盘'].iloc[-1] < df['BB_Lower'].iloc[-1] else '中轨'
        }
        
        # 计算成交量指标
        self.indicators['volume'] = {
            'current': df['成交量'].iloc[-1],
            'ma5': df['成交量'].rolling(window=5).mean().iloc[-1],
            'trend': '放量' if df['成交量'].iloc[-1] > df['成交量'].rolling(window=5).mean().iloc[-1] else '缩量'
        }
        
        return self.indicators
        
    def analyze(self) -> Dict:
        """综合分析"""
        if self.data is None or len(self.indicators) == 0:
            return {}
            
        analysis = {
            'trend_analysis': self._analyze_trend(),
            'momentum_analysis': self._analyze_momentum(),
            'volatility_analysis': self._analyze_volatility(),
            'volume_analysis': self._analyze_volume(),
            'summary': self._generate_summary()
        }
        
        return analysis
        
    def _analyze_trend(self) -> Dict:
        """趋势分析"""
        trend = self.indicators['trend']
        return {
            'status': '上升趋势' if trend['ma_trend'] == '上升' else '下降趋势',
            'strength': '强势' if trend['price'] > trend['ma5'] > trend['ma20'] else 
                      '弱势' if trend['price'] < trend['ma5'] < trend['ma20'] else '震荡',
            'ma_cross': '金叉' if trend['ma5'] > trend['ma20'] else '死叉'
        }
        
    def _analyze_momentum(self) -> Dict:
        """动量分析"""
        macd = self.indicators['macd']
        rsi = self.indicators['rsi']
        return {
            'macd_status': '强势' if macd['trend'] == '上升' and macd['hist'] > 0 else '弱势',
            'rsi_status': rsi['status'],
            'momentum': '强势' if rsi['value'] > 50 and macd['trend'] == '上升' else '弱势'
        }
        
    def _analyze_volatility(self) -> Dict:
        """波动性分析"""
        bollinger = self.indicators['bollinger']
        return {
            'volatility': '高' if bollinger['position'] in ['上轨', '下轨'] else '低',
            'position': bollinger['position'],
            'band_width': (bollinger['upper'] - bollinger['lower']) / bollinger['middle']
        }
        
    def _analyze_volume(self) -> Dict:
        """成交量分析"""
        volume = self.indicators['volume']
        return {
            'volume_status': volume['trend'],
            'strength': '强势' if volume['current'] > volume['ma5'] * 1.5 else '弱势'
        }
        
    def _generate_summary(self) -> Dict:
        """生成总结"""
        trend = self._analyze_trend()
        momentum = self._analyze_momentum()
        volatility = self._analyze_volatility()
        volume = self._analyze_volume()
        
        # 计算综合得分
        score = 0
        
        # 趋势得分（最重要，4分）
        if trend['status'] == '上升趋势':
            score += 2
        if trend['strength'] == '强势':
            score += 1
        if trend['ma_cross'] == '金叉':
            score += 1
        
        # 动量得分（3分）
        if momentum['macd_status'] == '强势':
            score += 1
        if momentum['rsi_status'] == '超卖':
            score += 1
        elif momentum['rsi_status'] == '正常' and momentum['momentum'] == '强势':
            score += 1
        
        # 波动性得分（2分）
        if volatility['volatility'] == '低':
            score += 1
        if float(volatility['band_width']) < 0.05:  # 窄幅盘整
            score += 1
        
        # 成交量得分（1分）
        if volume['strength'] == '强势' and volume['volume_status'] == '放量':
            score += 1
        
        # 调整评分标准
        status = '强势'
        if score >= 7:  # 需要更高的分数才能判定为强势
            status = '强势'
        elif score <= 3:  # 更容易判定为弱势
            status = '弱势'
        else:
            status = '中性'
        
        return {
            'score': score,
            'overall_status': status,
            'suggestion': self._generate_suggestion(trend, momentum, volatility, volume)
        }
        
    def _generate_suggestion(self, trend: Dict, momentum: Dict, volatility: Dict, volume: Dict) -> str:
        """生成建议"""
        # 买入条件：趋势向上 + 动量强势 + 合理波动 + 成交量配合
        if (trend['status'] == '上升趋势' and 
            trend['strength'] == '强势' and
            momentum['momentum'] == '强势' and 
            volatility['volatility'] != '高' and
            volume['strength'] == '强势'):
            return '建议持有或逢低买入'
        # 卖出条件：趋势向下 + 动量弱势 或 高波动
        elif ((trend['status'] == '下降趋势' and momentum['momentum'] == '弱势') or
              volatility['volatility'] == '高'):
            return '建议观望或逢高卖出'
        else:
            return '建议观望，等待更明确的信号'

    def get_realtime_data(self) -> Dict:
        """获取实时数据"""
        try:
            if self.is_etf:
                # 获取ETF实时数据
                df = ak.fund_etf_spot_em()
                stock_data = df[df['代码'] == self.code[2:]].iloc[0]
            else:
                # 获取股票实时数据
                df = ak.stock_zh_a_spot_em()
                stock_data = df[df['代码'] == self.code[2:]].iloc[0]
            
            self.realtime_data = stock_data.to_dict()
            return self.realtime_data
        except Exception as e:
            print(f"获取实时数据失败: {str(e)}")
            return {}

    def get_historical_data(self, days: int = 60) -> Dict:
        """获取历史数据"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            if self.is_etf:
                # 获取ETF历史数据
                df = ak.fund_etf_hist_em(symbol=self.code[2:], period="daily", 
                                       start_date=start_date, end_date=end_date, adjust="qfq")
            else:
                # 获取股票历史数据
                df = ak.stock_zh_a_hist(symbol=self.code[2:], period="daily", 
                                      start_date=start_date, end_date=end_date, adjust="qfq")
            
            self.historical_data = df.to_dict('records')
            return self.historical_data
        except Exception as e:
            print(f"获取历史数据失败: {str(e)}")
            return {}

    def analyze_price_pattern(self) -> Dict:
        """分析价格形态"""
        try:
            if not self.historical_data:
                return {"error": "没有历史数据"}

            df = pd.DataFrame(self.historical_data)
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期')

            # 计算K线形态
            patterns = {
                "双底形态": self._check_double_bottom(df),
                "头肩顶形态": self._check_head_shoulders(df),
                "三角形形态": self._check_triangle(df),
                "突破形态": self._check_breakout(df)
            }

            return patterns
        except Exception as e:
            print(f"分析价格形态失败: {str(e)}")
            return {}

    def analyze_volume(self) -> Dict:
        """分析成交量"""
        try:
            if not self.historical_data:
                return {"error": "没有历史数据"}

            df = pd.DataFrame(self.historical_data)
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期')

            # 计算成交量指标
            volume_analysis = {
                "放量情况": self._check_volume_increase(df),
                "量价配合": self._check_volume_price_correlation(df),
                "大单交易": self._check_large_orders(df),
                "换手率变化": self._check_turnover_rate(df)
            }

            return volume_analysis
        except Exception as e:
            print(f"分析成交量失败: {str(e)}")
            return {}

    def analyze_market_sentiment(self) -> Dict:
        """分析市场情绪"""
        try:
            if not self.realtime_data:
                return {"error": "没有实时数据"}

            sentiment_analysis = {
                "涨速分析": self._analyze_price_speed(),
                "振幅分析": self._analyze_amplitude(),
                "市场关注度": self._analyze_market_attention()
            }

            return sentiment_analysis
        except Exception as e:
            print(f"分析市场情绪失败: {str(e)}")
            return {}

    def analyze_risk(self) -> Dict:
        """分析风险"""
        try:
            risk_analysis = {
                "技术风险": self._analyze_technical_risk(),
                "市场风险": self._analyze_market_risk(),
                "操作风险": self._analyze_operation_risk()
            }

            return risk_analysis
        except Exception as e:
            print(f"分析风险失败: {str(e)}")
            return {}

    def generate_trading_signal(self) -> Dict:
        """生成交易信号"""
        try:
            # 综合分析所有维度
            price_pattern = self.analyze_price_pattern()
            volume_analysis = self.analyze_volume()
            market_sentiment = self.analyze_market_sentiment()
            risk_analysis = self.analyze_risk()
            technical_indicators = self.calculate_indicators()

            # 生成交易信号
            signal = {
                "交易方向": self._determine_trading_direction(
                    price_pattern, volume_analysis, market_sentiment, 
                    risk_analysis, technical_indicators
                ),
                "信号强度": self._calculate_signal_strength(
                    price_pattern, volume_analysis, market_sentiment, 
                    risk_analysis, technical_indicators
                ),
                "建议价格": self._suggest_price_levels(),
                "止损位": self._calculate_stop_loss(),
                "止盈位": self._calculate_take_profit(),
                "建议仓位": self._suggest_position_size(),
                "分析依据": {
                    "价格形态": price_pattern,
                    "成交量分析": volume_analysis,
                    "市场情绪": market_sentiment,
                    "风险分析": risk_analysis,
                    "技术指标": technical_indicators
                }
            }

            return signal
        except Exception as e:
            print(f"生成交易信号失败: {str(e)}")
            return {}

    # 私有辅助方法
    def _check_double_bottom(self, df: pd.DataFrame) -> Dict:
        """检查双底形态
        双底形态特征：
        1. 两个低点价格相近（差异不超过5%）
        2. 两个低点之间有一个明显的反弹高点
        3. 两个低点之间的时间间隔适中（10-60个交易日）
        """
        if len(df) < 60:
            return {"形态": "无", "可信度": 0, "说明": "数据不足"}
        
        # 获取最近60个交易日的数据
        recent_data = df.tail(60)
        lows = recent_data['最低'].values
        dates = recent_data.index
        
        # 寻找局部低点
        bottoms = []
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                bottoms.append((i, lows[i], dates[i]))
        
        if len(bottoms) < 2:
            return {"形态": "无", "可信度": 0, "说明": "未找到足够的低点"}
        
        # 检查最后两个低点是否形成双底
        last_two_bottoms = bottoms[-2:]
        price_diff = abs(last_two_bottoms[0][1] - last_two_bottoms[1][1]) / last_two_bottoms[0][1]
        days_between = (last_two_bottoms[1][2] - last_two_bottoms[0][2]).days
        
        # 检查两个低点之间的反弹
        between_data = recent_data.loc[last_two_bottoms[0][2]:last_two_bottoms[1][2]]
        rebound_high = between_data['最高'].max()
        rebound_ratio = (rebound_high - last_two_bottoms[0][1]) / last_two_bottoms[0][1]
        
        if price_diff <= 0.05 and 10 <= days_between <= 60 and rebound_ratio >= 0.1:
            return {
                "形态": "双底",
                "可信度": 0.8,
                "说明": f"检测到双底形态，两个低点价格差异{price_diff*100:.1f}%，间隔{days_between}天",
                "细节": {
                    "第一底": {"日期": last_two_bottoms[0][2].strftime('%Y-%m-%d'), "价格": last_two_bottoms[0][1]},
                    "第二底": {"日期": last_two_bottoms[1][2].strftime('%Y-%m-%d'), "价格": last_two_bottoms[1][1]},
                    "反弹高点": rebound_high,
                    "反弹幅度": f"{rebound_ratio*100:.1f}%"
                }
            }
        
        return {"形态": "无", "可信度": 0, "说明": "未检测到双底形态"}

    def _check_head_shoulders(self, df: pd.DataFrame) -> Dict:
        """检查头肩顶形态
        头肩顶形态特征：
        1. 三个高点，中间高点最高
        2. 左右两个高点价格相近（差异不超过5%）
        3. 三个高点之间的时间间隔相近
        """
        if len(df) < 60:
            return {"形态": "无", "可信度": 0, "说明": "数据不足"}
        
        recent_data = df.tail(60)
        highs = recent_data['最高'].values
        dates = recent_data.index
        
        # 寻找局部高点
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i], dates[i]))
        
        if len(peaks) < 3:
            return {"形态": "无", "可信度": 0, "说明": "未找到足够的高点"}
        
        # 检查最后三个高点是否形成头肩顶
        last_three_peaks = peaks[-3:]
        if last_three_peaks[1][1] > last_three_peaks[0][1] and \
           last_three_peaks[1][1] > last_three_peaks[2][1]:
            # 检查左右肩价格是否相近
            shoulder_diff = abs(last_three_peaks[0][1] - last_three_peaks[2][1]) / last_three_peaks[0][1]
            # 检查时间间隔是否相近
            left_interval = (last_three_peaks[1][2] - last_three_peaks[0][2]).days
            right_interval = (last_three_peaks[2][2] - last_three_peaks[1][2]).days
            interval_diff = abs(left_interval - right_interval)
            
            if shoulder_diff <= 0.05 and interval_diff <= 10:
                return {
                    "形态": "头肩顶",
                    "可信度": 0.7,
                    "说明": f"检测到头肩顶形态，左右肩价格差异{shoulder_diff*100:.1f}%",
                    "细节": {
                        "左肩": {"日期": last_three_peaks[0][2].strftime('%Y-%m-%d'), "价格": last_three_peaks[0][1]},
                        "头部": {"日期": last_three_peaks[1][2].strftime('%Y-%m-%d'), "价格": last_three_peaks[1][1]},
                        "右肩": {"日期": last_three_peaks[2][2].strftime('%Y-%m-%d'), "价格": last_three_peaks[2][1]},
                        "左肩间隔": f"{left_interval}天",
                        "右肩间隔": f"{right_interval}天"
                    }
                }
        
        return {"形态": "无", "可信度": 0, "说明": "未检测到头肩顶形态"}

    def _check_triangle(self, df: pd.DataFrame) -> Dict:
        """检查三角形形态
        三角形形态特征：
        1. 上升三角形：高点基本持平，低点逐步抬高
        2. 下降三角形：低点基本持平，高点逐步降低
        3. 对称三角形：高点和低点都向中间收敛
        """
        if len(df) < 30:
            return {"形态": "无", "可信度": 0, "说明": "数据不足"}
        
        recent_data = df.tail(30)
        highs = recent_data['最高'].values
        lows = recent_data['最低'].values
        
        # 计算高点和低点的趋势
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # 计算高点和低点的波动范围
        high_std = np.std(highs)
        low_std = np.std(lows)
        
        # 判断三角形类型
        if abs(high_slope) < 0.01 and low_slope > 0.01:
            return {
                "形态": "上升三角形",
                "可信度": 0.6,
                "说明": "检测到上升三角形形态，高点基本持平，低点逐步抬高",
                "细节": {
                    "高点斜率": f"{high_slope:.3f}",
                    "低点斜率": f"{low_slope:.3f}",
                    "高点波动": f"{high_std:.2f}",
                    "低点波动": f"{low_std:.2f}"
                }
            }
        elif high_slope < -0.01 and abs(low_slope) < 0.01:
            return {
                "形态": "下降三角形",
                "可信度": 0.6,
                "说明": "检测到下降三角形形态，低点基本持平，高点逐步降低",
                "细节": {
                    "高点斜率": f"{high_slope:.3f}",
                    "低点斜率": f"{low_slope:.3f}",
                    "高点波动": f"{high_std:.2f}",
                    "低点波动": f"{low_std:.2f}"
                }
            }
        elif high_slope < -0.01 and low_slope > 0.01:
            return {
                "形态": "对称三角形",
                "可信度": 0.6,
                "说明": "检测到对称三角形形态，高点和低点都向中间收敛",
                "细节": {
                    "高点斜率": f"{high_slope:.3f}",
                    "低点斜率": f"{low_slope:.3f}",
                    "高点波动": f"{high_std:.2f}",
                    "低点波动": f"{low_std:.2f}"
                }
            }
        
        return {"形态": "无", "可信度": 0, "说明": "未检测到三角形形态"}

    def _check_breakout(self, df: pd.DataFrame) -> Dict:
        """检查突破形态
        突破形态特征：
        1. 价格突破前期高点或低点
        2. 成交量显著放大
        3. 突破后价格持续运行在突破位置之上或之下
        """
        if len(df) < 20:
            return {"形态": "无", "可信度": 0, "说明": "数据不足"}
        
        recent_data = df.tail(20)
        current_price = recent_data['收盘'].iloc[-1]
        current_volume = recent_data['成交量'].iloc[-1]
        
        # 计算前期高点和低点
        prev_high = recent_data['最高'].iloc[:-1].max()
        prev_low = recent_data['最低'].iloc[:-1].min()
        
        # 计算成交量均值
        volume_ma = recent_data['成交量'].iloc[:-1].mean()
        
        # 判断突破方向
        if current_price > prev_high and current_volume > volume_ma * 1.5:
            return {
                "形态": "向上突破",
                "可信度": 0.9,
                "说明": "检测到向上突破形态，价格突破前期高点且成交量放大",
                "细节": {
                    "突破价格": current_price,
                    "前期高点": prev_high,
                    "突破幅度": f"{(current_price/prev_high-1)*100:.1f}%",
                    "成交量放大": f"{(current_volume/volume_ma-1)*100:.1f}%"
                }
            }
        elif current_price < prev_low and current_volume > volume_ma * 1.5:
            return {
                "形态": "向下突破",
                "可信度": 0.9,
                "说明": "检测到向下突破形态，价格突破前期低点且成交量放大",
                "细节": {
                    "突破价格": current_price,
                    "前期低点": prev_low,
                    "突破幅度": f"{(current_price/prev_low-1)*100:.1f}%",
                    "成交量放大": f"{(current_volume/volume_ma-1)*100:.1f}%"
                }
            }
        
        return {"形态": "无", "可信度": 0, "说明": "未检测到突破形态"}

    def _check_volume_increase(self, df: pd.DataFrame) -> Dict:
        """检查放量情况
        放量特征：
        1. 当日成交量显著高于前期均值
        2. 连续多日成交量逐步增加
        3. 成交量增加与价格变动方向一致
        """
        if len(df) < 10:
            return {"状态": "无", "程度": "无", "说明": "数据不足"}
        
        recent_data = df.tail(10)
        current_volume = recent_data['成交量'].iloc[-1]
        volume_ma5 = recent_data['成交量'].rolling(window=5).mean().iloc[-1]
        volume_ma10 = recent_data['成交量'].rolling(window=10).mean().iloc[-1]
        
        # 计算成交量增加程度
        volume_increase = (current_volume / volume_ma5 - 1) * 100
        
        # 检查连续放量
        volume_trend = recent_data['成交量'].values
        volume_increasing = all(volume_trend[i] >= volume_trend[i-1] for i in range(1, len(volume_trend)))
        
        # 检查量价配合
        price_trend = recent_data['收盘'].values
        price_increasing = price_trend[-1] > price_trend[-2]
        volume_price_match = (volume_increasing and price_increasing) or \
                            (not volume_increasing and not price_increasing)
        
        if volume_increase > 50:
            return {
                "状态": "放量",
                "程度": "显著",
                "说明": f"成交量显著增加{volume_increase:.1f}%",
                "细节": {
                    "当前成交量": current_volume,
                    "5日均量": volume_ma5,
                    "10日均量": volume_ma10,
                    "连续放量": "是" if volume_increasing else "否",
                    "量价配合": "良好" if volume_price_match else "不佳"
                }
            }
        elif volume_increase > 20:
            return {
                "状态": "放量",
                "程度": "温和",
                "说明": f"成交量温和增加{volume_increase:.1f}%",
                "细节": {
                    "当前成交量": current_volume,
                    "5日均量": volume_ma5,
                    "10日均量": volume_ma10,
                    "连续放量": "是" if volume_increasing else "否",
                    "量价配合": "良好" if volume_price_match else "不佳"
                }
            }
        
        return {"状态": "缩量", "程度": "正常", "说明": "成交量处于正常水平"}

    def _check_volume_price_correlation(self, df: pd.DataFrame) -> Dict:
        """检查量价配合
        量价配合特征：
        1. 价格上涨时成交量增加
        2. 价格下跌时成交量减少
        3. 量价变动幅度匹配
        """
        if len(df) < 10:
            return {"状态": "无", "说明": "数据不足"}
        
        recent_data = df.tail(10)
        prices = recent_data['收盘'].values
        volumes = recent_data['成交量'].values
        
        # 计算价格和成交量的变化率
        price_changes = np.diff(prices) / prices[:-1]
        volume_changes = np.diff(volumes) / volumes[:-1]
        
        # 计算量价相关性
        correlation = np.corrcoef(price_changes, volume_changes)[0,1]
        
        # 计算量价匹配度
        match_count = sum((price_changes[i] > 0 and volume_changes[i] > 0) or \
                         (price_changes[i] < 0 and volume_changes[i] < 0) \
                         for i in range(len(price_changes)))
        match_ratio = match_count / len(price_changes)
        
        if correlation > 0.5 and match_ratio > 0.7:
            return {
                "状态": "配合",
                "说明": "量价配合良好",
                "细节": {
                    "量价相关性": f"{correlation:.2f}",
                    "量价匹配度": f"{match_ratio*100:.1f}%"
                }
            }
        elif correlation > 0.3 and match_ratio > 0.5:
            return {
                "状态": "部分配合",
                "说明": "量价部分配合",
                "细节": {
                    "量价相关性": f"{correlation:.2f}",
                    "量价匹配度": f"{match_ratio*100:.1f}%"
                }
            }
        
        return {"状态": "不配合", "说明": "量价配合不佳"}

    def _check_large_orders(self, df: pd.DataFrame) -> Dict:
        """检查大单交易
        大单交易特征：
        1. 大单成交占比高
        2. 大单买入/卖出方向一致
        3. 大单交易持续性
        """
        if len(df) < 5:
            return {"状态": "无", "说明": "数据不足"}
        
        recent_data = df.tail(5)
        
        # 计算大单交易指标
        # 注：由于数据限制，这里使用成交量和价格变动来间接判断大单交易
        volume_ma = recent_data['成交量'].mean()
        price_std = recent_data['收盘'].std()
        
        # 计算大单交易活跃度
        large_volume_days = sum(recent_data['成交量'] > volume_ma * 1.5)
        price_volatility = price_std / recent_data['收盘'].mean()
        
        if large_volume_days >= 3 and price_volatility > 0.02:
            return {
                "状态": "活跃",
                "说明": "大单交易活跃",
                "细节": {
                    "大单天数": large_volume_days,
                    "价格波动率": f"{price_volatility*100:.1f}%",
                    "成交量均值": volume_ma
                }
            }
        elif large_volume_days >= 2:
            return {
                "状态": "一般",
                "说明": "大单交易一般",
                "细节": {
                    "大单天数": large_volume_days,
                    "价格波动率": f"{price_volatility*100:.1f}%",
                    "成交量均值": volume_ma
                }
            }
        
        return {"状态": "不活跃", "说明": "大单交易不活跃"}

    def _check_turnover_rate(self, df: pd.DataFrame) -> Dict:
        """检查换手率变化
        换手率特征：
        1. 换手率水平
        2. 换手率变化趋势
        3. 换手率与价格变动关系
        """
        if len(df) < 10:
            return {"状态": "无", "说明": "数据不足"}
        
        recent_data = df.tail(10)
        turnover_rates = recent_data['换手率'].values
        prices = recent_data['收盘'].values
        
        # 计算换手率统计指标
        current_turnover = turnover_rates[-1]
        turnover_ma5 = np.mean(turnover_rates[-5:])
        turnover_ma10 = np.mean(turnover_rates)
        
        # 计算换手率变化趋势
        turnover_trend = np.polyfit(range(len(turnover_rates)), turnover_rates, 1)[0]
        
        # 计算换手率与价格的相关性
        price_changes = np.diff(prices) / prices[:-1]
        turnover_changes = np.diff(turnover_rates) / turnover_rates[:-1]
        correlation = np.corrcoef(price_changes, turnover_changes)[0,1]
        
        # 判断换手率状态
        if current_turnover > turnover_ma5 * 1.5 and turnover_trend > 0:
            return {
                "状态": "活跃",
                "说明": "换手率显著增加且呈上升趋势",
                "细节": {
                    "当前换手率": f"{current_turnover:.2f}%",
                    "5日均换手率": f"{turnover_ma5:.2f}%",
                    "10日均换手率": f"{turnover_ma10:.2f}%",
                    "换手率趋势": "上升" if turnover_trend > 0 else "下降",
                    "量价相关性": f"{correlation:.2f}"
                }
            }
        elif current_turnover > turnover_ma5 * 1.2:
            return {
                "状态": "温和",
                "说明": "换手率温和增加",
                "细节": {
                    "当前换手率": f"{current_turnover:.2f}%",
                    "5日均换手率": f"{turnover_ma5:.2f}%",
                    "10日均换手率": f"{turnover_ma10:.2f}%",
                    "换手率趋势": "上升" if turnover_trend > 0 else "下降",
                    "量价相关性": f"{correlation:.2f}"
                }
            }
        
        return {"状态": "正常", "说明": "换手率处于正常水平"}

    def _analyze_price_speed(self) -> Dict:
        """分析涨速
        涨速特征：
        1. 价格变动速度
        2. 加速度变化
        3. 与历史涨速对比
        """
        if self.data is None or len(self.data) < 20:
            return {"状态": "无", "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        prices = recent_data['收盘'].values
        
        # 计算价格变化速度
        price_changes = np.diff(prices) / prices[:-1]
        speed = np.mean(price_changes[-5:])  # 最近5天的平均速度
        speed_ma = np.mean(price_changes)  # 20天的平均速度
        
        # 计算加速度
        acceleration = np.diff(price_changes)
        recent_acceleration = np.mean(acceleration[-3:])  # 最近3天的平均加速度
        
        # 计算速度变化率
        speed_change = (speed / speed_ma - 1) * 100
        
        if speed > 0.02 and recent_acceleration > 0:
            return {
                "状态": "加速上涨",
                "说明": "价格加速上涨",
                "细节": {
                    "当前速度": f"{speed*100:.2f}%",
                    "平均速度": f"{speed_ma*100:.2f}%",
                    "速度变化": f"{speed_change:.1f}%",
                    "加速度": f"{recent_acceleration*100:.2f}%"
                }
            }
        elif speed > 0.01:
            return {
                "状态": "温和上涨",
                "说明": "价格温和上涨",
                "细节": {
                    "当前速度": f"{speed*100:.2f}%",
                    "平均速度": f"{speed_ma*100:.2f}%",
                    "速度变化": f"{speed_change:.1f}%",
                    "加速度": f"{recent_acceleration*100:.2f}%"
                }
            }
        elif speed < -0.02 and recent_acceleration < 0:
            return {
                "状态": "加速下跌",
                "说明": "价格加速下跌",
                "细节": {
                    "当前速度": f"{speed*100:.2f}%",
                    "平均速度": f"{speed_ma*100:.2f}%",
                    "速度变化": f"{speed_change:.1f}%",
                    "加速度": f"{recent_acceleration*100:.2f}%"
                }
            }
        elif speed < -0.01:
            return {
                "状态": "温和下跌",
                "说明": "价格温和下跌",
                "细节": {
                    "当前速度": f"{speed*100:.2f}%",
                    "平均速度": f"{speed_ma*100:.2f}%",
                    "速度变化": f"{speed_change:.1f}%",
                    "加速度": f"{recent_acceleration*100:.2f}%"
                }
            }
        
        return {"状态": "平稳", "说明": "价格变动平稳"}

    def _analyze_amplitude(self) -> Dict:
        """分析振幅
        振幅特征：
        1. 日内波动幅度
        2. 振幅变化趋势
        3. 与历史振幅对比
        """
        if self.data is None or len(self.data) < 20:
            return {"状态": "无", "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        
        # 计算振幅
        daily_amplitude = (recent_data['最高'] - recent_data['最低']) / recent_data['收盘']
        current_amplitude = daily_amplitude.iloc[-1]
        amplitude_ma5 = daily_amplitude.rolling(window=5).mean().iloc[-1]
        amplitude_ma20 = daily_amplitude.mean()
        
        # 计算振幅变化趋势
        amplitude_trend = np.polyfit(range(len(daily_amplitude)), daily_amplitude, 1)[0]
        
        # 计算振幅变化率
        amplitude_change = (current_amplitude / amplitude_ma20 - 1) * 100
        
        if current_amplitude > amplitude_ma20 * 1.5 and amplitude_trend > 0:
            return {
                "状态": "显著扩大",
                "说明": "振幅显著扩大且呈上升趋势",
                "细节": {
                    "当前振幅": f"{current_amplitude*100:.2f}%",
                    "5日均振幅": f"{amplitude_ma5*100:.2f}%",
                    "20日均振幅": f"{amplitude_ma20*100:.2f}%",
                    "振幅变化": f"{amplitude_change:.1f}%",
                    "振幅趋势": "上升" if amplitude_trend > 0 else "下降"
                }
            }
        elif current_amplitude > amplitude_ma20 * 1.2:
            return {
                "状态": "温和扩大",
                "说明": "振幅温和扩大",
                "细节": {
                    "当前振幅": f"{current_amplitude*100:.2f}%",
                    "5日均振幅": f"{amplitude_ma5*100:.2f}%",
                    "20日均振幅": f"{amplitude_ma20*100:.2f}%",
                    "振幅变化": f"{amplitude_change:.1f}%",
                    "振幅趋势": "上升" if amplitude_trend > 0 else "下降"
                }
            }
        elif current_amplitude < amplitude_ma20 * 0.8 and amplitude_trend < 0:
            return {
                "状态": "显著收窄",
                "说明": "振幅显著收窄且呈下降趋势",
                "细节": {
                    "当前振幅": f"{current_amplitude*100:.2f}%",
                    "5日均振幅": f"{amplitude_ma5*100:.2f}%",
                    "20日均振幅": f"{amplitude_ma20*100:.2f}%",
                    "振幅变化": f"{amplitude_change:.1f}%",
                    "振幅趋势": "上升" if amplitude_trend > 0 else "下降"
                }
            }
        
        return {"状态": "稳定", "说明": "振幅保持稳定"}

    def _analyze_market_attention(self) -> Dict:
        """分析市场关注度
        市场关注度特征：
        1. 成交量变化
        2. 换手率变化
        3. 价格波动性
        4. 涨跌幅
        """
        if self.data is None or len(self.data) < 20:
            return {"状态": "无", "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        
        # 计算成交量变化
        volume_ma5 = recent_data['成交量'].rolling(window=5).mean().iloc[-1]
        volume_ma20 = recent_data['成交量'].mean()
        volume_increase = (volume_ma5 / volume_ma20 - 1) * 100
        
        # 计算换手率变化
        turnover_ma5 = recent_data['换手率'].rolling(window=5).mean().iloc[-1]
        turnover_ma20 = recent_data['换手率'].mean()
        turnover_increase = (turnover_ma5 / turnover_ma20 - 1) * 100
        
        # 计算价格波动性
        price_std = recent_data['收盘'].std()
        price_volatility = price_std / recent_data['收盘'].mean()
        
        # 计算涨跌幅
        price_change = (recent_data['收盘'].iloc[-1] / recent_data['收盘'].iloc[0] - 1) * 100
        
        # 综合评分
        attention_score = 0
        if volume_increase > 30:
            attention_score += 2
        elif volume_increase > 10:
            attention_score += 1
        
        if turnover_increase > 30:
            attention_score += 2
        elif turnover_increase > 10:
            attention_score += 1
        
        if price_volatility > 0.02:
            attention_score += 1
        
        if abs(price_change) > 5:
            attention_score += 1
        
        if attention_score >= 5:
            return {
                "状态": "高度关注",
                "说明": "市场高度关注",
                "细节": {
                    "成交量变化": f"{volume_increase:.1f}%",
                    "换手率变化": f"{turnover_increase:.1f}%",
                    "价格波动性": f"{price_volatility*100:.2f}%",
                    "价格涨跌幅": f"{price_change:.1f}%",
                    "关注度得分": attention_score
                }
            }
        elif attention_score >= 3:
            return {
                "状态": "中度关注",
                "说明": "市场中度关注",
                "细节": {
                    "成交量变化": f"{volume_increase:.1f}%",
                    "换手率变化": f"{turnover_increase:.1f}%",
                    "价格波动性": f"{price_volatility*100:.2f}%",
                    "价格涨跌幅": f"{price_change:.1f}%",
                    "关注度得分": attention_score
                }
            }
        
        return {"状态": "低度关注", "说明": "市场关注度较低"}

    def _analyze_technical_risk(self) -> Dict:
        """分析技术风险
        技术风险特征：
        1. 趋势强度
        2. 支撑阻力位
        3. 技术指标背离
        4. 波动性
        """
        if self.data is None or len(self.data) < 20:
            return {"风险等级": "无", "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        
        # 计算趋势强度
        ma5 = recent_data['收盘'].rolling(window=5).mean()
        ma20 = recent_data['收盘'].rolling(window=20).mean()
        trend_strength = (ma5.iloc[-1] / ma20.iloc[-1] - 1) * 100
        
        # 计算支撑阻力位
        recent_highs = recent_data['最高'].values
        recent_lows = recent_data['最低'].values
        current_price = recent_data['收盘'].iloc[-1]
        
        resistance = np.max(recent_highs[:-1])
        support = np.min(recent_lows[:-1])
        
        # 计算与支撑阻力位的距离
        distance_to_resistance = (resistance / current_price - 1) * 100
        distance_to_support = (current_price / support - 1) * 100
        
        # 计算技术指标背离
        rsi = recent_data['RSI'].values
        price = recent_data['收盘'].values
        
        # 计算RSI背离
        rsi_divergence = 0
        if rsi[-1] > 70 and price[-1] < price[-2]:
            rsi_divergence = 1  # 顶背离
        elif rsi[-1] < 30 and price[-1] > price[-2]:
            rsi_divergence = -1  # 底背离
        
        # 计算波动性
        volatility = recent_data['收盘'].std() / recent_data['收盘'].mean()
        
        # 综合评分
        risk_score = 0
        
        # 趋势风险
        if abs(trend_strength) > 10:
            risk_score += 2
        elif abs(trend_strength) > 5:
            risk_score += 1
        
        # 支撑阻力位风险
        if distance_to_resistance < 5:
            risk_score += 2
        elif distance_to_resistance < 10:
            risk_score += 1
        
        if distance_to_support < 5:
            risk_score += 2
        elif distance_to_support < 10:
            risk_score += 1
        
        # 背离风险
        if rsi_divergence != 0:
            risk_score += 1
        
        # 波动性风险
        if volatility > 0.03:
            risk_score += 2
        elif volatility > 0.02:
            risk_score += 1
        
        if risk_score >= 6:
            return {
                "风险等级": "高",
                "说明": "技术面风险较高",
                "细节": {
                    "趋势强度": f"{trend_strength:.1f}%",
                    "距离阻力位": f"{distance_to_resistance:.1f}%",
                    "距离支撑位": f"{distance_to_support:.1f}%",
                    "RSI背离": "顶背离" if rsi_divergence == 1 else "底背离" if rsi_divergence == -1 else "无",
                    "波动性": f"{volatility*100:.2f}%",
                    "风险得分": risk_score
                }
            }
        elif risk_score >= 3:
            return {
                "风险等级": "中",
                "说明": "技术面风险中等",
                "细节": {
                    "趋势强度": f"{trend_strength:.1f}%",
                    "距离阻力位": f"{distance_to_resistance:.1f}%",
                    "距离支撑位": f"{distance_to_support:.1f}%",
                    "RSI背离": "顶背离" if rsi_divergence == 1 else "底背离" if rsi_divergence == -1 else "无",
                    "波动性": f"{volatility*100:.2f}%",
                    "风险得分": risk_score
                }
            }
        
        return {
            "风险等级": "低",
            "说明": "技术面风险较低",
            "细节": {
                "趋势强度": f"{trend_strength:.1f}%",
                "距离阻力位": f"{distance_to_resistance:.1f}%",
                "距离支撑位": f"{distance_to_support:.1f}%",
                "RSI背离": "顶背离" if rsi_divergence == 1 else "底背离" if rsi_divergence == -1 else "无",
                "波动性": f"{volatility*100:.2f}%",
                "风险得分": risk_score
            }
        }

    def _analyze_market_risk(self) -> Dict:
        """分析市场风险
        市场风险特征：
        1. 大盘走势
        2. 行业走势
        3. 市场情绪
        4. 资金面
        """
        if self.data is None or len(self.data) < 20:
            return {"风险等级": "无", "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        
        # 计算个股走势
        stock_return = (recent_data['收盘'].iloc[-1] / recent_data['收盘'].iloc[0] - 1) * 100
        
        # 计算波动性
        volatility = recent_data['收盘'].std() / recent_data['收盘'].mean()
        
        # 计算成交量变化
        volume_ma5 = recent_data['成交量'].rolling(window=5).mean().iloc[-1]
        volume_ma20 = recent_data['成交量'].mean()
        volume_change = (volume_ma5 / volume_ma20 - 1) * 100
        
        # 计算换手率变化
        turnover_ma5 = recent_data['换手率'].rolling(window=5).mean().iloc[-1]
        turnover_ma20 = recent_data['换手率'].mean()
        turnover_change = (turnover_ma5 / turnover_ma20 - 1) * 100
        
        # 综合评分
        risk_score = 0
        
        # 走势风险
        if abs(stock_return) > 10:
            risk_score += 2
        elif abs(stock_return) > 5:
            risk_score += 1
        
        # 波动性风险
        if volatility > 0.03:
            risk_score += 2
        elif volatility > 0.02:
            risk_score += 1
        
        # 成交量风险
        if volume_change < -30:
            risk_score += 2
        elif volume_change < -10:
            risk_score += 1
        
        # 换手率风险
        if turnover_change < -30:
            risk_score += 2
        elif turnover_change < -10:
            risk_score += 1
        
        if risk_score >= 6:
            return {
                "风险等级": "高",
                "说明": "市场风险较高",
                "细节": {
                    "个股涨跌幅": f"{stock_return:.1f}%",
                    "波动性": f"{volatility*100:.2f}%",
                    "成交量变化": f"{volume_change:.1f}%",
                    "换手率变化": f"{turnover_change:.1f}%",
                    "风险得分": risk_score
                }
            }
        elif risk_score >= 3:
            return {
                "风险等级": "中",
                "说明": "市场风险中等",
                "细节": {
                    "个股涨跌幅": f"{stock_return:.1f}%",
                    "波动性": f"{volatility*100:.2f}%",
                    "成交量变化": f"{volume_change:.1f}%",
                    "换手率变化": f"{turnover_change:.1f}%",
                    "风险得分": risk_score
                }
            }
        
        return {
            "风险等级": "低",
            "说明": "市场风险较低",
            "细节": {
                "个股涨跌幅": f"{stock_return:.1f}%",
                "波动性": f"{volatility*100:.2f}%",
                "成交量变化": f"{volume_change:.1f}%",
                "换手率变化": f"{turnover_change:.1f}%",
                "风险得分": risk_score
            }
        }

    def _analyze_operation_risk(self) -> Dict:
        """分析操作风险
        操作风险特征：
        1. 流动性
        2. 交易成本
        3. 滑点风险
        4. 操作难度
        """
        if self.data is None or len(self.data) < 20:
            return {"风险等级": "无", "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        
        # 计算流动性指标
        avg_volume = recent_data['成交量'].mean()
        avg_amount = recent_data['成交额'].mean()
        price = recent_data['收盘'].iloc[-1]
        
        # 计算交易成本
        turnover_rate = recent_data['换手率'].mean()
        
        # 计算价格波动性
        volatility = recent_data['收盘'].std() / recent_data['收盘'].mean()
        
        # 计算操作难度
        price_range = (recent_data['最高'].max() - recent_data['最低'].min()) / recent_data['最低'].min()
        
        # 综合评分
        risk_score = 0
        
        # 流动性风险
        if avg_volume < 1000000:  # 假设100万股为流动性阈值
            risk_score += 2
        elif avg_volume < 5000000:
            risk_score += 1
        
        # 交易成本风险
        if turnover_rate < 0.5:
            risk_score += 2
        elif turnover_rate < 1:
            risk_score += 1
        
        # 波动性风险
        if volatility > 0.03:
            risk_score += 2
        elif volatility > 0.02:
            risk_score += 1
        
        # 操作难度风险
        if price_range > 0.2:
            risk_score += 2
        elif price_range > 0.1:
            risk_score += 1
        
        if risk_score >= 6:
            return {
                "风险等级": "高",
                "说明": "操作风险较高",
                "细节": {
                    "平均成交量": f"{avg_volume:.0f}",
                    "平均成交额": f"{avg_amount:.0f}",
                    "换手率": f"{turnover_rate:.2f}%",
                    "波动性": f"{volatility*100:.2f}%",
                    "价格区间": f"{price_range*100:.1f}%",
                    "风险得分": risk_score
                }
            }
        elif risk_score >= 3:
            return {
                "风险等级": "中",
                "说明": "操作风险中等",
                "细节": {
                    "平均成交量": f"{avg_volume:.0f}",
                    "平均成交额": f"{avg_amount:.0f}",
                    "换手率": f"{turnover_rate:.2f}%",
                    "波动性": f"{volatility*100:.2f}%",
                    "价格区间": f"{price_range*100:.1f}%",
                    "风险得分": risk_score
                }
            }
        
        return {
            "风险等级": "低",
            "说明": "操作风险较低",
            "细节": {
                "平均成交量": f"{avg_volume:.0f}",
                "平均成交额": f"{avg_amount:.0f}",
                "换手率": f"{turnover_rate:.2f}%",
                "波动性": f"{volatility*100:.2f}%",
                "价格区间": f"{price_range*100:.1f}%",
                "风险得分": risk_score
            }
        }

    def _determine_trading_direction(self, price_pattern: Dict, volume_analysis: Dict,
                                   market_sentiment: Dict, risk_analysis: Dict,
                                   technical_indicators: Dict) -> str:
        """确定交易方向
        综合考虑：
        1. 价格形态
        2. 成交量分析
        3. 市场情绪
        4. 风险分析
        5. 技术指标
        """
        # 初始化得分
        buy_score = 0
        sell_score = 0
        
        # 价格形态得分
        if price_pattern.get('形态') in ['双底', '突破']:
            buy_score += 2
        elif price_pattern.get('形态') in ['头肩顶', '三角形']:
            sell_score += 2
        
        # 成交量分析得分
        if volume_analysis.get('状态') == '放量' and volume_analysis.get('程度') == '显著':
            if volume_analysis.get('量价配合') == '良好':
                buy_score += 1
            else:
                sell_score += 1
        
        # 市场情绪得分
        if market_sentiment.get('状态') == '高度关注':
            if market_sentiment.get('价格涨跌幅', 0) > 0:
                buy_score += 1
            else:
                sell_score += 1
        
        # 风险分析得分
        if risk_analysis.get('风险等级') == '低':
            if technical_indicators.get('trend', {}).get('ma_trend') == '上升':
                buy_score += 1
            else:
                sell_score += 1
        
        # 技术指标得分
        if technical_indicators.get('macd', {}).get('trend') == '上升':
            buy_score += 1
        elif technical_indicators.get('macd', {}).get('trend') == '下降':
            sell_score += 1
        
        if technical_indicators.get('rsi', {}).get('status') == '超卖':
            buy_score += 1
        elif technical_indicators.get('rsi', {}).get('status') == '超买':
            sell_score += 1
        
        # 判断交易方向
        if buy_score > sell_score:
            return "买入"
        elif sell_score > buy_score:
            return "卖出"
        else:
            return "观望"

    def _calculate_signal_strength(self, price_pattern: Dict, volume_analysis: Dict,
                                 market_sentiment: Dict, risk_analysis: Dict,
                                 technical_indicators: Dict) -> int:
        """计算信号强度
        信号强度评分标准：
        1. 价格形态可信度
        2. 成交量配合度
        3. 市场情绪强度
        4. 风险等级
        5. 技术指标一致性
        """
        strength = 0
        
        # 价格形态可信度
        if price_pattern.get('可信度', 0) > 0.8:
            strength += 2
        elif price_pattern.get('可信度', 0) > 0.6:
            strength += 1
        
        # 成交量配合度
        if volume_analysis.get('状态') == '配合':
            strength += 2
        elif volume_analysis.get('状态') == '部分配合':
            strength += 1
        
        # 市场情绪强度
        if market_sentiment.get('状态') == '高度关注':
            strength += 2
        elif market_sentiment.get('状态') == '中度关注':
            strength += 1
        
        # 风险等级
        if risk_analysis.get('风险等级') == '低':
            strength += 2
        elif risk_analysis.get('风险等级') == '中':
            strength += 1
        
        # 技术指标一致性
        technical_score = 0
        if technical_indicators.get('macd', {}).get('trend') == technical_indicators.get('trend', {}).get('ma_trend'):
            technical_score += 1
        if technical_indicators.get('rsi', {}).get('status') == '正常':
            technical_score += 1
        if technical_indicators.get('bollinger', {}).get('position') == '中轨':
            technical_score += 1
        
        strength += technical_score
        
        return min(strength, 5)  # 限制最大强度为5

    def _suggest_price_levels(self) -> Dict:
        """建议价格水平
        基于：
        1. 支撑阻力位
        2. 布林带
        3. 历史价格区间
        """
        if self.data is None or len(self.data) < 20:
            return {"买入价": 0, "卖出价": 0, "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        current_price = recent_data['收盘'].iloc[-1]
        
        # 计算支撑阻力位
        resistance = recent_data['最高'].max()
        support = recent_data['最低'].min()
        
        # 计算布林带
        bb_upper = recent_data['BB_Upper'].iloc[-1]
        bb_lower = recent_data['BB_Lower'].iloc[-1]
        
        # 计算历史价格区间
        price_range = (resistance - support) / support
        
        # 确定买入卖出价格
        if current_price > bb_upper:
            return {
                "买入价": current_price * 0.95,
                "卖出价": current_price * 1.05,
                "说明": "当前价格处于布林带上轨，建议回调买入"
            }
        elif current_price < bb_lower:
            return {
                "买入价": current_price * 1.05,
                "卖出价": current_price * 1.15,
                "说明": "当前价格处于布林带下轨，建议反弹买入"
            }
        else:
            return {
                "买入价": current_price * 0.98,
                "卖出价": current_price * 1.08,
                "说明": "当前价格处于布林带中轨，建议区间操作"
            }

    def _calculate_stop_loss(self) -> Dict:
        """计算止损位
        基于：
        1. 支撑位
        2. 布林带
        3. 波动率
        """
        if self.data is None or len(self.data) < 20:
            return {"价格": 0, "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        current_price = recent_data['收盘'].iloc[-1]
        
        # 计算支撑位
        support = recent_data['最低'].min()
        
        # 计算布林带下轨
        bb_lower = recent_data['BB_Lower'].iloc[-1]
        
        # 计算波动率
        volatility = recent_data['收盘'].std() / recent_data['收盘'].mean()
        
        # 确定止损位
        stop_loss = min(support, bb_lower)
        stop_loss = max(stop_loss, current_price * (1 - volatility * 2))
        
        return {
            "价格": stop_loss,
            "说明": f"建议止损位设在{stop_loss:.2f}元，止损幅度{(stop_loss/current_price-1)*100:.1f}%"
        }

    def _calculate_take_profit(self) -> Dict:
        """计算止盈位
        基于：
        1. 阻力位
        2. 布林带
        3. 波动率
        """
        if self.data is None or len(self.data) < 20:
            return {"价格": 0, "说明": "数据不足"}
        
        recent_data = self.data.tail(20)
        current_price = recent_data['收盘'].iloc[-1]
        
        # 计算阻力位
        resistance = recent_data['最高'].max()
        
        # 计算布林带上轨
        bb_upper = recent_data['BB_Upper'].iloc[-1]
        
        # 计算波动率
        volatility = recent_data['收盘'].std() / recent_data['收盘'].mean()
        
        # 确定止盈位
        take_profit = max(resistance, bb_upper)
        take_profit = min(take_profit, current_price * (1 + volatility * 3))
        
        return {
            "价格": take_profit,
            "说明": f"建议止盈位设在{take_profit:.2f}元，止盈幅度{(take_profit/current_price-1)*100:.1f}%"
        }

    def _suggest_position_size(self) -> Dict:
        """建议仓位
        基于：
        1. 风险等级
        2. 信号强度
        3. 市场环境
        """
        # 获取风险分析结果
        risk_analysis = self.analyze_risk()
        technical_risk = risk_analysis.get('技术风险', {})
        market_risk = risk_analysis.get('市场风险', {})
        operation_risk = risk_analysis.get('操作风险', {})
        
        # 获取信号强度
        signal_strength = self._calculate_signal_strength(
            self.analyze_price_pattern(),
            self.analyze_volume(),
            self.analyze_market_sentiment(),
            risk_analysis,
            self.calculate_indicators()
        )
        
        # 计算风险得分
        risk_score = 0
        for risk in [technical_risk, market_risk, operation_risk]:
            if risk.get('风险等级') == '高':
                risk_score += 2
            elif risk.get('风险等级') == '中':
                risk_score += 1
        
        # 确定仓位
        if risk_score >= 4:
            position = "30%"
            explanation = "风险较高，建议轻仓"
        elif risk_score >= 2:
            if signal_strength >= 4:
                position = "70%"
                explanation = "风险中等，信号较强，可以重仓"
            else:
                position = "50%"
                explanation = "风险中等，建议半仓"
        else:
            if signal_strength >= 4:
                position = "90%"
                explanation = "风险较低，信号较强，可以满仓"
            else:
                position = "70%"
                explanation = "风险较低，建议重仓"
        
        return {
            "仓位": position,
            "说明": explanation,
            "细节": {
                "风险得分": risk_score,
                "信号强度": signal_strength
            }
        }

def analyze_stock(stock_code: str) -> Dict:
    """
    分析股票并生成交易建议
    :param stock_code: 股票代码，例如：sh600036
    :return: 分析结果字典
    """
    try:
        analyzer = ComprehensiveStockAnalyzer(stock_code)
        analysis_result = analyzer.generate_trading_signal()
        
        # 添加时间戳
        analysis_result["分析时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analysis_result["股票代码"] = stock_code
        
        return analysis_result
    except Exception as e:
        print(f"分析股票失败: {str(e)}")
        return {"error": str(e)}

def main():
    # 创建分析器实例
    analyzer = ComprehensiveStockAnalyzer("sh510300")
    
    # 获取历史数据
    print("正在获取历史数据...")
    df = analyzer.fetch_data("20230101", "20240321")
    if df is None:
        print("获取数据失败")
        return
        
    # 获取分析结果
    print("正在计算技术指标...")
    indicators = analyzer.calculate_indicators()
    
    print("正在分析市场情绪...")
    sentiment = analyzer.analyze_market_sentiment()
    
    print("正在分析风险...")
    risk = analyzer.analyze_risk()
    
    print("正在生成交易信号...")
    signal = analyzer.generate_trading_signal()
    
    # 自定义JSON编码器
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # 使用自定义编码器输出结果
    print("\n=== 技术指标 ===")
    print(json.dumps(indicators, indent=2, ensure_ascii=False, cls=NumpyEncoder))
    
    print("\n=== 市场情绪分析 ===")
    print(json.dumps(sentiment, indent=2, ensure_ascii=False, cls=NumpyEncoder))
    
    print("\n=== 风险分析 ===")
    print(json.dumps(risk, indent=2, ensure_ascii=False, cls=NumpyEncoder))
    
    print("\n=== 交易信号 ===")
    print(json.dumps(signal, indent=2, ensure_ascii=False, cls=NumpyEncoder))

if __name__ == "__main__":
    main() 