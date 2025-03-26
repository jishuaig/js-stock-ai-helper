# 股票 AI 分析助手

这是一个基于 Python 的股票分析工具，结合了技术分析和 AI 分析能力，为投资者提供实时的股票分析和交易建议。

## 功能特点

- 实时股票数据追踪
- 多维度技术指标分析
- AI 驱动的交易信号生成
- 支持单只股票和多只股票同时分析
- 历史数据分析
- 自定义追踪周期

## 主要组件

- `stock_tracker_single.py`: 单只股票实时追踪和分析
- `stock_tracker_multi.py`: 多只股票同时追踪和分析
- `stock_analysis.py`: 股票数据分析模块
- `stock_func.py`: 核心功能函数库

## 技术指标

系统分析以下技术指标：
- 移动平均线 (MA)
- 相对强弱指标 (RSI)
- 移动平均收敛散度 (MACD)
- 随机指标 (KDJ)
- 布林带 (Bollinger Bands)

## 安装要求

```bash
pip install akshare pandas tabulate langchain-deepseek
```

## 使用方法

### 单只股票追踪

```bash
python stock_tracker_single.py --stock_code sh600036 --interval 60 --api_key YOUR_API_KEY
```

参数说明：
- `--stock_code`: 股票代码（例如：sh600036）
- `--interval`: 追踪周期（秒）
- `--api_key`: DeepSeek API 密钥
- `--verbose`: 是否显示详细信息
- `--history_days`: 历史数据天数

### 多只股票追踪

```bash
python stock_tracker_multi.py --stock_list sh600036,sz000001 --interval 60 --api_key YOUR_API_KEY
```

## AI 分析输出

系统会生成包含以下信息的分析报告：
- 交易信号（买入/卖出/持有）
- 信号强度（1-5）
- 分析理由
- 风险因素
- 技术分析详情
- 价格目标

## 注意事项

1. 使用前请确保已获取 DeepSeek API 密钥
2. 建议在实盘交易前进行充分的回测
3. 该工具仅供参考，不构成投资建议
4. 请遵守相关法律法规和交易所规则

## 许可证

MIT License
