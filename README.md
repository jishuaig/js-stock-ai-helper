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

## 环境配置

在使用之前，需要设置 DeepSeek API 密钥作为环境变量。以下是不同操作系统的设置方法：

### macOS 设置方法

1. 临时设置（仅当前终端会话有效）：
```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```

2. 永久设置（推荐）：
```bash
# 编辑 ~/.zshrc 文件
echo 'export DEEPSEEK_API_KEY="your-api-key-here"' >> ~/.zshrc

# 使配置立即生效
source ~/.zshrc
```

### Linux 设置方法

1. 临时设置：
```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```

2. 永久设置：
```bash
# 编辑 ~/.bashrc 或 ~/.zshrc 文件
echo 'export DEEPSEEK_API_KEY="your-api-key-here"' >> ~/.bashrc
# 或
echo 'export DEEPSEEK_API_KEY="your-api-key-here"' >> ~/.zshrc

# 使配置立即生效
source ~/.bashrc
# 或
source ~/.zshrc
```

### Windows 设置方法

1. 临时设置（CMD）：
```cmd
set DEEPSEEK_API_KEY=your-api-key-here
```

2. 临时设置（PowerShell）：
```powershell
$env:DEEPSEEK_API_KEY='your-api-key-here'
```

3. 永久设置：
- 右键"此电脑" -> "属性" -> "高级系统设置" -> "环境变量"
- 在"用户变量"区域点击"新建"
- 变量名输入：`DEEPSEEK_API_KEY`
- 变量值输入：您的 API 密钥

### 验证环境变量

设置完成后，可以通过以下命令验证环境变量是否设置成功：

```bash
# Linux/macOS
echo $DEEPSEEK_API_KEY

# Windows (CMD)
echo %DEEPSEEK_API_KEY%

# Windows (PowerShell)
echo $env:DEEPSEEK_API_KEY
```

## 使用方法

### 单只股票追踪

```bash
python stock_tracker_single.py --stock_code sh600036 --interval 60
```

参数说明：
- `--stock_code`: 股票代码（例如：sh600036）
- `--interval`: 追踪周期（秒）
- `--verbose`: 是否显示详细信息
- `--history_days`: 历史数据天数

### 多只股票追踪

```bash
python stock_tracker_multi.py --stock_list sh600036,sz000001 --interval 60
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

1. 使用前请确保已正确设置 DEEPSEEK_API_KEY 环境变量
2. 建议在实盘交易前进行充分的回测
3. 该工具仅供参考，不构成投资建议
4. 请遵守相关法律法规和交易所规则
5. 如果遇到环境变量未设置的错误，请按照上述步骤重新设置环境变量

## 许可证

MIT License
