from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import os

# 从环境变量获取 API 密钥
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

# 创建DeepSeek实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 可选模型：deepseek-chat（通用对话）或deepseek-reasoner（复杂推理）
    temperature=0,          # 控制输出随机性（0为确定性输出，1为高创造性）
    max_tokens=None,        # 不限制生成长度
    timeout=30,             # API超时时间（秒）
    max_retries=2,           # API调用失败重试次数
    api_key=api_key
)

@tool
def get_now_weather(city: str) -> str:
    """获取指定城市的实时天气信息，参数为城市中文名称"""
    if "北京" in city:
        return "晴，25℃"
    return "多云，23℃"

@tool
def calculate_expression(expr: str) -> float:
    """计算数学表达式结果，支持加减乘除"""
    return eval(expr)

tools = [get_now_weather, calculate_expression]
model_with_tools = model.bind_tools(tools)  # 关键绑定操作

# 首次调用获取工具指令
messages = [HumanMessage(content="北京今天适合穿什么衣服？")]
ai_message = model_with_tools.invoke(messages)
print(f"工具调用指令：{ai_message.tool_calls}")

# 执行工具调用
# 直接将原始AI消息添加到消息列表中，不需要重新创建AIMessage
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    tool_name = tool_call["name"]
    if tool_name == "get_now_weather":
        tool_func = get_now_weather
    elif tool_name == "calculate_expression":
        tool_func = calculate_expression
    else:
        continue
    result = tool_func.invoke(tool_call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

# 最终响应生成
response = model_with_tools.invoke(messages)
print(f"最终答案：{response.content}")



