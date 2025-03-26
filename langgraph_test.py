from typing import Annotated
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# 为Tavily搜索添加API密钥
tool = TavilySearchResults(max_results=2, tavily_api_key="您的Tavily API密钥")
tools = [tool]
# 创建DeepSeek实例
llm = ChatDeepSeek(
    model="deepseek-chat",  # 可选模型：deepseek-chat（通用对话）或deepseek-reasoner（复杂推理）
    temperature=0,          # 控制输出随机性（0为确定性输出，1为高创造性）
    max_tokens=None,        # 不限制生成长度
    timeout=30,             # API超时时间（秒）
    max_retries=2,           # API调用失败重试次数
    api_key="sk-aacb49dcd0654de78c2b0d694296d5d1")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

# 添加一个示例运行代码
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    messages = [HumanMessage(content="什么是人工智能？")]
    result = graph.invoke({"messages": messages})
    
    # 打印结果
    for message in result["messages"]:
        if hasattr(message, "content") and message.content:
            print(f"{message.type}: {message.content}")







