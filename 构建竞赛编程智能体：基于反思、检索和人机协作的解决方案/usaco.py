import os

os.environ["OPENAI_API_KEY"] = "sk-qX4v4ERHcUAgZ48D6761Ae1a91194bB9BdD0561b502e6688"
os.environ["OPENAI_API_BASE"] = "https://openkey.cloud/v1"

import zipfile

import datasets
import requests

usaco_url = "https://storage.googleapis.com/benchmarks-artifacts/usaco/usaco_sampled_with_tests.zip"
zip_path = "usaco.zip"
extract_path = "usaco_datasets"

response = requests.get(usaco_url)
with open(zip_path, "wb") as file:
    file.write(response.content)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

os.remove(zip_path)

ds = datasets.load_from_disk(os.path.join(extract_path, "usaco_v3_sampled_with_tests"))




import multiprocessing
import queue
import subprocess
import sys
import time
import traceback
import sys, multiprocessing

# if sys.platform == "win32":
multiprocessing.set_start_method("spawn", force=True)
# else:
#     multiprocessing.set_start_method("fork", force=True)

def exec_program(q, program, input_data, expected_output, timeout):
    """执行程序并返回结果"""
    try:
        start_time = time.time()
        process = subprocess.Popen(
            [sys.executable, "-c", program],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
        if time.time() - start_time > timeout:
            raise TimeoutError("Execution timed out.")
        if process.returncode != 0:
            q.put(f"failed: {stderr}")
        else:
            if stdout.strip() == expected_output.strip():
                q.put("passed")
            else:
                q.put(f"wrong answer. Expected '{expected_output}', got '{stdout}'")
    except subprocess.TimeoutExpired:
        process.kill()
        q.put("timed out")
    except Exception:
        q.put(f"failed: {traceback.format_exc()}")
def check_correctness(
    program: str, input_data: str, expected_output: str, timeout: float
) -> str:
    """检查程序正确性"""
    q = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=exec_program, args=(q, program, input_data, expected_output, timeout)
    )
    process.start()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.terminate()
        process.join()
        result = "timed out"
    else:
        try:
            result = q.get_nowait()
        except queue.Empty:
            result = "no result returned"
    return result

program_code = "print('hello, world!')"
input_data = ""
expected_output = "hello, world!"
timeout = 2

test_result = check_correctness(program_code, input_data, expected_output, timeout)
print("Example 1: ", test_result)
test_result = check_correctness("print('goodbye')", input_data, "hi there", timeout)
print("Example 2: ", test_result)

exit


from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages



from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
class TestCase(TypedDict):
    inputs: str
    outputs: str
class State(TypedDict):
    # 只追加的聊天记忆，以便智能体可以尝试从初始错误中恢复
    messages: Annotated[list[AnyMessage], add_messages]
    # 来自数据集，用于测试
    test_cases: list[TestCase]
    runtime_limit: int
    status: str


input_states = [
    {
        "messages": [("user", row["description"])],
        "test_cases": row["test_cases"],
        "runtime_limit": row["runtime_limit"],
        "status": "in_progress",
        "problem_level": row["problem_level"],
    }
    for row in ds
]

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field


class writePython(BaseModel):
    """Write python code that resolves the problem."""

    reasoning: str = Field(..., description="Conceptual solution.")
    pseudocode: str = Field(..., description="Detailed English pseudocode.")
    code: str = Field(..., description="Valid Python 3 solution to the problem")


class Solver:
    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        self.runnable = prompt | llm.bind_tools([writePython])

    def __call__(self, state: State) -> dict:
        # Our agent only can see the "messages" and will ignore the test info
        return {"messages": [self.runnable.invoke({"messages": state["messages"]})]}

from langchain import hub
from langchain_anthropic import ChatAnthropic

# For this section, we are testing zero-shot performance and won't have
# any examples. Partial them out to pre-fill the template.
prompt = hub.pull("wfh/usaco-draft-solver").partial(examples="")
print("*" * 35 + "Prompt" + "*" * 35)
prompt.pretty_print()

# Use Haiku if you want to save $$ while (almost) never correctly answering the question
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-opus-20240229")

solver = Solver(llm, prompt)

print("*" * 34 + " Example " + "*" * 34)
result = solver(
    {
        "messages": [
            (
                "user",
                "How do I get a perfectly random sample from an infinite stream",
            )
        ]
    }
)
result["messages"][0].pretty_print()


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
def format_tool_message(response: str, ai_message: AIMessage):
    """格式化工具消息"""
    return ToolMessage(
        content=response + "\n使用writePython工具进行所有修复。",
        tool_call_id=ai_message.tool_calls[0]["id"],
    )
def evaluate(state: State):
    """评估提交的代码"""
    test_cases = state["test_cases"]
    ai_message: AIMessage = state["messages"][-1]
    if not ai_message.tool_calls:
        return {
            "messages": [
                HumanMessage(
                    content="未提交代码。请使用正确的Python代码再试一次。"
                )
            ]
        }
    try:
        code = ai_message.tool_calls[0]["args"]["code"]
    except Exception as e:
        return {"messages": [format_tool_message(repr(e), ai_message)]}
    num_test_cases = len(test_cases)
    succeeded = 0
    test_results = []
    # 运行所有测试用例
    for test_case in test_cases:
        input_data = test_case["inputs"]
        expected_output = test_case["outputs"]
        test_result = check_correctness(code, input_data, expected_output, 2)  # 2秒超时
        test_results.append(test_result)
        if test_result == "passed":
            succeeded += 1
    pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
    if pass_rate == 1:
        return {"status": "success"}
    # 格式化失败结果
    responses = "\n".join(
        [f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)]
    )
    response = f"提交错误。请响应更新的代码。\n通过率: {succeeded}/{num_test_cases}\n结果:\n{responses}"
    formatted_message = format_tool_message(response, ai_message)
    return {"messages": [formatted_message]}

from langgraph.graph import END, StateGraph, START
builder = StateGraph(State)
builder.add_node("solver", solver)
builder.add_edge(START, "solver")
builder.add_node("evaluate", evaluate)
builder.add_edge("solver", "evaluate")
def control_edge(state: State):
    """控制边：决定是结束还是继续求解"""
    if state.get("status") == "success":
        return END
    return "solver"
builder.add_conditional_edges("evaluate", control_edge, {END: END, "solver": "solver"})

graph = builder.compile()

input_state = input_states[0].copy()
# We will reduce the test cases to speed this notebook up
input_state["test_cases"] = input_state["test_cases"][:3]
print(input_state["messages"][0][1])


class State(TypedDict):
    # 新增！用于检索的候选和格式化获取的示例作为"记忆"
    candidate: AIMessage
    examples: str
    # 从第一部分重复的字段
    messages: Annotated[list[AnyMessage], add_messages]
    test_cases: list[TestCase]
    runtime_limit: int
    status: str


    
class Solver:
    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        self.runnable = prompt | llm.bind_tools([writePython])
    def __call__(self, state: State) -> dict:
        inputs = {"messages": state["messages"]}
        has_examples = bool(state.get("examples"))
        output_key = "candidate"  # 在草稿节点中使用
        if has_examples:
            output_key = "messages"  # 在求解节点中使用
            inputs["examples"] = state["examples"]
        response = self.runnable.invoke(inputs)
        if not response.content:
            return {
                output_key: AIMessage(
                    content="我需要逐步思考这个问题。"
                )
            }
        return {output_key: response}
# 创建实例
prompt = hub.pull("wfh/usaco-draft-solver")
llm = ChatAnthropic(model="claude-3-opus-20240229")
draft_solver = Solver(llm, prompt.partial(examples=""))
solver = Solver(llm, prompt)


# 准备训练和测试数据
test_indices = [0, 2]
train_ds = [row for i, row in enumerate(ds) if i not in test_indices]
test_ds = [row for i, row in enumerate(ds) if i in test_indices]
from langchain_community.retrievers import BM25Retriever
def format_example(row):
    """格式化示例为检索格式"""
    question = row["description"]
    answer = row["solution"]
    return f"""<problem>
{question}
</problem>
<solution>
{answer}
</solution>"""
# 跳过我们的"测试示例"以避免作弊
# 这是"模拟"已经看过其他上下文示例
retriever = BM25Retriever.from_texts([format_example(row) for row in train_ds])
def retrieve_examples(state: State, config: RunnableConfig):
    """检索相似示例"""
    top_k = config["configurable"].get("k") or 2
    ai_message: AIMessage = state["candidate"]
    if not ai_message.tool_calls:
        raise ValueError("草稿智能体没有产生有效的代码块")
    code = ai_message.tool_calls[0]["args"]["code"]
    examples_str = "\n".join(
        [doc.page_content for doc in retriever.invoke(code)[:top_k]]
    )
    examples_str = f"""
您之前在这个竞赛中解决了以下问题：
<Examples>
{examples_str}
</Examples>
以类似的成熟度处理这个新问题。"""
    return {"examples": examples_str}

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START

builder = StateGraph(State)
builder.add_node("draft", draft_solver)
builder.add_edge(START, "draft")
builder.add_node("retrieve", retrieve_examples)
builder.add_node("solve", solver)
builder.add_node("evaluate", evaluate)
# Add connectivity
builder.add_edge("draft", "retrieve")
builder.add_edge("retrieve", "solve")
builder.add_edge("solve", "evaluate")


def control_edge(state: State):
    if state.get("status") == "success":
        return END
    return "solve"


builder.add_conditional_edges("evaluate", control_edge, {END: END, "solve": "solve"})


checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client
def _hide_test_cases(inputs):
    """隐藏测试用例以减少跟踪大小"""
    copied = inputs.copy()
    copied["test_cases"] = "..."
    return copied
client = Client(hide_inputs=_hide_test_cases, hide_outputs=_hide_test_cases)
config = {"configurable": {"thread_id": "question-recall", "k": 3}}
with tracing_v2_enabled(client=client):
    events = graph.stream(input_state, config)
    for event in events:
        for value in event.values():
            messages = value.get("messages")
            if messages:
                if isinstance(messages, list):
                    messages = value["messages"][-1]
                print("Assistant:", str(messages.content).replace("\n", "\\n")[:50])
            elif value.get("examples"):
                print("检索到的示例:\n\n", value["examples"][:100] + "...")
            elif value.get("candidate"):
                print(str(value["candidate"].content)[:200])

                
checkpoint = graph.get_state(config)
print(checkpoint.values["status"]) 



from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
builder = StateGraph(State)
# 创建智能体组件
prompt = hub.pull("wfh/usaco-draft-solver")
llm = ChatAnthropic(model="claude-3-opus-20240229", max_tokens_to_sample=4000)
draft_solver = Solver(llm, prompt.partial(examples=""))
# 添加节点和边
builder.add_node("draft", draft_solver)
builder.add_edge(START, "draft")
builder.add_node("retrieve", retrieve_examples)
solver = Solver(llm, prompt)
builder.add_node("solve", solver)
builder.add_node("evaluate", evaluate)
builder.add_edge("draft", "retrieve")
builder.add_edge("retrieve", "solve")
builder.add_edge("solve", "evaluate")
def control_edge(state: State):
    if state.get("status") == "success":
        return END
    return "solve"
builder.add_conditional_edges("evaluate", control_edge, {END: END, "solve": "solve"})
checkpointer = InMemorySaver()
# 关键：设置interrupt_after=["evaluate"]指示智能体在继续执行前等待人工输入
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["evaluate"],  # 新增：告诉图在"human"节点处中断
)


silver_row = test_ds[1]
print(silver_row["problem_level"])  # 'silver'
silver_input = {
    "messages": [("user", silver_row["description"])],
    "test_cases": silver_row["test_cases"],
    "runtime_limit": silver_row["runtime_limit"],
    "status": "in_progress",
}
config = {"configurable": {"thread_id": "silver-hl-1", "k": 2}}
with tracing_v2_enabled(client=client):
    events = graph.stream(silver_input, config)
    for event in events:
        for value in event.values():
            messages = value.get("messages")
            if messages:
                if isinstance(messages, list):
                    messages = value["messages"][-1]
                print("Assistant:", str(messages.content).replace("\n", "\\n")[:50])
            elif value.get("examples"):
                print("检索到的示例:\n\n", value["examples"][:100] + "...")
            elif value.get("candidate"):
                print(str(value["candidate"].content)[:200])

# 检查当前状态
snapshot = graph.get_state(config)
print("当前问题描述:")
print(snapshot.values["messages"][0].content)
print("\n\n智能体当前的代码:")
ai_message = snapshot.values["messages"][-2]
if ai_message.tool_calls:
    print(ai_message.tool_calls[0]["args"]["code"])
print("\n\n测试结果:")
print(snapshot.values["messages"][-1].content[:200])


num_trials = 2
with tracing_v2_enabled(client=client):
    for _ in range(num_trials):
        events = graph.stream(None, updated_config)
        for event in events:
            for value in event.values():
                messages = value.get("messages")
                if messages:
                    if isinstance(messages, list):
                        messages = value["messages"][-1]
                    print("Assistant:", str(messages.content).replace("\n", "\\n")[:50])
                elif value.get("examples"):
                    print("检索到的示例:\n\n", value["examples"][:100] + "...")
                elif value.get("candidate"):
                    print(str(value["candidate"].content)[:200])
        if graph.get_state(config).values["status"] == "success":
            break
        print("继续...")