import os

os.environ["OPENAI_API_KEY"] = "sk-qX4v4ERHcUAgZ48D6761Ae1a91194bB9BdD0561b502e6688"
os.environ["OPENAI_API_BASE"] = "https://openkey.cloud/v1"
os.environ["TAVILY_API_KEY"] = "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"

import os
import shutil
import sqlite3
import pandas as pd
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
# 备份文件让我们可以在每个教程部分重新开始
backup_file = "travel2.backup.sqlite"
overwrite = False

if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # 确保请求成功
    with open(local_file, "wb") as f:
        f.write(response.content)
    # 备份 - 我们将使用这个来"重置"每个部分的数据库
    shutil.copy(local_file, backup_file)

# 将航班转换为当前时间用于教程
def update_dates(file):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    return file

db = update_dates(local_file)

from sentence_transformers import SentenceTransformer
import re
import numpy as np
import openai
from langchain_core.tools import tool
response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text
docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, embed_model):
        self._arr = np.array(vectors)
        self._docs = docs
        self._model = embed_model

    @classmethod
    def from_docs(cls, docs, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """从文档构建本地向量存储"""
        model = SentenceTransformer(model_name)
        texts = [doc["page_content"] for doc in docs]
        vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return cls(docs, vectors, model)

    def query(self, query: str, k: int = 5) -> list[dict]:
        """语义检索"""
        query_vec = self._model.encode([query], normalize_embeddings=True)[0]
        # 点积（余弦相似度） = 向量相乘
        scores = query_vec @ self._arr.T

        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]

        return [
            {**self._docs[idx], "similarity": float(scores[idx])}
            for idx in top_k_idx_sorted
        ]

retriever = VectorStoreRetriever.from_docs(docs, model_name="BAAI/bge-base-zh")


@tool
def lookup_policy(query: str) -> str:
    """查询公司政策以检查是否允许某些选项。
    在进行任何航班更改或执行其他'写入'事件之前使用此工具。"""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])


import sqlite3
from datetime import date, datetime
from typing import Optional
import pytz
from langchain_core.runnables import RunnableConfig
from datetime import date, datetime
from typing import Optional, Union

@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """获取用户的所有机票以及相应的航班信息和座位分配。

    返回：
        包含票务详情、相关航班详情和属于用户的每张票的座位分配的字典列表。
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客ID。")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, 
        f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return results

@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """根据出发机场、到达机场和出发时间范围搜索航班。"""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)

    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return results

@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """将用户的机票更新为新的有效航班。"""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客ID。")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "提供的新航班ID无效。"
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"不允许改签到距离当前时间不足3小时的航班。所选航班时间为 {departure_time}。"

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "未找到给定票号的现有机票。"

    # 检查登录用户是否确实拥有此机票
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"当前登录乘客ID {passenger_id} 不是机票 {ticket_no} 的所有者"

    # 在实际应用中，您可能会在这里添加额外的检查来执行业务逻辑，
    # 比如"新的出发机场是否与当前机票匹配"等等。
    # 虽然最好尝试向LLM主动"类型提示"政策
    # 但它不可避免地会出错，所以您**也**需要确保您的
    # API强制执行有效行为
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "机票已成功更新为新航班。"

@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """取消用户的机票并从数据库中删除。"""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("未配置乘客ID。")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "未找到给定票号的现有机票。"

    # 检查登录用户是否确实拥有此机票
    cursor.execute(
        "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"当前登录乘客ID {passenger_id} 不是机票 {ticket_no} 的所有者"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "机票已成功取消。"


@tool
def search_car_rentals(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    根据位置、名称、价格等级、开始日期和结束日期搜索汽车租赁。

    参数：
        location (Optional[str]): 汽车租赁的位置。默认为None。
        name (Optional[str]): 汽车租赁公司的名称。默认为None。
        price_tier (Optional[str]): 汽车租赁的价格等级。默认为None。
        start_date (Optional[Union[datetime, date]]): 汽车租赁的开始日期。默认为None。
        end_date (Optional[Union[datetime, date]]): 汽车租赁的结束日期。默认为None。

    返回：
        list[dict]: 符合搜索条件的汽车租赁字典列表。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM car_rentals WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # 对于我们的教程，我们将允许您匹配任何日期和价格等级。
    # （因为我们的玩具数据集没有太多数据）
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

@tool
def book_car_rental(rental_id: int) -> str:
    """
    通过ID预订汽车租赁。

    参数：
        rental_id (int): 要预订的汽车租赁的ID。

    返回：
        str: 指示汽车租赁是否成功预订的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"汽车租赁 {rental_id} 已成功预订。"
    else:
        conn.close()
        return f"未找到ID为 {rental_id} 的汽车租赁。"

@tool
def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    通过ID更新汽车租赁的开始和结束日期。

    参数：
        rental_id (int): 要更新的汽车租赁的ID。
        start_date (Optional[Union[datetime, date]]): 汽车租赁的新开始日期。默认为None。
        end_date (Optional[Union[datetime, date]]): 汽车租赁的新结束日期。默认为None。

    返回：
        str: 指示汽车租赁是否成功更新的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date, rental_id),
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id)
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"汽车租赁 {rental_id} 已成功更新。"
    else:
        conn.close()
        return f"未找到ID为 {rental_id} 的汽车租赁。"

@tool
def cancel_car_rental(rental_id: int) -> str:
    """
    通过ID取消汽车租赁。

    参数：
        rental_id (int): 要取消的汽车租赁的ID。

    返回：
        str: 指示汽车租赁是否成功取消的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"汽车租赁 {rental_id} 已成功取消。"
    else:
        conn.close()
        return f"未找到ID为 {rental_id} 的汽车租赁。"
    
@tool
def search_hotels(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    根据位置、名称、价格等级、入住日期和退房日期搜索酒店。

    参数：
        location (Optional[str]): 酒店的位置。默认为None。
        name (Optional[str]): 酒店的名称。默认为None。
        price_tier (Optional[str]): 酒店的价格等级。默认为None。示例：中档、高档中档、高端、豪华
        checkin_date (Optional[Union[datetime, date]]): 酒店的入住日期。默认为None。
        checkout_date (Optional[Union[datetime, date]]): 酒店的退房日期。默认为None。

    返回：
        list[dict]: 符合搜索条件的酒店字典列表。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM hotels WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # 为了本教程的目的，我们将允许您匹配任何日期和价格等级。
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

@tool
def book_hotel(hotel_id: int) -> str:
    """
    通过ID预订酒店。

    参数：
        hotel_id (int): 要预订的酒店的ID。

    返回：
        str: 指示酒店是否成功预订的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店 {hotel_id} 已成功预订。"
    else:
        conn.close()
        return f"未找到ID为 {hotel_id} 的酒店。"

@tool
def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    通过ID更新酒店的入住和退房日期。

    参数：
        hotel_id (int): 要更新的酒店的ID。
        checkin_date (Optional[Union[datetime, date]]): 酒店的新入住日期。默认为None。
        checkout_date (Optional[Union[datetime, date]]): 酒店的新退房日期。默认为None。

    返回：
        str: 指示酒店是否成功更新的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
        )
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date, hotel_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店 {hotel_id} 已成功更新。"
    else:
        conn.close()
        return f"未找到ID为 {hotel_id} 的酒店。"

@tool
def cancel_hotel(hotel_id: int) -> str:
    """
    通过ID取消酒店。

    参数：
        hotel_id (int): 要取消的酒店的ID。

    返回：
        str: 指示酒店是否成功取消的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店 {hotel_id} 已成功取消。"
    else:
        conn.close()
        return f"未找到ID为 {hotel_id} 的酒店。"

@tool
def search_trip_recommendations(
    location: Optional[str] = None,
    name: Optional[str] = None,
    keywords: Optional[str] = None,
) -> list[dict]:
    """
    根据位置、名称和关键词搜索旅行推荐。

    参数：
        location (Optional[str]): 旅行推荐的位置。默认为None。
        name (Optional[str]): 旅行推荐的名称。默认为None。
        keywords (Optional[str]): 与旅行推荐相关的关键词。默认为None。

    返回：
        list[dict]: 符合搜索条件的旅行推荐字典列表。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

@tool
def book_excursion(recommendation_id: int) -> str:
    """
    通过推荐ID预订游览。

    参数：
        recommendation_id (int): 要预订的旅行推荐的ID。

    返回：
        str: 指示旅行推荐是否成功预订的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"旅行推荐 {recommendation_id} 已成功预订。"
    else:
        conn.close()
        return f"未找到ID为 {recommendation_id} 的旅行推荐。"

@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    通过ID更新旅行推荐的详情。

    参数：
        recommendation_id (int): 要更新的旅行推荐的ID。
        details (str): 旅行推荐的新详情。

    返回：
        str: 指示旅行推荐是否成功更新的消息。
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"旅行推荐 {recommendation_id} 已成功更新。"
    else:
        conn.close()
        return f"未找到ID为 {recommendation_id} 的旅行推荐。"

@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully cancelled."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# 创建一个可以调用工具的节点，如果调用失败，就自动执行备用错误处理逻辑 handle_tool_error，并把异常信息保存到状态字段 error 里。
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """推入或弹出状态。"""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]


from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "请给出真实的输出。")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

##它是一个 Pydantic 模型类（BaseModel），
##在 LangGraph / LangChain 体系中，这样的类就表示一个“工具（Tool）”。
##换句话说：模型（LLM）在推理时，如果决定“我该结束任务”或“我无法处理”，它就会调用这个工具，向主助理发出“切换控制权”的信号。
class CompleteOrEscalate(BaseModel):
    """用于标记当前任务完成和/或将对话控制权上报给主助手的工具,
    主助手可以根据用户需求重新路由对话。"""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "用户改变了对当前任务的想法。",
            },
            "example 2": {
                "cancel": True,
                "reason": "我已经完全完成了任务。",
            },
            "example 3": {
                "cancel": False,
                "reason": "我需要搜索用户的邮件或日历以获取更多信息。",
            },
        }


llm = ChatOpenAI(model="gpt-4o", temperature=1)
# 航班预订助手
flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是处理航班更新的专业助手。"
            " 主助手在用户需要帮助更新预订时将工作委托给你。"
            "与客户确认更新的航班详情并告知任何额外费用。"
            " 搜索时要坚持。如果首次搜索没有结果,扩大查询范围。"
            "如果你需要更多信息或客户改变主意,将任务上报回主助手。"
            " 记住,只有在成功使用相关工具后,预订才算完成。"
            "\n\n当前用户航班信息:\n<Flights>\n{user_info}\n</Flights>"
            "\n当前时间: {time}。"
            "\n\n如果用户需要帮助,但你的工具都不适用,"
            ' 那么"CompleteOrEscalate"对话到主助手。不要浪费用户时间。不要编造无效的工具或函数。',
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

update_flight_safe_tools = [search_flights]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools
update_flight_runnable = flight_booking_prompt | llm.bind_tools(
    update_flight_tools + [CompleteOrEscalate]
)

# 酒店预订助手
book_hotel_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是处理酒店预订的专业助手。"
            "主助手在用户需要帮助预订酒店时将工作委托给你。"
            "根据用户偏好搜索可用酒店并与客户确认预订详情。"
            " 搜索时要坚持。如果首次搜索没有结果,扩大查询范围。"
            "如果你需要更多信息或客户改变主意,将任务上报回主助手。"
            " 记住,只有在成功使用相关工具后,预订才算完成。"
            "\n当前时间: {time}。"
            '\n\n如果用户需要帮助,但你的工具都不适用,那么"CompleteOrEscalate"对话到主助手。'
            " 不要浪费用户时间。不要编造无效的工具或函数。"
            "\n\n一些你应该CompleteOrEscalate的例子:\n"
            " - '这个时节天气怎么样?'\n"
            " - '算了我想我会分开预订'\n"
            " - '我需要弄清楚在那里的交通'\n"
            " - '哦等等我还没订机票我先订机票'\n"
            " - '酒店预订已确认'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_hotel_safe_tools = [search_hotels]
book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools
book_hotel_runnable = book_hotel_prompt | llm.bind_tools(
    book_hotel_tools + [CompleteOrEscalate]
)

## 租车助手
book_car_rental_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是处理租车预订的专业助手。"
            "主助手在用户需要帮助预订租车时将工作委托给你。"
            "根据用户偏好搜索可用租车并与客户确认预订详情。"
            " 搜索时要坚持。如果首次搜索没有结果,扩大查询范围。"
            "如果你需要更多信息或客户改变主意,将任务上报回主助手。"
            " 记住,只有在成功使用相关工具后,预订才算完成。"
            "\n当前时间: {time}。"
            "\n\n如果用户需要帮助,但你的工具都不适用,"
            '"CompleteOrEscalate"对话到主助手。不要浪费用户时间。不要编造无效的工具或函数。'
            "\n\n一些你应该CompleteOrEscalate的例子:\n"
            " - '这个时节天气怎么样?'\n"
            " - '有什么航班可用?'\n"
            " - '算了我想我会分开预订'\n"
            " - '哦等等我还没订机票我先订机票'\n"
            " - '租车预订已确认'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_car_rental_safe_tools = [search_car_rentals]
book_car_rental_sensitive_tools = [
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools
book_car_rental_runnable = book_car_rental_prompt | llm.bind_tools(
    book_car_rental_tools + [CompleteOrEscalate]
)

##旅游项目助手
book_excursion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是处理旅行推荐的专业助手。"
            "主助手在用户需要帮助预订推荐旅行时将工作委托给你。"
            "根据用户偏好搜索可用的旅行推荐并与客户确认预订详情。"
            "如果你需要更多信息或客户改变主意,将任务上报回主助手。"
            " 搜索时要坚持。如果首次搜索没有结果,扩大查询范围。"
            " 记住,只有在成功使用相关工具后,预订才算完成。"
            "\n当前时间: {time}。"
            '\n\n如果用户需要帮助,但你的工具都不适用,那么"CompleteOrEscalate"对话到主助手。不要浪费用户时间。不要编造无效的工具或函数。'
            "\n\n一些你应该CompleteOrEscalate的例子:\n"
            " - '算了我想我会分开预订'\n"
            " - '我需要弄清楚在那里的交通'\n"
            " - '哦等等我还没订机票我先订机票'\n"
            " - '旅游项目预订已确认!'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_excursion_safe_tools = [search_trip_recommendations]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools
book_excursion_runnable = book_excursion_prompt | llm.bind_tools(
    book_excursion_tools + [CompleteOrEscalate]
)

# 主助手
#我可以调用一个名为 ToFlightBookingAssistant 的工具，并传入 request 参数来把任务交给航班助手。
class ToFlightBookingAssistant(BaseModel):
    """将工作转移到专业助手以处理航班更新和取消。"""

    request: str = Field(
        description="航班更新助手在继续之前需要澄清的任何必要的后续问题。"
    )


class ToBookCarRental(BaseModel):
    """将工作转移到专业助手以处理租车预订。"""

    location: str = Field(
        description="用户想要租车的地点。"
    )
    start_date: str = Field(description="租车的开始日期。")
    end_date: str = Field(description="租车的结束日期。")
    request: str = Field(
        description="用户关于租车的任何额外信息或要求。"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "巴塞尔",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "我需要一辆带自动变速器的紧凑型车。",
            }
        }


class ToHotelBookingAssistant(BaseModel):
    """将工作转移到专业助手以处理酒店预订。"""

    location: str = Field(
        description="用户想要预订酒店的地点。"
    )
    checkin_date: str = Field(description="酒店的入住日期。")
    checkout_date: str = Field(description="酒店的退房日期。")
    request: str = Field(
        description="用户关于酒店预订的任何额外信息或要求。"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "苏黎世",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "我更喜欢靠近市中心的酒店,房间要有景观。",
            }
        }


class ToBookExcursion(BaseModel):
    """将工作转移到专业助手以处理旅行推荐和其他旅游项目预订。"""

    location: str = Field(
        description="用户想要预订推荐旅行的地点。"
    )
    request: str = Field(
        description="用户关于旅行推荐的任何额外信息或要求。"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "卢塞恩",
                "request": "用户对户外活动和风景优美的景色感兴趣。",
            }
        }


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是瑞士航空公司的有用客户支持助手。"
            "你的主要职责是搜索航班信息和公司政策以回答客户查询。"
            "如果客户要求更新或取消航班、预订租车、预订酒店或获取旅行推荐,"
            "通过调用相应工具将任务委托给适当的专业助手。你自己无法进行这些类型的更改。"
            " 只有专业助手被授权为用户执行此操作。"
            "用户不知道不同的专业助手,所以不要提及它们;只需通过函数调用悄悄委托。"
            "向客户提供详细信息,并在得出信息不可用的结论之前始终再次检查数据库。"
            " 搜索时要坚持。如果首次搜索没有结果,扩大查询范围。"
            " 如果搜索结果为空,在放弃之前扩大搜索范围。"
            "\n\n当前用户航班信息:\n<Flights>\n{user_info}\n</Flights>"
            "\n当前时间: {time}。",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]

assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToFlightBookingAssistant,
        ToBookCarRental,
        ToHotelBookingAssistant,
        ToBookExcursion,
    ]
)


from typing import Callable
from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"助手现在是{assistant_name}。回顾主助手和用户之间的上述对话。"
                    f" 用户的意图未得到满足。使用提供的工具来帮助用户。记住,你是{assistant_name},"
                    " 只有在你成功调用适当的工具之后,预订、更新或其他操作才算完成。"
                    " 如果用户改变主意或需要其他任务的帮助,调用CompleteOrEscalate函数让主助手接管控制。"
                    " 不要提及你是谁 - 只需作为助手的代理。",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node

from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

# 航班预订助手
builder.add_node(
    "enter_update_flight",
    create_entry_node("航班更新和预订助手", "update_flight"),
)
builder.add_node("update_flight", Assistant(update_flight_runnable))
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node(
    "update_flight_sensitive_tools",
    create_tool_node_with_fallback(update_flight_sensitive_tools),
)
builder.add_node(
    "update_flight_safe_tools",
    create_tool_node_with_fallback(update_flight_safe_tools),
)


def route_update_flight(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "update_flight_safe_tools"
    return "update_flight_sensitive_tools"


builder.add_edge("update_flight_sensitive_tools", "update_flight")
builder.add_edge("update_flight_safe_tools", "update_flight")
builder.add_conditional_edges(
    "update_flight",
    route_update_flight,
    ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END],
)


# 此节点将被所有专业助手共享用于退出
def pop_dialog_state(state: State) -> dict:
    """弹出对话栈并返回主助手。

    这让完整图谱明确跟踪对话流程并将控制委托给特定子图。
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # 注意:当前不处理LLM执行并行工具调用的边缘情况
        messages.append(
            ToolMessage(
                content="恢复与主助手的对话。请回顾过去的对话并根据需要帮助用户。",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")



# 租车助手
builder.add_node(
    "enter_book_car_rental",
    create_entry_node("租车助手", "book_car_rental"),
)
builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
builder.add_edge("enter_book_car_rental", "book_car_rental")
builder.add_node(
    "book_car_rental_safe_tools",
    create_tool_node_with_fallback(book_car_rental_safe_tools),
)
builder.add_node(
    "book_car_rental_sensitive_tools",
    create_tool_node_with_fallback(book_car_rental_sensitive_tools),
)


def route_book_car_rental(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in book_car_rental_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "book_car_rental_safe_tools"
    return "book_car_rental_sensitive_tools"


builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
builder.add_conditional_edges(
    "book_car_rental",
    route_book_car_rental,
    [
        "book_car_rental_safe_tools",
        "book_car_rental_sensitive_tools",
        "leave_skill",
        END,
    ],
)

# 酒店预订助手
builder.add_node(
    "enter_book_hotel", create_entry_node("酒店预订助手", "book_hotel")
)
builder.add_node("book_hotel", Assistant(book_hotel_runnable))
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node(
    "book_hotel_safe_tools",
    create_tool_node_with_fallback(book_hotel_safe_tools),
)
builder.add_node(
    "book_hotel_sensitive_tools",
    create_tool_node_with_fallback(book_hotel_sensitive_tools),
)


def route_book_hotel(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_hotel_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_hotel_safe_tools"
    return "book_hotel_sensitive_tools"


builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
builder.add_edge("book_hotel_safe_tools", "book_hotel")
builder.add_conditional_edges(
    "book_hotel",
    route_book_hotel,
    ["leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", END],
)

# 旅游项目助手
builder.add_node(
    "enter_book_excursion",
    create_entry_node("旅行推荐助手", "book_excursion"),
)
builder.add_node("book_excursion", Assistant(book_excursion_runnable))
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_node(
    "book_excursion_safe_tools",
    create_tool_node_with_fallback(book_excursion_safe_tools),
)
builder.add_node(
    "book_excursion_sensitive_tools",
    create_tool_node_with_fallback(book_excursion_sensitive_tools),
)


def route_book_excursion(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_excursion_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_excursion_safe_tools"
    return "book_excursion_sensitive_tools"


builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
builder.add_edge("book_excursion_safe_tools", "book_excursion")
builder.add_conditional_edges(
    "book_excursion",
    route_book_excursion,
    ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END],
)

# 主助手
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)
def route_primary_assistant(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("无效的路由")


# 助手可以路由到委托助手之一,
# 直接使用工具,或直接响应用户
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_update_flight",
        "enter_book_car_rental",
        "enter_book_hotel",
        "enter_book_excursion",
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# 每个委托工作流都可以直接响应用户
# 当用户响应时,我们希望返回到当前活动的工作流
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
]:
    """如果我们处于委托状态,直接路由到相应的助手。"""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# 编译图谱
memory = InMemorySaver()
part_4_graph = builder.compile(
    checkpointer=memory,
    # 让用户批准或拒绝使用敏感工具
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ],
)

import shutil
import uuid

# 使用备份文件更新,这样我们可以在每个部分从原始位置重新开始
db = update_dates(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # passenger_id 用于我们的航班工具
        # 获取用户的航班信息
        "passenger_id": "3442 587242",
        # 检查点通过 thread_id 访问
        "thread_id": thread_id,
    }
}

_printed = set()
# 我们可以重用第1部分的教程问题来看看它的表现


import shutil
import uuid

# 创建用户可能与助手进行的示例对话
tutorial_questions = [
    "您好，我的航班是什么时间？",
    # "我可以将航班更新为更早的时间吗？我想今天晚些时候离开。",
    # "那么将我的航班更新为下周某个时间",
    # "下一个可用选项很好",
    # "住宿和交通怎么样？",
    # "是的，我想要一个经济实惠的酒店，住一周（7天）。我还想租一辆车。",
    # "好的，您能为您推荐的酒店预订吗？听起来不错。",
    # "是的，请继续预订任何费用适中且有空房的酒店。",
    # "现在关于汽车，我有什么选择？",
    # "太棒了，让我们选择最便宜的选项。请预订7天",
    # "很好，现在您对游览有什么建议？",
    # "我在那里的时候有这些活动吗？",
    # "有趣 - 我喜欢博物馆，有什么选择？",
    # "好的，很好，选择一个并为我在那里的第二天预订。",
]


for question in tutorial_questions:
    events = part_4_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = part_4_graph.get_state(config)
    while snapshot.next:
        # 我们遇到了中断!代理正在尝试使用工具,用户可以批准或拒绝它
        # 注意:此代码都在图谱外部。通常,你会将输出流式传输到UI。
        # 然后,当用户提供输入时,你会通过API调用触发新的运行。
        try:
            user_input = input(
                "你是否批准上述操作?输入 'y' 继续;"
                " 否则,解释你请求的更改。\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # 只是继续
            result = part_4_graph.invoke(
                None,
                config,
            )
        else:
            # 通过提供有关请求更改/改变主意的说明来满足工具调用
            result = part_4_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API调用被用户拒绝。原因:'{user_input}'。继续提供帮助,考虑用户的输入。",
                        )
                    ]
                },
                config,
            )
        snapshot = part_4_graph.get_state(config)