import datetime
import pytz
from typing import Annotated, List, Dict, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Опис стану системи ---

def add_expenses_reducer(prev: List[Dict], new: List[Dict]) -> List[Dict]:
    """Редуктор: додає нові витрати до існуючого списку."""
    return (prev or []) + (new or [])

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Історія діалогу
    expenses: Annotated[list[Dict], add_expenses_reducer] # Внутрішній фінансовий журнал
    monthly_limit: float                                  # Ліміт витрат

# --- Інструменти ---

@tool
def add_expense_tool(amount: float, category: str, description: str) -> str:
    """Додає нову витрату до фінансового журналу. Викликай для запису покупок/витрат."""
    return f"Система: Витрату {amount} грн додано у категорію '{category}' ({description})."

@tool
def total_spent(state: Annotated[dict, InjectedState]) -> str:
    """Рахує загальну суму всіх витрат з фінансового журналу."""
    expenses = state.get("expenses", [])
    total = sum(float(e.get("amount", 0)) for e in expenses)
    return f"Загальна сума витрат: {total:.2f} грн."

@tool
def summary_by_category(state: Annotated[dict, InjectedState]) -> str:
    """Повертає зведену таблицю витрат по кожній категорії."""
    expenses = state.get("expenses", [])
    if not expenses:
        return "Жодних витрат ще не зафіксовано."

    totals = {}
    for e in expenses:
        cat = (e.get("category") or "інше").lower()
        totals[cat] = totals.get(cat, 0.0) + float(e.get("amount", 0))

    lines = [f"- {cat.capitalize()}: {amt:.2f} грн" for cat, amt in totals.items()]
    return "Витрати по категоріях:\n" + "\n".join(lines)

@tool
def get_current_datetime() -> str:
    """Зовнішній інструмент: отримує поточну дату та час для фіксації в логах."""
    kyiv_tz = pytz.timezone('Europe/Kyiv')
    return datetime.datetime.now(kyiv_tz).strftime("%Y-%m-%d %H:%M")

tools_list = [add_expense_tool, total_spent, summary_by_category, get_current_datetime]
tool_node = ToolNode(tools_list)

# --- Налаштування вузлів ---

def agent_node(state: AgentState):
    """Основний вузол, який приймає рішення та оновлює фінансовий стан."""
    messages = state["messages"]
    limit = state.get("monthly_limit", 5000.0)

    sys_msg = SystemMessage(
        content=f"Ти фінансовий асистент 'BudgetGraph'. Допомагай вести облік витрат. "
                f"Встановлений місячний ліміт: {limit} грн. "
                f"Будь лаконічним. Аналізуй витрати, коли просять."
    )

    # Ініціалізація моделі (API ключ підтягнеться автоматично з середовища/secrets)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    llm_with_tools = llm.bind_tools(tools_list)

    response = llm_with_tools.invoke([sys_msg] + messages)

    new_expenses = []
    if hasattr(response, "tool_calls"):
        for tc in response.tool_calls:
            if tc["name"] == "add_expense_tool":
                args = tc["args"]
                new_expenses.append({
                    "amount": float(args.get("amount", 0)),
                    "category": args.get("category", "інше"),
                    "description": args.get("description", "")
                })

    return {"messages": [response], "expenses": new_expenses}

def budget_monitor_node(state: AgentState):
    """Додатковий вузол (moderation): перевіряє ліміти перед відповіддю."""
    expenses = state.get("expenses", [])
    limit = state.get("monthly_limit", 5000.0)
    total = sum(float(e.get("amount", 0)) for e in expenses)

    # Перевіряємо, чи ми вже попереджали користувача в останніх повідомленнях
    last_msgs_content = " ".join([str(m.content) for m in state["messages"][-2:]])

    if total > limit and "Увага" not in last_msgs_content:
        warning_msg = AIMessage(content=f"\n\nУвага: Ваші загальні витрати ({total:.2f} грн) перевищили місячний ліміт ({limit:.2f} грн)!")
        return {"messages": [warning_msg]}

    return {}

# --- Маршрутизація та збірка графа ---

def route_after_agent(state: AgentState) -> str:
    """Визначає наступний крок ReAct циклу."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return "budget_monitor"

def create_agent_graph():
    """Створює та компілює граф."""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("budget_monitor", budget_monitor_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "budget_monitor": "budget_monitor"}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("budget_monitor", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

# Експортуємо готовий граф для використання в app.py
graph = create_agent_graph()
