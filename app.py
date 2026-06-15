import streamlit as st
import pandas as pd
import time
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Імпортуємо нашого агента
from agent import graph

# Налаштування сторінки та CSS
st.set_page_config(page_title="BudgetGraph AI", page_icon="💰", layout="wide")

st.markdown("""
<style>
    /* Мінімалістичний та чистий дизайн */
    .stApp { background-color: #f8f9fa; }
    .stChatInput { padding-bottom: 20px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; color: #2c3e50; }
    .css-1d391kg { padding-top: 1rem; }
    .budget-warning { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Ініціалізація стану (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Привіт! Я твій фінансовий AI-асистент BudgetGraph. Розкажи про свої сьогоднішні витрати або запитай статистику."}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4()) # Унікальний ID для пам'яті LangGraph
if "expenses" not in st.session_state:
    st.session_state.expenses = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Бічна панель (Sidebar)
with st.sidebar:
    st.title("⚙️ Налаштування")
    
    # Вибір режиму роботи
    app_mode = st.radio("Режим роботи:", ["AI Асистент (LangGraph)", "Звичайний чат (Gemini)"])
    
    st.markdown("---")
    # Параметри та ліміти
    monthly_limit = st.number_input("💵 Місячний ліміт (грн):", min_value=1000.0, value=5000.0, step=500.0)
    temperature = st.slider("🌡️ Температура генерації:", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    st.markdown("---")
    # Очищення історії
    if st.button("🗑️ Очистити історію"):
        st.session_state.messages = [{"role": "assistant", "content": "Історію очищено. Почнемо з чистого аркуша!"}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.expenses = []
        st.rerun()
        
    # Експорт історії
    chat_history_str = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
    st.download_button(
        label="💾 Завантажити історію чату",
        data=chat_history_str,
        file_name="chat_history.json",
        mime="application/json"
    )

    st.markdown("---")
    st.info("ℹ️ **Про застосунок:** BudgetGraph аналізує ваші текстові повідомлення, автоматично витягує суми та категорії витрат, і контролює бюджет")

# Спеціалізовані UI-компоненти (Фінансовий Дашборд)
st.title("💰 BudgetGraph: Твій фінансовий трекер")

# Синхронізація ліміту з графом
graph.update_state(config, {"monthly_limit": monthly_limit})

# Отримуємо актуальні витрати з пам'яті графа
current_state = graph.get_state(config).values
st.session_state.expenses = current_state.get("expenses", [])

total_spent = sum(float(e.get("amount", 0)) for e in st.session_state.expenses)
budget_percentage = min(total_spent / monthly_limit, 1.0)

# Створюємо вкладки
tab_chat, tab_analytics = st.tabs(["💬 Чат з асистентом", "📊 Детальна аналітика"])

# Вкладка 1: Аналітика
with tab_analytics:
    st.subheader("Ваша фінансова картина")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Витрачено загалом", value=f"{total_spent:.2f} ₴")
        st.progress(budget_percentage, text=f"Використано {int(budget_percentage * 100)}% ліміту")
        if total_spent > monthly_limit:
            st.markdown(f"<p class='budget-warning'>⚠️ Перевищення ліміту на {(total_spent - monthly_limit):.2f} ₴!</p>", unsafe_allow_html=True)

    with col2:
        if st.session_state.expenses:
            df = pd.DataFrame(st.session_state.expenses)
            df_grouped = df.groupby('category')['amount'].sum().reset_index()
            st.bar_chart(df_grouped.set_index('category'))
        else:
            st.info("Додайте першу витрату в чаті, щоб побачити графік.")

    if st.session_state.expenses:
        st.markdown("**Останні транзакції:**")
        st.dataframe(pd.DataFrame(st.session_state.expenses), use_container_width=True, hide_index=True)


# Вкладка 2: Чат
with tab_chat:
    # Відображення історії чату
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Генератор для імітації красивого посимвольного стрімінгу
    def stream_text(text, delay=0.01):
        for char in text:
            yield char
            time.sleep(delay)

    # Обробка нового повідомлення
    if prompt := st.chat_input("Напиши витрату (напр., 'Купив каву за 60 грн') або задай питання..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Аналізую..."):
                if app_mode == "Звичайний чат (Gemini)":
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
                    lc_messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
                    
                    full_response = ""
                    for chunk in llm.stream(lc_messages):
                        full_response += chunk.content
                    st.write_stream(stream_text(full_response))
                    
                else:
                    full_response = ""
                    for event in graph.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                        for node_name, node_state in event.items():
                            if node_state and "messages" in node_state:
                                last_msg = node_state["messages"][-1]
                                if isinstance(last_msg, AIMessage) and last_msg.content:
                                    content = last_msg.content
                                    if isinstance(content, list):
                                        content = "".join([b.get("text", "") for b in content if isinstance(b, dict) and "text" in b])
                                    if isinstance(content, str) and content.strip():
                                        full_response += content + "\n\n"
                    
                    if full_response.strip():
                        st.write_stream(stream_text(full_response.strip()))
                    else:
                        fallback = "Дані збережено. Перегляньте вкладку 'Детальна аналітика'."
                        st.write_stream(stream_text(fallback))
                        full_response = fallback

            st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
            st.rerun()