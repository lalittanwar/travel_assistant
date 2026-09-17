"""
Streamlit UI for the Travel Planning Assistant.

Run:
    streamlit run app.py
"""

import asyncio
import threading
import uuid

import streamlit as st
from langchain_core.messages import AIMessage

from agent import build_agent

st.set_page_config(
    page_title="Travel Assistant",
    page_icon="🇸🇬",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Agent setup (built once, cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_background_loop():
    """
    A dedicated background thread that owns one event loop for the whole
    app session. Streamlit reruns each execute in a fresh thread, so a
    loop merely cached and re-driven from whatever thread happens to call
    it (e.g. via run_until_complete) still breaks async resources tied to
    subprocesses (our MCP servers) or persistent HTTP clients (Ollama).
    Running the loop forever in one fixed thread, and submitting work to
    it via run_coroutine_threadsafe, keeps every async resource bound to
    a single consistent thread for the app's lifetime.
    """
    loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    return loop


def run_async(coro):
    """Submit a coroutine to the shared background loop and wait for the result."""
    loop = get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


@st.cache_resource(show_spinner="Starting up the travel assistant")
def get_agent():
    agent, _cleanup = run_async(build_agent())
    return agent


TOOL_LABELS = {
    "search_singapore_knowledge_base": "📚 Knowledge base",
    "get_singapore_forecast": "🌦️ Live weather (MCP)",
    "convert_currency": "💱 Live currency conversion (MCP)",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "content": str, "tools_used": [str]}
    st.session_state.display_messages = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🇸🇬 About this assistant")
    st.markdown(
        "Ask about destinations, weather, or currency conversion. "
        "The assistant combines:\n"
        "- A **knowledge base** for destination facts\n"
        "- Live **MCP tools** for weather (Open-Meteo) and currency (Frankfurter)"
    )

    st.divider()
    st.subheader("Try asking")
    st.markdown(
        "- *What are the must-visit attractions in Singapore?*\n"
        "- *What's the weather forecast for the next 3 days?*\n"
        "- *Convert INR 50,000 to SGD*\n"
        "- *Create a 3-day itinerary and adjust it for the weather*"
    )

    st.divider()
    if st.button("🔄 Reset conversation"):
        st.session_state.display_messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("Travel Planning Assistant")

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            badges = " ".join(f"`{TOOL_LABELS.get(t, t)}`" for t in msg["tools_used"])
            st.caption(f"Sources used: {badges}")

user_input = st.chat_input("Ask about travel...")

if user_input:
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent = get_agent()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                response = run_async(
                    agent.ainvoke({"messages": [("user", user_input)]}, config=config)
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

            all_messages = response["messages"]
            final_answer = all_messages[-1].content

            # Figure out which tools were actually called this turn
            tools_used = []
            for m in all_messages:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        name = tc.get("name") if isinstance(tc, dict) else tc.name
                        if name and name not in tools_used:
                            tools_used.append(name)

            st.markdown(final_answer)
            if tools_used:
                badges = " ".join(f"`{TOOL_LABELS.get(t, t)}`" for t in tools_used)
                st.caption(f"Sources used: {badges}")

    st.session_state.display_messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "tools_used": tools_used,
        }
    )
