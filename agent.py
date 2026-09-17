"""
The core Travel Assistant agent.

Combines three tools:
  1. search_singapore_knowledge_base  - RAG over destination content (Chroma)
  2. get_singapore_forecast           - MCP tool (weather_server.py)
  3. convert_currency                 - MCP tool (currency_server.py)

Usage:
    import asyncio
    from agent import build_agent

    async def main():
        agent, cleanup = await build_agent()
        response = await agent.ainvoke({"messages": [("user", "What's the weather in Singapore?")]})
        print(response["messages"][-1].content)
        await cleanup()

    asyncio.run(main())
"""

import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

BASE_DIR = Path(__file__).parent
RAG_DIR = BASE_DIR / "rag"
MCP_SERVERS_DIR = BASE_DIR / "mcp_servers"
CHROMA_DIR = str(RAG_DIR / "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a travel planning assistant for Singapore.

You have three tools:

1. search_singapore_knowledge_base — use this for ANY question about
   destination facts: attractions, neighbourhoods, transportation, culture,
   food, itineraries, indoor/outdoor activity suggestions. This is your ONLY
   source for destination facts. Always cite the source_title returned by
   this tool when you use its results.

2. get_singapore_forecast — use this ONLY for weather / forecast questions,
   or when a request asks you to adjust an itinerary based on weather
   (e.g. "next week", "tomorrow", "is rain expected"). Never guess a
   forecast from memory.

3. convert_currency — use this ONLY for currency conversion questions
   (e.g. converting a travel budget between two currencies). Never guess
   an exchange rate from memory.

Rules you must follow:
- Never use an MCP tool (weather or currency) to answer a question the
  knowledge base already covers.
- Never invent destination facts, weather data, or exchange rates. If the
  knowledge base has no relevant content for a destination question, say so
  explicitly instead of fabricating an answer.
- If a tool call fails or returns an "error" key, tell the user plainly
  that the information is currently unavailable — do not fabricate a
  substitute answer.
- When you combine knowledge-base content with live tool data (e.g. a
  weather-adjusted itinerary), clearly label which parts of your answer
  are: (a) knowledge-base facts, (b) live data from a tool, and
  (c) your own recommendation/suggestion.
- Preserve relevant user preferences and details (dates, budget, traveller
  type, prior answers) across the conversation.
- Keep answers structured and clear — use short sections or a day-by-day
  breakdown for itineraries.
"""


def build_rag_tool():
    """Wrap the Chroma retriever as a LangChain tool the agent can call."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def search_singapore_knowledge_base(query: str) -> str:
        """
        Search the Singapore travel knowledge base for destination facts:
        attractions, neighbourhoods, transportation, culture, food, and
        sample itineraries. Always use this for destination questions
        instead of relying on your own knowledge.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the knowledge base for this query."

        formatted = []
        for d in docs:
            title = d.metadata.get("source_title", "Unknown source")
            url = d.metadata.get("source_url", "")
            formatted.append(f"[Source: {title} | {url}]\n{d.page_content}")
        return "\n\n---\n\n".join(formatted)

    return search_singapore_knowledge_base


async def build_agent(llm=None):
    """
    Builds and returns (agent, cleanup_fn).
    Call `await cleanup_fn()` when you're done to close MCP connections.

    If llm is not provided, defaults to a local Ollama model.
    """
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="llama3.1", temperature=0.2)

    rag_tool = build_rag_tool()

    mcp_client = MultiServerMCPClient(
        {
            "weather": {
                "command": sys.executable,
                "args": [str(MCP_SERVERS_DIR / "weather_server.py")],
                "transport": "stdio",
            },
            "currency": {
                "command": sys.executable,
                "args": [str(MCP_SERVERS_DIR / "currency_server.py")],
                "transport": "stdio",
            },
        }
    )

    mcp_tools = await mcp_client.get_tools()
    all_tools = [rag_tool] + mcp_tools

    print(f"Loaded {len(all_tools)} tools: {[t.name for t in all_tools]}")

    checkpointer = MemorySaver()  # in-memory multi-turn conversation state
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    async def cleanup():
        # MultiServerMCPClient manages its own subprocess lifecycles;
        # nothing extra to close explicitly in current versions, but kept
        # here as an extension point if that changes.
        pass

    return agent, cleanup
