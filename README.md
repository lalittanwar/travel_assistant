# Travel Planning Assistant

An AI travel assistant that combines a **RAG knowledge base** (destination facts:
attractions, neighbourhoods, transport, culture, itineraries) with **custom
MCP tools** for live, time-sensitive information (weather, currency
conversion), orchestrated by a LangChain/LangGraph agent with multi-turn
memory.

---

## 1. Architecture

```
                         ┌──────────────────────┐
                         │     Streamlit UI     │  app.py
                         │   (chat interface)   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │   LangGraph Agent      │  agent.py
                         │  (create_react_agent)  │
                         │  + MemorySaver         │  (multi-turn memory)
                         │  + system prompt       │
                         └─────┬──────────┬───────┘
                               │          │
                 ┌─────────────┘          └─────────────┐
                 │                                       │
     ┌───────────▼──────────────┐              ┌────────────▼──────────────┐
     │  RAG tool                │              │  MCP tools (via           │
     │  search_singapore_       │              │  MultiServerMCPClient)    │
     │  knowledge_base          │              │                           │
     │                          │              │  ├─ get_singapore_        │
     │  Chroma vector store     │              │  │  forecast              │
     │  (sentence-transformers  │              │  │  → Open-Meteo API      │
     │   embeddings)            │              │  │                        │
     └───────────┬──────────────┘              │  └─ convert_currency      │
                 │                             │     → Frankfurter API     │
     ┌───────────▼────────────────┐            └───────────────────────────┘
     │  knowledge_base/content/   │              (weather_server.py,
     │  (Wikivoyage + Visit       │               currency_server.py —
     │   Singapore, scraped)      │               custom MCP servers, stdio
     └────────────────────────────┘               transport)
```


---

## 2. Knowledge base sources

| Source | Covers |
|---|---|
| Wikivoyage Singapore Travel Guide | Districts, attractions, transport, food, itineraries |
| Visit Singapore: Essential Travel Information | Climate, language, connectivity, practical tips |
| Visit Singapore: Sample Itineraries | Itinerary ideas by duration/traveller profile |
| Visit Singapore: Things to Do | Attractions/activities by interest category |

Full metadata (title/URL) is in `knowledge_base/sources.json`. Raw scraped
text lives in `knowledge_base/content/*.md`, each tagged with a
`TITLE / URL / RETRIEVED` header so retrieved chunks can be cited back to
their source.

---

## 3. RAG workflow

1. `knowledge_base/scrape_sources.py` fetches each URL (fast HTTP first,
   Playwright headless-browser fallback for JS-rendered pages), strips
   navigation/ads/scripts, and saves clean article text per source.
2. `rag/ingest.py` loads those files, splits them into ~800-character
   chunks (100-char overlap) via `RecursiveCharacterTextSplitter`, embeds
   them with `sentence-transformers/all-MiniLM-L6-v2`, and persists them
   into a local **Chroma** vector store.
3. At query time, `agent.py` wraps the Chroma retriever as a LangChain
   `@tool` (`search_singapore_knowledge_base`) that returns the top-k
   chunks along with their source title/URL, so the agent can cite them.

---

## 4. MCP tools

Both are custom-written MCP servers (not pre-built/marketplace servers) —
see `mcp_servers/weather_server.py` and `mcp_servers/currency_server.py`.

| Tool | Server | 
|---|---|
| `get_singapore_forecast(days)` | `weather_server.py` | 
| `convert_currency(amount, from_currency, to_currency)` | `currency_server.py` |

Both use the v1 `FastMCP` API (`mcp<2` pinned — see Troubleshooting) and
run over **stdio transport**, launched as subprocesses by
`MultiServerMCPClient` in `agent.py`. Both return an `"error"` key instead
of fabricating data if the underlying HTTP call fails.

---

## 5. Prompt & context strategy

The system prompt in `agent.py` (`SYSTEM_PROMPT`) instructs the model to:
- Use the RAG tool exclusively for destination facts, and cite sources.
- Use MCP tools only for time-sensitive data (weather, currency) — never
  for anything the knowledge base already covers.
- Never fabricate destination facts, forecasts, or exchange rates; state
  plainly when information is unavailable or a tool call failed.
- Label combined answers by origin: knowledge-base fact vs. live tool data
  vs. the model's own recommendation.
- Preserve user preferences (budget, dates, traveller type) across turns.

Multi-turn memory is handled by LangGraph's `MemorySaver` checkpointer,
keyed on a `thread_id` that stays fixed for the duration of one
conversation (reset in the UI via the sidebar button, which issues a new
`thread_id`).

---

## 6. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- `git` (to clone/manage the repo)

---

## 7. Step-by-step setup

### 7.1 Clone and create a virtual environment

```bash
git clone https://github.com/lalittanwar/travel_assistant
cd travel-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 7.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 7.3 Pull the local LLM

```bash
ollama pull llama3.1:8b
```

Confirm Ollama's background service is running:
```bash
curl http://localhost:11434
# should print: Ollama is running
```
If not, start it with `ollama serve` in a separate terminal, or open the
Ollama desktop app.

### 7.4 Build the knowledge base content

If `knowledge_base/content/*.md` files aren't already populated:

```bash
cd knowledge_base
python scrape_sources.py
```

Check each file afterward — sites like visitsingapore.com render via
JavaScript, so the script falls back to Playwright automatically. If a
file still looks empty/short, follow the manual-paste fallback in
`knowledge_base/instructions.md`.

```bash
cd ..
```

### 7.5 Ingest the knowledge base into Chroma

```bash
python rag/ingest.py
```

This downloads the embedding model on first run and persists the vector
store to `rag/chroma_db/`. You should see a sanity-check search result
printed at the end.

### 7.6 (Optional) Test each MCP server standalone

```bash
cd mcp_servers
mcp dev weather_server.py
# In the Inspector UI: Tools tab → get_singapore_forecast → days=3 → Run Tool

mcp dev currency_server.py
# Tools tab → convert_currency → amount=50000, from_currency=INR, to_currency=SGD → Run Tool
cd ..
```

### 7.7 Run the full app (Streamlit UI)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Try the required combined scenario:

> "Create a three-day Singapore itinerary for next week and adjust it
> according to the weather forecast."

Each response shows a "Sources used" caption indicating which tool(s)
fired (knowledge base / weather MCP / currency MCP).

---

## 8. Sample questions to demo

| Type | Question |
|---|---|
| RAG only | What are the must-visit attractions in Singapore? |
| RAG only | Suggest activities for a family with children. |
| MCP only | What's the weather forecast for the next 3 days? |
| MCP only | Convert INR 50,000 to SGD. |
| Combined | Create a three-day Singapore itinerary for next week and adjust it for the weather forecast. |
| Combined | I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary. |
| Multi-turn | "I'm planning a trip with two kids" → "How many kids am I planning with?" |

---


## 9. Project folder structure

```
travel-assistant/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                      # Streamlit UI
├── agent.py                    # Agent: RAG tool + MCP tools + system prompt
├── test_agent_cli.py           # CLI test harness
├── raw_mcp_client_demo.py      # Raw MCP client (protocol demonstration)
├── knowledge_base/
│   ├── sources.json
│   ├── scrape_sources.py
│   ├── instructions.md
│   └── content/*.md
├── mcp_servers/
│   ├── weather_server.py
│   └── currency_server.py
└── rag/
    ├── ingest.py
    └── chroma_db/               # generated, gitignored
```
