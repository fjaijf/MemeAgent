# MemeAgent

`MemeAgent` is a minimal agent scaffold modeled after the way `TradingAgents`
creates and calls LLMs:

1. Load config
2. Create an LLM client
3. Build an agent object
4. Call `llm.invoke(...)` inside the agent

## Files

- `memeagent/config.py`: reads environment configuration
- `memeagent/llm.py`: creates the model client
- `memeagent/agent.py`: contains the actual agent logic
- `memeagent/search_agent.py`: retrieves public web and news context
- `memeagent/workflow.py`: coordinates retrieval + meme analysis
- `main.py`: command-line entrypoint

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your API key.

3. Run:

```bash
python main.py --topic "PEPE"
```

To search the public web first and feed the results into the analysis:

```bash
python main.py --topic "PEPE" --search --show-search
```

The search agent currently gathers:

- public web search results
- public news results

The default search provider is `ddgs`, which does not require an API key:

```bash
MEMEAGENT_SEARCH_PROVIDER=ddgs
```

For a more stable all-web search API, use Brave Search:

```bash
MEMEAGENT_SEARCH_PROVIDER=brave
MEMEAGENT_SEARCH_API_KEY=your_brave_search_api_key
```

Or use Tavily, which is designed for agent/RAG search workflows:

```bash
MEMEAGENT_SEARCH_PROVIDER=tavily
MEMEAGENT_SEARCH_API_KEY=your_tavily_api_key
```

If the selected search API needs a local HTTP proxy, set:

```bash
MEMEAGENT_SEARCH_PROXY=http://127.0.0.1:7890
```

## How This Maps To TradingAgents

In `TradingAgents`, model creation happens in:

- `tradingagents/graph/trading_graph.py`
- `tradingagents/llm_clients/factory.py`

Then each agent node calls the model with one of these patterns:

- `llm.invoke(...)`
- `prompt | llm.bind_tools(tools)`
- `llm.with_structured_output(...)`

This scaffold starts with the simplest pattern first: `llm.invoke(...)`.
