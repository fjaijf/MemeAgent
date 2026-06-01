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

## How This Maps To TradingAgents

In `TradingAgents`, model creation happens in:

- `tradingagents/graph/trading_graph.py`
- `tradingagents/llm_clients/factory.py`

Then each agent node calls the model with one of these patterns:

- `llm.invoke(...)`
- `prompt | llm.bind_tools(tools)`
- `llm.with_structured_output(...)`

This scaffold starts with the simplest pattern first: `llm.invoke(...)`.
