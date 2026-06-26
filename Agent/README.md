# MemeAgent

`MemeAgent` is a minimal agent scaffold for meme studies and online discourse
analysis, modeled after the way `TradingAgents` creates and calls LLMs:

1. Load config
2. Create an LLM client
3. Build an agent object
4. Call `llm.invoke(...)` inside the agent

## Files

- `memeagent/config.py`: reads environment configuration
- `memeagent/llm.py`: creates the model client
- `memeagent/agent.py`: contains the actual agent logic
- `memeagent/search_agent.py`: retrieves public web and news context
- `memeagent/retrieve_cli.py`: standalone retrieval CLI without LLM calls
- `memeagent/workflow.py`: coordinates retrieval + meme analysis
- `main.py`: command-line entrypoint
- `retrieve.py`: command-line entrypoint for retrieval-only tuning

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install MemeAgent as a local terminal command:

```bash
cd D:\自研Agent\MemeAgent\Agent
pip install -e .
```

After installation, run it from any terminal with either command:

```bash
meme --topic "PEPE" --search --stream
memeagent --topic "PEPE" --search --show-search
```

2. Copy `.env.example` to `.env` and fill in your API key.

3. Run:

```bash
python main.py --topic "PEPE"
```

Before development, run a quick smoke test for the LLM and search provider:

```bash
python test_smoke.py
```

You can test only one side if needed:

```bash
python test_smoke.py --skip-search
python test_smoke.py --skip-llm
```

If final analysis times out on image-heavy runs, increase the timeout or limit
generation length:

```bash
MEMEAGENT_TIMEOUT=180
MEMEAGENT_MAX_TOKENS=1200
```

To use GLM as a separate controller for planning, retrieval decisions, and
retrieval reflection while keeping the primary model for vision/final analysis:

```bash
MEMEAGENT_PROVIDER=openai-compatible
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEMEAGENT_MODEL=qwen3.7-plus
DASHSCOPE_API_KEY=your_dashscope_api_key

MEMEAGENT_CONTROLLER_PROVIDER=glm
MEMEAGENT_CONTROLLER_MODEL=glm-5.1
MEMEAGENT_CONTROLLER_THINKING=true
ZAI_API_KEY=your_zai_api_key
```

To run the agent with locally deployed Hugging Face models while keeping the
remote GLM/Qwen API configuration available, set the main multimodal model and
controller model separately:

```bash
MEMEAGENT_PROVIDER=local-transformers
MEMEAGENT_MODEL=/data/ggbond/Qwen3.6-27B
MEMEAGENT_LOCAL_THINKING=false
MEMEAGENT_LOCAL_STRIP_THINKING=true
MEMEAGENT_LOCAL_MAX_NEW_TOKENS=2048

MEMEAGENT_CONTROLLER_PROVIDER=local-transformers
MEMEAGENT_CONTROLLER_MODEL=/data/ggbond/Qwen3-32B
MEMEAGENT_CONTROLLER_THINKING=true
```

Local mode loads models through Hugging Face Transformers. Install the optional
dependencies before using it:

```bash
pip install -e ".[local]"
# or
pip install -r requirements-local.txt
```

To allow MemeAgent to decide whether public web/news retrieval is needed:

```bash
python main.py --topic "PEPE" --search --show-search
```

To bypass that retrieval-needed decision and force web/news retrieval:

```bash
python main.py --topic "PEPE" --force-search --show-search
```

To tune retrieval without spending tokens on planning, vision, or final
analysis, use the retrieval-only entrypoint:

```bash
python retrieve.py --topic "PEPE" --mode plan
python retrieve.py --topic "PEPE" --mode both
python retrieve.py --query '"this is fine"' --mode web
python retrieve.py --topic "PEPE" --context-file notes.txt --json --output runs/retrieval.json
```

If installed with `pip install -e .`, the same tool is available as:

```bash
meme-retrieve --topic "PEPE" --mode both
```

The CLI shows a terminal status panel and live activity indicator while it waits
for image pre-analysis, retrieval, and final LLM analysis. For plain text output
in logs or batch jobs, add:

```bash
python main.py --topic "PEPE" --search --show-search --plain
```

To stream the final analysis as it is generated, add:

```bash
python main.py --topic "PEPE" --search --stream
```

Streaming uses plain incremental text by default so Markdown does not jump or
re-render while the model is still generating. If you prefer live Markdown
rendering, add:

```bash
python main.py --topic "PEPE" --search --stream --stream-markdown
```

MemeAgent supports three input modes:

```bash
# Text only: search from topic/context, then analyze.
python main.py --topic "a meme name or phrase" --search --show-search

# Image only: describe the image, search from extracted OCR/keywords, then analyze.
python main.py --image path/to/meme.png --search --show-search

# Text and image: combine the text hint with image-derived keywords.
python main.py --topic "a meme name or phrase" --image path/to/meme.png --search --show-search
```

## Evaluation Script

For harmful meme detection benchmarks, use the dedicated evaluator:

```bash
python evaluate_harmful_memes.py --dataset D:\自研Agent\Dataset\label_test.json --mode direct --resume
```

Common options:

```bash
python evaluate_harmful_memes.py --dataset D:\自研Agent\Dataset\label_test.json --schema auto --workers 4
python evaluate_harmful_memes.py --dataset D:\自研Agent\Dataset\label_test.json --mode workflow --search
python evaluate_harmful_memes.py --dataset D:\自研Agent\Dataset\label_test.json --limit 100 --workers 4 --output runs/eval.jsonl
```

The script writes per-sample predictions to JSONL and a summary metric file
with accuracy, precision, recall, and F1.

When images are attached, MemeAgent first asks the vision model to produce an
OCR/visual-description/keyword report. That image-derived report is then used
as additional search context before the final meme analysis:

```bash
python main.py --image path/to/meme.png --search --show-search
```

When `--search` is enabled, MemeAgent first asks the LLM whether external
retrieval is needed. It can skip web/news retrieval when the input and local
memory are already sufficient. When `--force-search` is enabled, MemeAgent skips
that decision and directly plans retrieval queries before searching. The stable
OCR/visual-anchor queries remain the primary search inputs; the LLM plan can add
only a few extra web/news queries when the input provides concrete people,
events, OCR phrases, platforms, or background clues.

The search agent currently gathers research context for:

- public web search results
- public news results
- meme harmfulness and risk analysis
- sentiment, audience reception, and intent recognition
- meme evolution and cross-platform discourse tracking

The default search provider is `ddgs`, which does not require an API key:

```bash
MEMEAGENT_SEARCH_PROVIDER=ddgs
```

You can combine providers with commas. For example, use DDGS and Zhihu together:

```bash
MEMEAGENT_SEARCH_PROVIDER=ddgs,zhihu
MEMEAGENT_ZHIHU_API_KEY=your_zhihu_access_secret
```

To use Tavily, which is designed for agent/RAG search workflows:

```bash
MEMEAGENT_SEARCH_PROVIDER=tavily
MEMEAGENT_TAVILY_API_KEY=your_tavily_api_key
```

To use Zhihu's official search API:

```bash
MEMEAGENT_SEARCH_PROVIDER=zhihu
MEMEAGENT_ZHIHU_API_KEY=your_zhihu_access_secret
```

To use Anspire Search:

```bash
MEMEAGENT_SEARCH_PROVIDER=anspire
MEMEAGENT_ANSPIRE_API_KEY=your_anspire_api_key
```

The Anspire demo runs the same retrieve pipeline with `search_provider=anspire`
without changing your default provider:

```bash
python retrieve_anspire_demo.py --topic "meme sentiment analysis" --max-results 3
```

To use GLM/Zhipu's web search API:

```bash
MEMEAGENT_SEARCH_PROVIDER=glm
ZAI_API_KEY=your_zai_api_key
MEMEAGENT_GLM_SEARCH_ENGINE=search_pro
MEMEAGENT_GLM_SEARCH_RECENCY_FILTER=noLimit
MEMEAGENT_GLM_SEARCH_CONTENT_SIZE=medium
```

Optional GLM domain filtering:

```bash
MEMEAGENT_GLM_SEARCH_DOMAIN_FILTER=www.sohu.com
```

The GLM search provider uses the `zhipuai` Python package. Install it if you
select `MEMEAGENT_SEARCH_PROVIDER=glm`:

```bash
pip install zhipuai
```

If the selected search API needs a local HTTP proxy, set:

```bash
MEMEAGENT_SEARCH_PROXY=http://127.0.0.1:7890
```

Thread/Page Context fetching can use a proxy only when direct page fetching
fails with connection, timeout, or retryable HTTP errors:

```bash
MEMEAGENT_CONTEXT_PROXY=http://127.0.0.1:7890
```

If `MEMEAGENT_CONTEXT_PROXY` is not set, it falls back to
`MEMEAGENT_SEARCH_PROXY`.

## How This Maps To TradingAgents

In `TradingAgents`, model creation happens in:

- `tradingagents/graph/trading_graph.py`
- `tradingagents/llm_clients/factory.py`

Then each agent node calls the model with one of these patterns:

- `llm.invoke(...)`
- `prompt | llm.bind_tools(tools)`
- `llm.with_structured_output(...)`

This scaffold starts with the simplest pattern first: `llm.invoke(...)`.
