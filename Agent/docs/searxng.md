# SearXNG for MemeAgent

SearXNG is useful when DDGS is unstable or when you want one local search
endpoint that can aggregate multiple engines for MemeAgent.

## Local Docker Deploy

The official project recommends Docker Compose for container deployment. Create
a SearXNG deployment directory:

```powershell
mkdir searxng
mkdir searxng\core-config
cd searxng
```

Download the current compose template and environment example:

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/searxng/searxng/master/container/docker-compose.yml -OutFile docker-compose.yml
Invoke-WebRequest https://raw.githubusercontent.com/searxng/searxng/master/container/.env.example -OutFile .env.example
Copy-Item .env.example .env
```

Edit `.env` and set local development values:

```env
SEARXNG_HOSTNAME=localhost
LETSENCRYPT_EMAIL=
```

Then create or edit `core-config/settings.yml` and enable JSON output:

```yaml
use_default_settings: true

search:
  formats:
    - html
    - json
```

For local-only development, expose SearXNG on port `8888` in
`docker-compose.yml` if the template does not already expose a local port:

```yaml
ports:
  - "127.0.0.1:8888:8080"
```

Start it:

```powershell
docker compose up -d
```

Check JSON search from PowerShell:

```powershell
Invoke-RestMethod "http://localhost:8888/search?q=this%20is%20fine&format=json"
```

If this returns HTTP 403, JSON output is still disabled in `settings.yml`.

## MemeAgent Config

Use SearXNG as the only provider:

```env
MEMEAGENT_SEARCH_PROVIDER=searxng
MEMEAGENT_SEARXNG_URL=http://localhost:8888
MEMEAGENT_SEARXNG_WEB_CATEGORIES=general
MEMEAGENT_SEARXNG_NEWS_CATEGORIES=news
```

Optionally pin engines that your SearXNG instance has enabled:

```env
MEMEAGENT_SEARXNG_ENGINES=google,bing,brave,duckduckgo
```

You can also combine it with other providers:

```env
MEMEAGENT_SEARCH_PROVIDER=searxng,zhihu
```

Run a provider-only connectivity check:

```powershell
python test_search_connectivity.py --providers searxng --query "this is fine meme"
```

Run MemeAgent with SearXNG for retrieval:

```powershell
python main.py --topic "this is fine" --force-search --show-search --search-provider searxng
```

## Notes

- `MEMEAGENT_SEARXNG_URL` can be a base URL such as `http://localhost:8888`, a
  path-mounted URL such as `http://localhost:8888/searxng`, or a direct
  `/search` URL.
- If your SearXNG URL already contains query parameters, MemeAgent preserves
  them and adds `q`, `format=json`, `language`, `categories`, and optional
  `engines`.
- Keep public SearXNG instances out of heavy automated workloads. A local
  instance is more reliable and avoids surprising rate limits.
