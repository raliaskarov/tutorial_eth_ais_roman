# MCP server for web search tasks with the following tools:
# - search.query
#
# Simple web search via DuckDuckGo Instant Answer API (no API key needed).

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json, time, urllib.parse, urllib.request
from ddgs import DDGS

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search-server")

ROOT = Path(__file__).resolve().parent.parent  # project root


# --- helpers ---

def _ts() -> float:
    """Return current timestamp in seconds."""
    return time.time()

def _ddg_instant_answer(q: str) -> Dict[str, Any]:
    """Call DuckDuckGo Instant Answer JSON API."""
    base = "https://api.duckduckgo.com/"
    url = f"{base}?{urllib.parse.urlencode({'q': q, 'format': 'json', 'no_redirect': 1, 'no_html': 1})}"
    req = urllib.request.Request(url, headers={"User-Agent": "agentic-ai-notebook/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}

def _ddg_web(q: str, max_results: int) -> List[Dict[str, str]]:
    """Unofficial DuckDuckGo web results via ddgs."""
    items = []
    with DDGS() as ddg:
        for r in ddg.text(q, max_results=max_results):
            # r has keys: title, href, body
            items.append({
                "title": (r.get("title") or "").strip()[:160] or r.get("href",""),
                "url": (r.get("href") or "").strip(),
                "snippet": (r.get("body") or "").strip()
            })
            if len(items) >= max_results:
                break
    return items

def _extract_results(ddg: Dict[str, Any], top_k: int, q: str) -> List[Dict[str, str]]:
    """Extract results from DuckDuckGo Instant Answer JSON."""
    if "_error" in ddg:
        return []
    results = []

    # 1) Abstract (even without a URL)
    abs_txt = (ddg.get("AbstractText") or "").strip()
    abs_heading = (ddg.get("Heading") or "").strip()
    abs_url = (ddg.get("AbstractURL") or "").strip()
    if abs_txt:
        results.append({
            "title": abs_heading or "Summary",
            "url": abs_url or f"https://duckduckgo.com/?{urllib.parse.urlencode({'q': q})}",
            "snippet": abs_txt
        })

    # 2) Results
    for item in ddg.get("Results") or []:
        title = (item.get("Text") or "").strip()[:160]
        url = (item.get("FirstURL") or "").strip()
        if url:
            results.append({"title": title or url, "url": url, "snippet": ""})

    # 3) RelatedTopics (flat or nested)
    for rt in ddg.get("RelatedTopics") or []:
        topics = rt.get("Topics") if isinstance(rt, dict) else None
        iter_items = topics or [rt]
        for t in iter_items:
            title = (t.get("Text") or "").strip()[:160]
            url = (t.get("FirstURL") or "").strip()
            if url:
                results.append({"title": title or url, "url": url, "snippet": ""})

    # Dedup & trim
    seen, out = set(), []
    for r in results:
        u = r.get("url", "")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
        if len(out) >= top_k:
            break
    return out


# --------- tools ---------

@mcp.tool("search.query")
def search_query(q: str, top_k: int = 5, site: Optional[str] = None, provider: str = "auto") -> Dict[str, Any]:
    """
    Web search with multiple providers:
      provider: 'auto' | 'ddgs' | 'ddg_ia'
    """
    q = (q or "").strip()
    if not q:
        return {"query": q, "top_k": 0, "results": [], "provider": provider, "ts": _ts(), "note": "empty query"}
    if site:
        q = f"site:{site} {q}"
    top_k = max(1, min(int(top_k), 10))

    used = provider
    results = []
    error = None

    try:
        if provider in ("auto", "ddgs"):
            results = _ddg_web(q, top_k)
            used = "duckduckgo_web(ddgs)"
    except Exception as e:
        error = f"ddgs error: {e}"

    if not results and provider in ("auto", "ddg_ia"):
        ddg = _ddg_instant_answer(q)
        results = _extract_results(ddg, top_k, q)
        used = "duckduckgo_instant_answer"
        if ddg.get("_error"):
            error = ddg["_error"]

    return {"query": q, "top_k": top_k, "results": results, "provider": used, "ts": _ts(), "error": error}

if __name__ == "__main__":
    mcp.run(transport="stdio")
